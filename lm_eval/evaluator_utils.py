import collections
import logging
import math
import pathlib
import sys
from typing import List, Optional, Tuple, Union

from lm_eval.api.eval_stats import passk_question_scores, summarize
from lm_eval.api.group import ConfigurableGroup
from lm_eval.api.metrics import (
    aggregate_subtask_metrics,
    mean,
    pooled_sample_stderr,
    stderr_for_metric,
)
from lm_eval.api.task import Task
from lm_eval.utils import positional_deprecated


def _is_reserved_key(name) -> bool:
    """Reserved per-sample keys use the __name__ pattern (e.g. __cluster__).
    They carry metadata, not metrics, and must never be aggregated/scored."""
    return isinstance(name, str) and name.startswith("__") and name.endswith("__")


eval_logger = logging.getLogger(__name__)


class TaskOutput:
    """
    Wrapper class for Task outputs.It contains various attributes and methods to manage and calculate metrics for the task.

        Attributes:
            task (object): The task object.
            task_name (str): The name of the task.
            task_config (dict): The configuration of the task.
            version (str): The version of the task.
            group_name (str): The name of the task group.
            n_shot (int): The number of shots for the task.
            task_alias (str): The alias of the task.
            group_alias (str): The alias of the task group.
            is_group (bool): Indicates if the task is a group.
            logged_samples (list): The list of logged samples.
            sample_len (int): The length of the samples.
            sample_metrics (defaultdict): The dictionary of samples' metrics.
            agg_metrics (defaultdict): The dictionary of aggregate metrics.

        Methods:
            from_taskdict(cls, task_name: str, task):
                Creates a TaskOutput instance from a task dictionary.

            calculate_aggregate_metric(bootstrap_iters=100000) -> None:
                Calculates the aggregate metrics for the task.
    """

    def __init__(
        self,
        task=None,
        task_name=None,
        task_config=None,
        version=None,
        group_name=None,
        n_shot=None,
        task_alias=None,
        group_alias=None,
        is_group=None,
    ):
        self.task = task
        self.task_config = task_config
        self.task_name = task_name
        self.group_name = group_name
        self.version = version
        self.n_shot = n_shot
        self.task_alias = task_alias
        self.group_alias = group_alias
        self.is_group = is_group
        self.logged_samples = []
        self.sample_len = None
        self.sample_metrics = collections.defaultdict(list)
        self.agg_metrics = collections.defaultdict(list)

    @classmethod
    def from_taskdict(cls, task_name: str, task):
        if isinstance(task, tuple):
            group_name, task = task
        else:
            group_name = None
        if not task:
            # these gets filtered out in get_task_list
            # once they are added to group hierarchy
            is_group = True
            return cls(
                task=task, task_name=task_name, is_group=is_group, group_name=group_name
            )
        version = task.VERSION
        task_config = dict(task.dump_config())
        if (n_shot := task_config.get("num_fewshot")) == 0:
            n_shot = task_config.get("metadata", {}).get("num_fewshot", 0)
        task_alias = task_config.get("alias")
        group_alias = task_config.get("group_alias")
        return cls(
            task=task,
            task_name=task_name,
            task_config=task_config,
            group_name=group_name,
            version=version,
            n_shot=n_shot,
            task_alias=task_alias,
            group_alias=group_alias,
        )

    def _resolve_clusters(self, raw_clusters, n_scores, metric):
        """Decide clustering mode for one metric's per-sample cluster ids.

        all ids present -> clustered ; none present -> IID (return None) ;
        some-but-not-all present -> raise (malformed task; never half-cluster).
        """
        if raw_clusters is None:
            return None
        if len(raw_clusters) != n_scores:
            raise ValueError(
                f"Task '{self.task_name}' metric '{metric}': got "
                f"{len(raw_clusters)} cluster ids for {n_scores} scores; "
                "cluster ids must align 1:1 with scores."
            )
        present = [c is not None for c in raw_clusters]
        if all(present):
            return raw_clusters
        if not any(present):
            return None
        raise ValueError(
            f"Task '{self.task_name}' metric '{metric}': "
            f"{sum(present)}/{len(present)} samples carry a __cluster__ id. "
            "Clustering requires all samples or none to be clustered; refusing "
            "to silently half-cluster. Check the task's cluster_key and data."
        )

    def calculate_aggregate_metric(self, bootstrap_iters=100000, confidence=0.95) -> None:
        # Split reserved per-sample keys (__name__, e.g. __cluster__) out of the
        # metric set: they are metadata and must NEVER be treated as metrics.
        # __cluster__ ids are kept aligned (per filter) with the scores so they
        # can drive clustered standard errors below.
        reserved = collections.defaultdict(dict)
        real_sample_metrics = {}
        for (metric, filter_key), items in self.sample_metrics.items():
            if _is_reserved_key(metric):
                reserved[metric][filter_key] = items
            else:
                real_sample_metrics[(metric, filter_key)] = items
        cluster_by_filter = reserved.get("__cluster__", {})

        for (metric, filter_key), items in real_sample_metrics.items():
            try:
                agg_fn = self.task.aggregation()[metric]
            except KeyError:
                # This is when process results output an arbitrary metric
                # TODO: Handle this better and allow other aggregate functions other than mean.
                agg_fn = mean
            metric_key = f"{metric},{filter_key}"
            self.agg_metrics[metric_key] = agg_fn(items)
            self.sample_len = len(items)  # TODO: same sample size for each metric?
            if isinstance(bootstrap_iters, int):
                stderr_fn = stderr_for_metric(
                    metric=agg_fn,
                    bootstrap_iters=min(bootstrap_iters, 100)
                    if metric in ["bleu", "chrf", "ter"]
                    else bootstrap_iters,
                )
                self.agg_metrics[f"{metric}_stderr,{filter_key}"] = (
                    stderr_fn(items) if (stderr_fn and len(items) > 1) else "N/A"
                )
            else:
                raise ValueError(
                    f"Received bootstrap_iters '{bootstrap_iters}' but expected an integer. Set to 0 to turn off stderr calculations."
                )

            # --- statistical rigor: clustered / CLT uncertainty summary ---
            # Only valid where the aggregate is the per-question mean (acc, f1,
            # exact_match, em, ...). Corpus-level aggregations (chrf/bleu/ter)
            # and perplexity are not means of per-sample scores, so we skip them.
            if agg_fn is mean:
                clusters_arg = self._resolve_clusters(
                    cluster_by_filter.get(filter_key), len(items), metric
                )
                summary = summarize(
                    items, clusters=clusters_arg, confidence=confidence
                )
                # The reported stderr becomes the rigorous SE: clustered when a
                # cluster id is present on every sample, otherwise the CLT SE.
                self.agg_metrics[f"{metric}_stderr,{filter_key}"] = summary.se
                self.agg_metrics[f"{metric}_mean,{filter_key}"] = summary.mean
                self.agg_metrics[f"{metric}_ci_low,{filter_key}"] = summary.ci_reported[0]
                self.agg_metrics[f"{metric}_ci_high,{filter_key}"] = summary.ci_reported[1]
                self.agg_metrics[f"{metric}_n,{filter_key}"] = summary.n
                self.agg_metrics[f"{metric}_clusters,{filter_key}"] = summary.num_clusters
                self.agg_metrics[f"{metric}_stat_warnings,{filter_key}"] = summary.notes

        # --- unbiased pass@k (opt-in via task config `passk: [...]`) ---
        # Uses the per-question (n_questions, K) binary matrix carried by the
        # reserved __passk__ key. For each declared k we compute one per-question
        # pass@k score and feed it through the same summarize() SE/CI path.
        passk_ks = getattr(self.task.config, "passk", None) or []
        passk_by_filter = reserved.get("__passk__", {})
        if passk_ks:
            if not passk_by_filter:
                eval_logger.warning(
                    f"Task '{self.task_name}' declares passk={passk_ks} but no "
                    "per-sample K data was found. pass@k requires repeats>1 and the "
                    "`avg_at_k` filter to keep the K samples per question; skipping."
                )
            for filter_key, rows in passk_by_filter.items():
                # rows: list of per-question binary vectors, each length K
                K = len(rows[0]) if rows else 0
                clusters_raw = cluster_by_filter.get(filter_key)
                for k in passk_ks:
                    if K <= k:
                        eval_logger.warning(
                            f"Task '{self.task_name}' filter '{filter_key}': "
                            f"skipping pass@{k} because K={K} <= k={k} "
                            "(need K > k samples per question)."
                        )
                        continue
                    scores = passk_question_scores(rows, k)
                    clusters_arg = self._resolve_clusters(
                        clusters_raw, len(scores), f"pass@{k}"
                    )
                    summary = summarize(
                        scores, clusters=clusters_arg, confidence=confidence
                    )
                    name = f"pass@{k}"
                    self.agg_metrics[f"{name},{filter_key}"] = summary.mean
                    self.agg_metrics[f"{name}_mean,{filter_key}"] = summary.mean
                    self.agg_metrics[f"{name}_stderr,{filter_key}"] = summary.se
                    self.agg_metrics[f"{name}_ci_low,{filter_key}"] = summary.ci_reported[0]
                    self.agg_metrics[f"{name}_ci_high,{filter_key}"] = summary.ci_reported[1]
                    self.agg_metrics[f"{name}_n,{filter_key}"] = summary.n
                    self.agg_metrics[f"{name}_clusters,{filter_key}"] = summary.num_clusters
                    self.agg_metrics[f"{name}_stat_warnings,{filter_key}"] = summary.notes

    def __repr__(self):
        return (
            f"TaskOutput(task_name={self.task_name}, "
            f"group_name={self.group_name}, "
            f"version={self.version}, "
            f"n_shot={self.n_shot}, "
            f"task_alias={self.task_alias}, "
            f"group_alias={self.group_alias})"
        )


def get_task_list(task_dict: dict) -> List[TaskOutput]:
    outputs = []
    for task_name, task_obj in task_dict.items():
        if isinstance(task_obj, dict):
            _outputs = get_task_list(task_obj)
            outputs.extend(_outputs)
        else:
            task_output = TaskOutput.from_taskdict(task_name, task_obj)
            outputs.append(task_output)

    return outputs


def get_subtask_list(task_dict, task_root=None, depth=0):
    subtask_list = {}
    for group_obj, task_obj in task_dict.items():
        if isinstance(group_obj, ConfigurableGroup):
            # group_name = group_obj.group_name
            group_name = group_obj.group_name
        else:
            group_name = group_obj
        if isinstance(task_obj, dict):
            _subtask_list = get_subtask_list(
                task_obj, task_root=group_name, depth=depth + 1
            )
            if task_root:
                subtask_list.setdefault((task_root, depth), []).extend(
                    [
                        _task
                        for (_task, _depth) in _subtask_list.keys()
                        if (_depth - 1) == depth
                    ]
                )

            subtask_list = {**subtask_list, **_subtask_list}
        else:
            if isinstance(task_obj, ConfigurableGroup):
                # group_or_task_name = task_obj.group_name
                group_or_task_name = task_obj.group_name
            elif isinstance(task_obj, Task):
                # group_or_task_name = task_obj.task_name
                group_or_task_name = task_obj.task_name

            if task_root is None:
                subtask_list.setdefault((group_or_task_name, depth), [])
            else:
                subtask_list.setdefault((task_root, depth), []).append(
                    group_or_task_name
                )

    if depth == 0:
        _subtask_list = {}
        for group_key, task_list in subtask_list.items():
            group_name, depth = group_key
            _subtask_list[group_name] = task_list
        subtask_list = _subtask_list

    return subtask_list


def print_writeout(task) -> None:
    for inst in task.instances:
        # print the prompt for the first few documents
        if inst.doc_id < 1:
            eval_logger.info(
                f"Task: {task}; document {inst.doc_id}; context prompt (starting on next line):\
    \n{inst.args[0]}\n(end of prompt on previous line)\ntarget string or answer choice index (starting on next line):\n{task.doc_to_target(inst.doc)}\n(end of target on previous line)"
            )
            eval_logger.info(f"Request: {str(inst)}")


def get_sample_size(task, limit: Optional[int]) -> Union[int, None]:
    if limit is not None:
        limit = (
            int(math.ceil(len(task.eval_docs) * limit)) if limit < 1.0 else int(limit)
        )
    return limit


def prepare_print_tasks(
    task_dict: dict,
    results: dict,
    task_depth=0,
    group_depth=0,
) -> Tuple[dict, dict]:
    """
    @param task_dict: Dictionary representing the group hierarchy of tasks. Each key is a group name and its
    value is a list of task names.
    @param results: Dictionary containing the results of each task. Each key is a
    group name and its value is a dictionary of task results.
    @param task_depth: The indentation level for printing the task
    hierarchy. Default is 0.
    @param group_depth: The indentation level for printing the group
    hierarchy. Default is 0.
    @return: A tuple of two dictionaries: results_agg and groups_agg. results_agg contains
    aggregated results for each task, and groups_agg contains aggregated results for each group.

    Prepares the task hierarchy and aggregates the results for each task and group recursively for printing.
    """

    def _sort_task_dict(task_dict):
        """
        Helper utility. Sorts the task dict at the current level of the hierarchy based on alphabetized task name.
        Required so that we end up sorting within each sub-header correctly.
        """

        return dict(
            sorted(
                task_dict.items(),
                key=lambda item: item[0].group_name
                if isinstance(item[0], ConfigurableGroup)
                else item[0],
            )
        )

    task_agg = collections.defaultdict(dict)
    group_agg = collections.defaultdict(dict)
    task_dict = _sort_task_dict(task_dict)
    for task_or_group_name, task_or_group_obj in task_dict.items():
        tab_string = " " * task_depth + "- " if task_depth > 0 else ""
        if isinstance(task_or_group_name, ConfigurableGroup):
            # string_name = task_or_group_name.group_name
            name = task_or_group_name.group_name
            from_configurable_group = True
            task_or_group_obj = _sort_task_dict(task_or_group_obj)
        elif isinstance(task_or_group_name, str):
            name = task_or_group_name
            if isinstance(task_or_group_obj, Task):
                # string_name = task_or_group_obj.task_name
                name = task_or_group_obj.task_name
            from_configurable_group = False

        task_agg[name] = results[name].copy()
        if from_configurable_group:
            if task_or_group_name.group_alias is not None:
                alias = task_or_group_name.group_alias
            else:
                alias = task_or_group_name.group
        else:
            if "alias" in task_agg[name]:
                alias = task_agg[name]["alias"]
            else:
                alias = name

        task_agg[name]["alias"] = tab_string + alias
        if "samples" in task_agg[name]:
            task_agg[name].pop("samples")

        if from_configurable_group and (" " not in results[name]):
            group_tab_string = " " * group_depth + "- " if group_depth > 0 else ""
            group_agg[name] = results[name].copy()
            group_agg[name]["alias"] = group_tab_string + alias
            if "samples" in group_agg[name]:
                group_agg[name].pop("samples")

        if isinstance(task_or_group_obj, dict):
            task_depth += 1
            group_depth += 1
            _task_agg, _group_agg = prepare_print_tasks(
                task_or_group_obj, results, task_depth, group_depth
            )
            task_agg = {
                **task_agg,
                **_task_agg,
            }
            group_agg = {**group_agg, **_group_agg}
            task_depth -= 1
            group_depth -= 1
    return task_agg, group_agg


def consolidate_results(
    eval_tasks: List[TaskOutput],
) -> Tuple[dict, dict, dict, dict, dict, dict]:
    """
    @param eval_tasks: list(TaskOutput).
    @return: A tuple containing the consolidated results, samples, configs, versions, and num_fewshot.

    Consolidates the results of multiple evaluation tasks into a single structure.

    The method iterates over each evaluation instance and extracts relevant information to create the consolidated
    results structure. The consolidated results structure has the following properties:

    - results: A defaultdict with task names as keys and dictionaries as values. Each dictionary contains
    metric/filter pairs as keys and corresponding metric values as values. The "alias" key is used to store task
    aliases specified in the task configuration.
    - samples: A defaultdict with task names as keys and lists of log samples as values.
    - configs: A defaultdict with task names as keys and task configurations as values.
    - versions: A defaultdict with task names as keys and task versions as values.
    - num_fewshot: A defaultdict with task names as keys and number of few-shot samples as values.
    - higher_is_better: A defaultdict with task names as keys and indicators of whether higher values are better
    for each metric as values.

    The method then returns the consolidated results, samples, configs, versions, and num_fewshot as a tuple.
    """
    # stores the final result for each task, for each metric/filter pair.
    results = collections.defaultdict(dict)
    # logs info about each document evaluated.
    samples = collections.defaultdict(list)
    # store num-fewshot value per task
    num_fewshot = collections.defaultdict(int)
    # Tracks the YAML configs of all chosen task
    configs = collections.defaultdict(dict)
    # Tracks each task's version.
    versions = collections.defaultdict(dict)
    # Track `higher_is_better` for each metric
    higher_is_better = collections.defaultdict(dict)

    for task_output in eval_tasks:
        if "task_alias" in (task_config := task_output.task_config):
            results[task_output.task_name]["alias"] = task_config["task_alias"]
        else:
            results[task_output.task_name]["alias"] = task_output.task_name
        if group_alias := task_output.group_alias:
            if group_alias not in results and (group_name := task_output.group_name):
                results[group_name]["alias"] = group_alias
        num_fewshot[task_output.task_name] = task_output.n_shot
        configs[task_output.task_name] = task_output.task_config
        versions[task_output.task_name] = task_output.version
        samples[task_output.task_name] = task_output.logged_samples
        higher_is_better[task_output.task_name] = task_output.task.higher_is_better()
        for (metric, filter_key), items in task_output.sample_metrics.items():
            # reserved keys (e.g. __cluster__) are per-sample metadata, never metrics
            if _is_reserved_key(metric):
                continue
            metric_key = f"{metric},{filter_key}"
            results[task_output.task_name][metric_key] = task_output.agg_metrics[
                metric_key
            ]
            results[task_output.task_name]["samples"] = task_output.sample_len
            results[task_output.task_name][f"{metric}_stderr,{filter_key}"] = (
                task_output.agg_metrics[f"{metric}_stderr,{filter_key}"]
            )
            # surface the statistical-rigor family emitted by calculate_aggregate_metric
            for suffix in (
                "_mean",
                "_ci_low",
                "_ci_high",
                "_n",
                "_clusters",
                "_stat_warnings",
            ):
                agg_key = f"{metric}{suffix},{filter_key}"
                if agg_key in task_output.agg_metrics:
                    results[task_output.task_name][agg_key] = task_output.agg_metrics[
                        agg_key
                    ]
        # pass@k metrics are derived (from the reserved __passk__ matrix) and have
        # no sample_metrics entry, so surface them directly from agg_metrics.
        for agg_key, value in task_output.agg_metrics.items():
            if agg_key.startswith("pass@"):
                results[task_output.task_name][agg_key] = value
                results[task_output.task_name]["samples"] = task_output.sample_len
    return results, samples, configs, versions, num_fewshot, higher_is_better


def consolidate_group_results(
    results,
    versions,
    task_dict,
    task_root=None,
    show_group_table=False,
    task_aggregation_list=None,
) -> Tuple[dict, dict, bool, Union[None,]]:
    """
    (Recursively) calculates groups' aggregated metrics and updates the results and versions dictionaries with this info.

    @return: a tuple [results, versions, show_group_table, task_aggregation_list] with formats described below:

    - results: A defaultdict with task names (and, after this function is called, group names of
    groups that perform aggregation) as keys, and dictionaries with "alias" and metric,filter_name pairs as keys.
    - versions: A defaultdict with task names (and, after this function is called, group names of
    groups that perform aggregation) as keys, and float values representing the task or group's version if a version is specified. (defaulting to None).
    - show_group_table: a boolean which is true if there exists a group that requires printing of its aggregated scores in a group table.
    - task_aggregation_list: a defaultdict listing the subtasks to average over to produce a given group's end metric.

    The method then returns the updated results, versions, show_group_table, and task_aggregation_list as a tuple.
    In the top-level invocation of this function, task_aggregation_list is ignored.
    """
    if task_root is None:
        task_root = {}

    if task_aggregation_list is None:
        task_aggregation_list = {}

    for group_or_task, group_or_task_info in task_dict.items():
        # Convert to string
        if isinstance(group_or_task, ConfigurableGroup):
            group_config = group_or_task.config
            group_or_task = group_or_task.group_name
        else:
            group_config = None

        if isinstance(group_or_task_info, Task):
            if task_root:
                task_aggregation_list.setdefault(task_root, []).append(
                    group_or_task_info.task_name
                )
        else:
            (
                results,
                versions,
                show_group_table,
                _task_aggregation_list,
            ) = consolidate_group_results(
                results,
                versions,
                group_or_task_info,
                group_or_task,
                show_group_table,
                task_aggregation_list,
            )
            if task_root:
                task_aggregation_list.setdefault(task_root, []).extend(
                    task_aggregation_list.get(group_or_task, [])
                )

            if (group_config is None) or (
                group_config["aggregate_metric_list"] is None
            ):
                results[group_or_task][" "] = " "
                continue

            if "aggregate_metric_list" in group_config:
                agg_metric_list = group_config["aggregate_metric_list"]

            show_group_table = show_group_table | bool(
                group_config["aggregate_metric_list"]
            )

            task_list = _task_aggregation_list[group_or_task]

            metric_list = list(
                {
                    key
                    for task in task_list
                    for key in results[task].keys()
                    if "_stderr" not in key and key not in ["task", "alias", "samples"]
                }
            )
            for metric in metric_list:
                stderr = "_stderr,".join(metric.split(","))

                # gather metrics, sizes, and stderrs from subtasks
                metrics = [
                    results[task][metric]
                    for task in task_list
                    if metric in results[task]
                ]  # TODO: copy?
                stderrs = [
                    results[task][stderr]
                    for task in task_list
                    if stderr in results[task]
                ]
                sizes = [
                    results[task]["samples"]
                    for task in task_list
                    if metric in results[task]
                ]

                for metric_config in agg_metric_list:
                    for filter_name in metric_config["filter_list"]:
                        if metric != ",".join([metric_config["metric"], filter_name]):
                            continue

                        # compute group's pooled metric and stderr
                        if metric_config["aggregation"] == "mean":
                            aggregate_fn = aggregate_subtask_metrics
                        elif callable(metric_config["aggregation"]):
                            aggregate_fn = metric_config["aggregation"]
                        else:
                            raise ValueError(
                                f"Currently, only 'mean' is supported for automatically aggregating scores across groups' subtasks. Got '{metric_config['aggregation']}' for group '{group_or_task}'"
                            )

                        results[group_or_task][metric] = aggregate_fn(
                            metrics,
                            sizes,
                            metric_config["weight_by_size"],
                        )
                        # TODO: calculate groups' metrics using arbitrary agg fns
                        if "N/A" in stderrs:
                            results[group_or_task][stderr] = "N/A"
                        else:
                            # NOTE: this assumes we are using the mean to aggregate. There are warnings about this elsewhere
                            results[group_or_task][stderr] = pooled_sample_stderr(
                                stderrs, sizes
                            )

                results[group_or_task]["samples"] = sum(sizes)
                group_metadata = group_config.get("metadata", None)
                if group_metadata is not None:
                    versions[group_or_task] = group_metadata.get("version", None)
    # print(results)
    return results, versions, show_group_table, task_aggregation_list


@positional_deprecated
def find_test_root(start_path: pathlib.Path) -> pathlib.Path:
    """
    Search upward in the directory tree to a maximum of three layers
    to find and return the package root (containing the 'tests' folder)
    """
    cur_path = start_path.resolve()
    max_layers = 3
    for _ in range(max_layers):
        if (cur_path / "tests" / "test_version_stable.py").exists():
            return cur_path
        else:
            cur_path = cur_path.parent.resolve()
    raise FileNotFoundError(
        f"Unable to find package root within {max_layers} upwards" + f"of {start_path}"
    )


@positional_deprecated
def run_task_tests(task_list: List[str]):
    """
    Find the package root and run the tests for the given tasks
    """
    import pytest

    package_root = find_test_root(start_path=pathlib.Path(__file__))
    task_string = " or ".join(task_list)
    args = [
        f"{package_root}/tests/test_version_stable.py",
        f"--rootdir={package_root}",
        "-k",
        f"{task_string}",
    ]
    sys.path.append(str(package_root))
    pytest_return_val = pytest.main(args)
    if pytest_return_val:
        raise ValueError(
            f"Not all tests for the specified tasks ({task_list}) ran successfully! Error code: {pytest_return_val}"
        )
