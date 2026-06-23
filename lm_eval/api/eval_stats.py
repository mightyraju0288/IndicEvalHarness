"""
Statistical rigor for LLM evaluation aggregation.

Implements the recommendations from:
  [1] Miller, "Adding Error Bars to Evals" (arXiv:2411.00640)
  [2] Bowyer et al., "Don't Use the CLT ... Fewer Than a Few Hundred Datapoints"
      (arXiv:2503.01747)
  [3] Hochlehnert et al., "A Sober Look at Progress in LM Reasoning"
      (arXiv:2504.07086)  -- motivation: report variance, not point estimates.

Design: pure functions over numpy arrays of per-question scores. No dependency
on lm_eval internals, so this file is unit-testable in isolation. The harness
calls `summarize()` / `compare()` from evaluator_utils.

Conventions
-----------
scores      : 1-D array, one score per question (float; binary or fractional).
clusters    : 1-D array of cluster ids, same length as scores. For BhashaBench
              a cluster = one source question and all its language variants.
score_matrix: 2-D array (n_questions, K) when K resamples per question exist.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats as _sps

# n below which CLT-based intervals are unreliable per [2]. Not a hard error;
# we attach a warning flag so the harness can surface it in results JSON.
CLT_MIN_N = 100


# --------------------------------------------------------------------------- #
# z / t critical values
# --------------------------------------------------------------------------- #
def z_crit(confidence: float = 0.95) -> float:
    """Two-sided z critical value, e.g. 1.96 for 95%."""
    return float(_sps.norm.ppf(1 - (1 - confidence) / 2))


def t_crit(df: int, confidence: float = 0.95) -> float:
    """Two-sided Student-t critical value. Used for small n instead of z."""
    return float(_sps.t.ppf(1 - (1 - confidence) / 2, df))


# --------------------------------------------------------------------------- #
# Point estimates
# --------------------------------------------------------------------------- #
def mean_score(scores: np.ndarray) -> float:
    return float(np.mean(scores))


def sample_sd(scores: np.ndarray) -> float:
    # ddof=1 -> unbiased sample standard deviation
    return float(np.std(scores, ddof=1))


# --------------------------------------------------------------------------- #
# Standard errors
# --------------------------------------------------------------------------- #
def clt_standard_error(scores: np.ndarray) -> float:
    """CLT SE of the sample mean: s / sqrt(n). Valid only when questions IID."""
    n = len(scores)
    if n < 2:
        return float("nan")
    return sample_sd(scores) / math.sqrt(n)


def bernoulli_standard_error(scores: np.ndarray) -> float:
    """Closed-form SE for binary scores: sqrt(p(1-p)/n).

    Only valid when scores are truly in {0,1}; raises otherwise so callers
    don't silently misuse it on fractional metrics (f1, BLEU, partial credit).
    """
    uniq = np.unique(scores)
    if not np.all(np.isin(uniq, (0.0, 1.0))):
        raise ValueError("bernoulli_standard_error requires binary {0,1} scores")
    n = len(scores)
    p = float(np.mean(scores))
    return math.sqrt(p * (1 - p) / n)


def clustered_standard_error(scores: np.ndarray, clusters: np.ndarray) -> float:
    """Cluster-adjusted SE [1].

    sqrt( SE_clt^2 + (1/n^2) * sum_c sum_{i in c} sum_{j!=i in c}
            (s_i - s_bar)(s_j - s_bar) )

    Interpolates between: every cluster = 1 effective observation (perfect
    intra-cluster correlation) and the IID CLT case (zero correlation).
    For BhashaBench, clusters are translation groups of one source question.
    """
    scores = np.asarray(scores, dtype=float)
    clusters = np.asarray(clusters)
    n = len(scores)
    s_bar = float(np.mean(scores))
    se_clt_sq = clt_standard_error(scores) ** 2

    cross = 0.0
    for c in np.unique(clusters):
        r = scores[clusters == c] - s_bar
        # sum_{i!=j} r_i r_j = (sum r)^2 - sum r^2
        cross += r.sum() ** 2 - np.sum(r ** 2)

    var_hat = se_clt_sq + cross / (n ** 2)
    # numerical floor: clustered var can dip slightly negative on tiny samples
    return math.sqrt(max(var_hat, 0.0))


# --------------------------------------------------------------------------- #
# Confidence intervals
# --------------------------------------------------------------------------- #
def ci(mean: float, se: float, n: int, confidence: float = 0.95,
       use_t: bool = True) -> tuple[float, float]:
    """Symmetric CI. Uses t for small n (more honest than z), z otherwise."""
    if math.isnan(se):
        return (float("nan"), float("nan"))
    crit = t_crit(n - 1, confidence) if (use_t and n > 1) else z_crit(confidence)
    return (mean - crit * se, mean + crit * se)


# --------------------------------------------------------------------------- #
# Variance reduction: resampling (K samples per question)
# --------------------------------------------------------------------------- #
def resampled_question_means(score_matrix: np.ndarray) -> np.ndarray:
    """Per-question mean over K resamples. Shrinks within-question variance 1/K."""
    return np.mean(np.asarray(score_matrix, dtype=float), axis=1)


def within_question_variance(score_matrix: np.ndarray) -> float:
    """E[sigma_i^2] estimate. Needs K>=2 (else 0 -- can't estimate)."""
    m = np.asarray(score_matrix, dtype=float)
    if m.ndim != 2 or m.shape[1] < 2:
        return 0.0
    return float(np.mean(np.var(m, axis=1, ddof=1)))


# --------------------------------------------------------------------------- #
# Single-model summary
# --------------------------------------------------------------------------- #
@dataclass
class ModelSummary:
    mean: float
    n: int
    num_clusters: int
    se_clt: float
    ci_clt: tuple[float, float]
    se_cluster: Optional[float]
    ci_cluster: Optional[tuple[float, float]]
    se: float  # the SE we recommend reporting (clustered if clusters present)
    ci_reported: tuple[float, float]
    is_binary: bool
    small_n_warning: bool
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        for k in ("ci_clt", "ci_cluster", "ci_reported"):
            if d[k] is not None:
                d[k] = list(d[k])
        return d


def summarize(scores, clusters=None, confidence: float = 0.95) -> ModelSummary:
    """Full uncertainty summary for one model on one task."""
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    notes: list[str] = []

    is_binary = bool(np.all(np.isin(np.unique(scores), (0.0, 1.0))))
    mean = mean_score(scores)
    se_clt = clt_standard_error(scores)
    ci_clt = ci(mean, se_clt, n, confidence)

    if clusters is not None:
        clusters = np.asarray(clusters)
        num_clusters = int(len(np.unique(clusters)))
        se_cluster = clustered_standard_error(scores, clusters)
        # CI df for clustered SE uses (#clusters - 1), not (n-1)
        crit = t_crit(max(num_clusters - 1, 1), confidence)
        ci_cluster = (mean - crit * se_cluster, mean + crit * se_cluster)
        se, ci_reported = se_cluster, ci_cluster
        if num_clusters < n:
            notes.append(
                f"clustered SE reported ({num_clusters} clusters over {n} "
                f"questions); CLT SE would understate uncertainty"
            )
    else:
        num_clusters = n
        se_cluster, ci_cluster = None, None
        se, ci_reported = se_clt, ci_clt

    small_n = n < CLT_MIN_N
    if small_n:
        notes.append(
            f"n={n} < {CLT_MIN_N}: CLT/normal CIs are overconfident [Bowyer 2025]; "
            f"treat interval as a lower bound on true uncertainty"
        )
    if is_binary and mean in (0.0, 1.0):
        notes.append(
            "degenerate accuracy (all-correct or all-wrong): SE collapses to 0; "
            "interval is not meaningful, use a Bayesian/Wilson interval"
        )

    return ModelSummary(
        mean=mean, n=n, num_clusters=num_clusters,
        se_clt=se_clt, ci_clt=ci_clt,
        se_cluster=se_cluster, ci_cluster=ci_cluster,
        se=se, ci_reported=ci_reported,
        is_binary=is_binary, small_n_warning=small_n, notes=notes,
    )


# --------------------------------------------------------------------------- #
# Model comparison
# --------------------------------------------------------------------------- #
@dataclass
class Comparison:
    diff: float
    se_diff: float
    ci_diff: tuple[float, float]
    significant: bool          # CI excludes 0
    correlation: Optional[float]
    method: str                # "paired" | "unpaired"
    n: int
    notes: list[str] = field(default_factory=list)


def compare(scores_a, scores_b, clusters=None, paired: bool = True,
            confidence: float = 0.95) -> Comparison:
    """Compare two models. Paired (same questions) is strictly better when
    available: per-question score correlation is almost always positive, so
    the paired SE is smaller for free [1]. Significant := CI on the difference
    excludes zero (stricter and correct vs. checking CI overlap)."""
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    notes: list[str] = []

    if paired:
        if len(a) != len(b):
            raise ValueError("paired comparison needs aligned, equal-length scores")
        diffs = a - b
        mean_diff = float(np.mean(diffs))
        if clusters is not None:
            se = clustered_standard_error(diffs, np.asarray(clusters))
            df = len(np.unique(clusters)) - 1
        else:
            se = clt_standard_error(diffs)
            df = len(diffs) - 1
        corr = float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 else float("nan")
        crit = t_crit(max(df, 1), confidence)
        lo, hi = mean_diff - crit * se, mean_diff + crit * se
        method, n = "paired", len(a)
        if not (np.isnan(corr)) and corr <= 0:
            notes.append("non-positive score correlation: paired gives no SE gain here")
    else:
        mean_diff = mean_score(a) - mean_score(b)
        if clusters is not None:
            se_a = clustered_standard_error(a, np.asarray(clusters))
            se_b = clustered_standard_error(b, np.asarray(clusters))
        else:
            se_a, se_b = clt_standard_error(a), clt_standard_error(b)
        se = math.sqrt(se_a ** 2 + se_b ** 2)
        corr = None
        crit = z_crit(confidence)
        lo, hi = mean_diff - crit * se, mean_diff + crit * se
        method, n = "unpaired", min(len(a), len(b))

    return Comparison(
        diff=mean_diff, se_diff=se, ci_diff=(lo, hi),
        significant=not (lo <= 0 <= hi),
        correlation=corr, method=method, n=n, notes=notes,
    )


# --------------------------------------------------------------------------- #
# Power analysis / sample size (Miller [1])
# --------------------------------------------------------------------------- #
def required_sample_size(delta: float, omega_sq: float,
                         sigma_a_sq: float = 0.0, sigma_b_sq: float = 0.0,
                         K_a: int = 1, K_b: int = 1,
                         alpha: float = 0.05, beta: float = 0.20) -> float:
    """n needed to detect a true mean gap of `delta` with power 1-beta.

    n grows ~1/delta^2 (halving the detectable gap costs 4x questions) and
    shrinks with resampling K. Rearrange for `delta` to get a benchmark's
    minimum detectable effect at fixed n."""
    z_a2 = float(_sps.norm.ppf(1 - alpha / 2))
    z_b = float(_sps.norm.ppf(1 - beta))
    var_term = omega_sq + sigma_a_sq / K_a + sigma_b_sq / K_b
    return (z_a2 + z_b) ** 2 * var_term / (delta ** 2)


def minimum_detectable_effect(n: float, omega_sq: float,
                              sigma_a_sq: float = 0.0, sigma_b_sq: float = 0.0,
                              K_a: int = 1, K_b: int = 1,
                              alpha: float = 0.05, beta: float = 0.20) -> float:
    """Smallest true gap detectable with n questions -- 'is this eval worth running?'"""
    z_a2 = float(_sps.norm.ppf(1 - alpha / 2))
    z_b = float(_sps.norm.ppf(1 - beta))
    var_term = omega_sq + sigma_a_sq / K_a + sigma_b_sq / K_b
    return (z_a2 + z_b) * math.sqrt(var_term / n)


# --------------------------------------------------------------------------- #
# Unbiased pass@k (Chen et al., "Evaluating LLMs Trained on Code", Codex)
# --------------------------------------------------------------------------- #
def pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """Unbiased pass@k for one question.

    Estimator: 1 - C(n-c, k) / C(n, k), computed in the numerically stable
    product form from the Codex paper (avoids overflow in the binomials).
      n = num_samples (must be >= k)
      c = num_correct among those n samples
    Returns the probability that at least one of k draws (without replacement)
    succeeds.
    """
    n, c = num_samples, num_correct
    if n < k:
        raise ValueError(f"pass@k needs n >= k (got n={n}, k={k})")
    if c <= 0:
        return 0.0
    if n - c < k:
        # fewer than k failures -> every k-subset contains a correct sample
        return 1.0
    # 1 - prod_{i=n-c+1}^{n} (1 - k/i)
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def passk_question_scores(score_matrix: np.ndarray, k: int) -> np.ndarray:
    """Map a (n_questions, n_samples) binary matrix to one pass@k per question.

    Output is fractional in [0,1], so feed it to the CLT/clustered SE layer,
    NOT the Bernoulli SE path.
    """
    m = np.asarray(score_matrix, dtype=float)
    n_samples = m.shape[1]
    correct = m.sum(axis=1).astype(int)
    return np.array([pass_at_k(n_samples, int(c), k) for c in correct], dtype=float)
