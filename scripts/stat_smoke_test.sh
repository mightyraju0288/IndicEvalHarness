#!/usr/bin/env bash
#
# Statistical-rigor smoke test for the IndicEvalHarness fork.
# Run from the ROOT of this repo (the branch with the eval_stats features),
# on the GPU box. Exercises: acc/acc_norm logprob + rigorous SE family,
# clustered SE (cluster_key), avg@k + pass@k (real resampling), and
# confidence_level CI width.
#
# Usage:
#   bash scripts/stat_smoke_test.sh
# Override defaults via env, e.g.:
#   MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct LIMIT=200 bash scripts/stat_smoke_test.sh
#
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}"
LIMIT="${LIMIT:-100}"
DEVICE="${DEVICE:-cuda:0}"
BATCH="${BATCH:-auto}"
OUT="${OUT:-./stat_smoke_out}"
K="${K:-4}"                       # resamples for avg@k / pass@k
MARGS="pretrained=${MODEL_ID},dtype=bfloat16"

export HF_DATASETS_TRUST_REMOTE_CODE=1
export TOKENIZERS_PARALLELISM=false

echo "=================================================================="
echo " model=${MODEL_ID}  limit=${LIMIT}  device=${DEVICE}  K=${K}"
echo " out=${OUT}"
echo "=================================================================="

mkdir -p "${OUT}" custom_tasks

# ---- step 0: library + flags sanity (no GPU needed) ----
echo "--- [0] eval_stats import + CLI flags ---"
python -c "from lm_eval.api import eval_stats as es; assert abs(es.pass_at_k(5,2,3)-0.9)<1e-9; print('eval_stats OK: pass_at_k(5,2,3)=', es.pass_at_k(5,2,3))"
python -m lm_eval --help | grep -E -- "--confidence_level|--resamples" || { echo "FLAGS MISSING"; exit 1; }

# ---- write a custom generative task: gsm8k + avg_at_k filter + passk ----
# do_sample=true so the K resamples differ (otherwise pass@k is degenerate).
cat > custom_tasks/gsm8k_stat.yaml <<'YAML'
task: gsm8k_stat
dataset_path: gsm8k
dataset_name: main
output_type: generate_until
test_split: test
fewshot_split: train
num_fewshot: 5
passk: [1, 2]
doc_to_text: "Question: {{question}}\nAnswer:"
doc_to_target: "{{answer}}"
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
    ignore_case: true
    ignore_punctuation: false
    regexes_to_ignore:
      - ","
      - "(?s).*#### "
generation_kwargs:
  max_gen_toks: 256
  do_sample: true
  temperature: 0.8
  top_p: 0.95
  until:
    - "Question:"
    - "</s>"
filter_list:
  - name: "avg@K"
    filter:
      - function: "regex"
        regex_pattern: '#### (\-?[0-9\.\,]+)'
      - function: "avg_at_k"
YAML

# tolerate a single run failing (e.g. a dataset that won't load) so the rest
# still execute and the checker can report a partial checklist.
run () { echo; echo ">>> $*"; eval "$@" || echo "!! WARNING: that run failed; continuing so other checks still report"; }

# ---- [A] logprob MCQ: acc + acc_norm + rigorous SE family (IID) ----
run python -m lm_eval --model hf --model_args "${MARGS}" \
  --tasks arc_easy --device "${DEVICE}" --batch_size "${BATCH}" \
  --limit "${LIMIT}" --log_samples --output_path "${OUT}/arc95"

# ---- [B] same task at 0.90 confidence (CI width must shrink) ----
run python -m lm_eval --model hf --model_args "${MARGS}" \
  --tasks arc_easy --device "${DEVICE}" --batch_size "${BATCH}" \
  --limit "${LIMIT}" --confidence_level 0.90 --output_path "${OUT}/arc90"

# ---- [C] clustered SE: xnli_en has cluster_key=premise (expect clusters < n) ----
#   (if xnli fails to load on your datasets version, swap --tasks xnli_en for
#    --tasks xquad_en, which uses cluster_key=context.)
run python -m lm_eval --model hf --model_args "${MARGS}" \
  --tasks xnli_en --device "${DEVICE}" --batch_size "${BATCH}" \
  --limit "${LIMIT}" --log_samples --output_path "${OUT}/xnli"

# ---- [D] generative avg@k + pass@k with real resampling ----
run python -m lm_eval --model hf --model_args "${MARGS}" \
  --tasks gsm8k_stat --include_path ./custom_tasks --device "${DEVICE}" \
  --batch_size "${BATCH}" --limit "${LIMIT}" --resamples "${K}" \
  --log_samples --output_path "${OUT}/gsm8k_stat"

echo
echo "=================================================================="
echo " Runs complete. Verifying results..."
echo "=================================================================="
python scripts/check_stat_results.py "${OUT}"
