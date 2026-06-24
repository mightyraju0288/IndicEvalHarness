#!/usr/bin/env bash
#
# Reproducible, root-free environment for the IndicEvalHarness fork on a shared
# GPU box (no Docker / no admin needed). Creates an isolated venv with PINNED
# versions that match a CUDA 12.4 driver, so the dependency errors hit during
# manual setup (torch built for cu126/cu128, transformers >= 4.49) cannot recur.
#
# Everything lives under $SCRATCH so cleanup is a single `rm -rf $SCRATCH`.
#
# Usage (run from the repo root):
#   SCRATCH=~/iev-env bash scripts/setup_env.sh
# Then:
#   source ~/iev-env/venv/bin/activate
#
set -euo pipefail

SCRATCH="${SCRATCH:-/tmp/$USER-iev-env}"
TORCH_CUDA="${TORCH_CUDA:-cu124}"     # matches driver 550 / CUDA 12.4 on the box
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "=================================================================="
echo " scratch=$SCRATCH   repo=$REPO_ROOT   torch=$TORCH_CUDA"
echo "=================================================================="

mkdir -p "$SCRATCH"
export HF_HOME="$SCRATCH/hf"
export PIP_CACHE_DIR="$SCRATCH/pipcache"

# isolated venv (gives a `python`; never touches system packages)
python3 -m venv "$SCRATCH/venv"
# shellcheck disable=SC1091
source "$SCRATCH/venv/bin/activate"
python -m pip install --upgrade pip

# 1) torch built for the box's CUDA (12.4). Installed FIRST so the harness's
#    unbounded `torch>=1.8` requirement is already satisfied and not overridden.
python -m pip install torch --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"

# 2) transformers < 4.49 (AutoModelForVision2Seq was removed in 4.49).
python -m pip install "transformers<4.49"

# 3) the harness + remaining deps (scipy, needed by eval_stats, comes via scikit-learn).
python -m pip install -e "$REPO_ROOT"

# verify the environment is correct, fail loudly otherwise
python - <<'PY'
import torch, scipy, transformers
import lm_eval.api.eval_stats as es
print("torch       :", torch.__version__, "| cuda", torch.version.cuda)
print("transformers:", transformers.__version__)
print("scipy       :", scipy.__version__)
assert torch.version.cuda and torch.version.cuda.startswith("12.4"), "torch is not a cu124 build"
assert int(transformers.__version__.split(".")[1]) < 49, "transformers >= 4.49"
assert abs(es.pass_at_k(5, 2, 3) - 0.9) < 1e-9, "eval_stats broken"
assert torch.cuda.is_available(), "CUDA not available (check --gpus / driver)"
print("ENV OK  | cuda available:", torch.cuda.is_available())
PY

echo
echo "Environment ready. Activate it later with:"
echo "    source $SCRATCH/venv/bin/activate"
echo "Clean up everything with:"
echo "    rm -rf $SCRATCH"
