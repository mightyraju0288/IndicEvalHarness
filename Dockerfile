# Pinned environment for the IndicEvalHarness fork (statistical-rigor features).
#
# Base = official PyTorch image built for CUDA 12.4, so `torch` is already the
# correct cu124 build. Containers use the HOST NVIDIA driver, and your box reports
# CUDA 12.4 / driver 550, which supports CUDA runtime <= 12.4 -> cu124 matches.
# This removes the two issues hit during manual setup:
#   * torch built for cu126/cu128 vs a 12.4 driver  -> fixed (cu124 base)
#   * transformers >= 4.49 removed AutoModelForVision2Seq -> pinned < 4.49 below
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/data/hf \
    HF_DATASETS_TRUST_REMOTE_CODE=1

# git is required by some HF dataset loading scripts
RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . /workspace

# Pin transformers < 4.49 FIRST so the editable install below (transformers>=4.1,
# unbounded) does not pull a version that removed AutoModelForVision2Seq.
# torch (pyproject: >=1.8) is already satisfied by the base image's cu124 build,
# so pip will NOT replace it. scipy (needed by eval_stats) arrives via scikit-learn.
RUN pip install --upgrade pip \
 && pip install "transformers<4.49" \
 && pip install -e .

# Fail the build early (not at runtime) if the environment is wrong.
RUN python -c "import torch, scipy, transformers; \
print('torch', torch.__version__, '| cuda', torch.version.cuda, '| transformers', transformers.__version__); \
assert torch.version.cuda and torch.version.cuda.startswith('12.4'), 'torch is not a cu124 build'; \
assert int(transformers.__version__.split('.')[1]) < 49, 'transformers >= 4.49'; \
import lm_eval.api.eval_stats as es; assert abs(es.pass_at_k(5,2,3)-0.9) < 1e-9; \
print('eval_stats OK: pass_at_k(5,2,3)=', es.pass_at_k(5,2,3))"

# Default entrypoint is the CLI; override with `--entrypoint bash` to run scripts.
ENTRYPOINT ["python", "-m", "lm_eval"]
CMD ["--help"]
