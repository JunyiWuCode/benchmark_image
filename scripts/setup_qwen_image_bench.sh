#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
far_rl_root="${FAR_RL_ROOT:-/home/hcai/workspace/code/junyiwu/FAR-RL}"
conda_env_root="${CONDA_ENV_ROOT:-/home/hcai/workspace/anaconda3/envs}"
env_prefix="${Q_JUDGER_ENV_PREFIX:-${conda_env_root}/q_judger}"
source_root="${QWEN_IMAGE_BENCH_ROOT:-${far_rl_root}/third_party/reference_repos/Qwen-Image-Bench}"
model_root="${Q_JUDGER_MODEL:-${far_rl_root}/third_party/reference_models/Qwen-Image-Bench}"
official_revision="8ab1fb47df2fba7b0cb046770a87f6323b98ecfc"

if [[ ! -d "${source_root}/.git" ]]; then
  mkdir -p "$(dirname "${source_root}")"
  git clone https://github.com/QwenLM/Qwen-Image-Bench.git "${source_root}"
fi
git -C "${source_root}" fetch --depth 1 origin "${official_revision}"
git -C "${source_root}" checkout --detach "${official_revision}"

if [[ ! -x "${env_prefix}/bin/python" ]]; then
  conda create -y -p "${env_prefix}" python=3.11 pip
fi

python="${env_prefix}/bin/python"
if ! "${python}" - <<'PY' >/dev/null 2>&1
import torch
import torchvision

assert torch.__version__.startswith("2.11.0")
assert torchvision.__version__.startswith("0.26.0")
PY
then
  "${python}" -m pip install \
    --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.11.0 \
    torchvision==0.26.0
fi
"${python}" -m pip install -r "${source_root}/requirements.txt"
"${python}" -m pip install --no-deps --upgrade "${repo_root}"

mkdir -p "$(dirname "${model_root}")"
MODEL_ROOT="${model_root}" "${python}" - <<'PY'
import os

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen-Image-Bench",
    local_dir=os.environ["MODEL_ROOT"],
)
PY

SOURCE_ROOT="${source_root}" MODEL_ROOT="${model_root}" "${python}" - <<'PY'
import os
from pathlib import Path

import torch
import transformers
import swift
import torchvision

source = Path(os.environ["SOURCE_ROOT"])
model = Path(os.environ["MODEL_ROOT"])
assert source.joinpath("judge.py").is_file()
assert model.joinpath("config.json").is_file()
print("Qwen-Image-Bench source:", source)
print("Q-Judger model:", model)
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("transformers:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())
PY

echo "Qwen-Image-Bench setup complete."
