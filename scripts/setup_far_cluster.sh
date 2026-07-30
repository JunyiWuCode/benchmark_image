#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
far_rl_root="${FAR_RL_ROOT:-/home/hcai/workspace/code/junyiwu/FAR-RL}"
conda_env_root="${CONDA_ENV_ROOT:-/home/hcai/workspace/anaconda3/envs}"
main_python="${MAIN_PYTHON:-${conda_env_root}/far-anyflow15/bin/python}"
verify_only=0

usage() {
  cat <<'EOF'
Usage: setup_far_cluster.sh [options]

Install benchmark_image into the AnyFlow environment, fetch public benchmark
assets through FAR-RL, and verify every isolated scorer environment and weight.

Options:
  --far-rl-root PATH    FAR-RL checkout containing scorer assets
  --conda-env-root PATH Root containing the six benchmark conda environments
  --main-python PATH    Python used by AnyFlow
  --verify-only         Do not install or download; only run strict checks
  -h, --help            Show this help
EOF
}

while (($#)); do
  case "$1" in
    --far-rl-root)
      far_rl_root="$2"
      shift 2
      ;;
    --conda-env-root)
      conda_env_root="$2"
      shift 2
      ;;
    --main-python)
      main_python="$2"
      shift 2
      ;;
    --verify-only)
      verify_only=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "MISSING FILE: $1" >&2
    return 1
  fi
  echo "OK FILE: $1"
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "MISSING DIR: $1" >&2
    return 1
  fi
  echo "OK DIR: $1"
}

require_python() {
  local env_name="$1"
  local python="${conda_env_root}/${env_name}/bin/python"
  if [[ ! -x "${python}" ]]; then
    echo "MISSING ENV: ${env_name} (${python})" >&2
    return 1
  fi
  echo "OK ENV: ${env_name} (${python})"
}

if [[ ! -x "${main_python}" ]]; then
  echo "AnyFlow Python is not executable: ${main_python}" >&2
  exit 1
fi
if [[ ! -d "${far_rl_root}" ]]; then
  echo "FAR-RL checkout does not exist: ${far_rl_root}" >&2
  exit 1
fi

if [[ "${verify_only}" == "0" ]]; then
  "${main_python}" -m pip install --no-deps --upgrade "${repo_root}"
  if [[ ! -f "${far_rl_root}/hps_ckpt/HPS_v2.1_compressed.pt" \
     || ! -f "${far_rl_root}/hps_ckpt/open_clip_pytorch_model.bin" \
     || ! -f "${far_rl_root}/benchmark_assets/sac+logos+ava1-l14-linearMSE.pth" \
     || ! -f "${far_rl_root}/third_party/reference_repos/geneval/evaluation/evaluate_images.py" ]]; then
    if [[ ! -f "${far_rl_root}/scripts/setup_zimage_official_benchmarks.sh" ]]; then
      echo "Missing FAR-RL asset setup script." >&2
      exit 1
    fi
    (
      cd "${far_rl_root}"
      bash scripts/setup_zimage_official_benchmarks.sh
    )
  fi
  hps3_checkpoint="${far_rl_root}/third_party/HPSv3-PlusPlus/checkpoints/hpsv3++.pth"
  if [[ ! -f "${hps3_checkpoint}" ]]; then
    mkdir -p "$(dirname "${hps3_checkpoint}")"
    "${conda_env_root}/hpsv3pp/bin/hf" download \
      Junjun2333/HPSv3-PlusPlus hpsv3++.pth \
      --local-dir "$(dirname "${hps3_checkpoint}")"
  fi
fi

status=0
for env_name in \
  far-anyflow15 \
  hpsv3pp \
  paddleocr_gpu_official \
  longtext_ocr \
  flow_opd_geneval_reward \
  geneval2_official; do
  require_python "${env_name}" || status=1
done

for path in \
  "${far_rl_root}/hps_ckpt/HPS_v2.1_compressed.pt" \
  "${far_rl_root}/hps_ckpt/open_clip_pytorch_model.bin" \
  "${far_rl_root}/benchmark_assets/sac+logos+ava1-l14-linearMSE.pth" \
  "${far_rl_root}/third_party/HPSv3-PlusPlus/hpsv3/config/train_stage2.yaml" \
  "${far_rl_root}/third_party/HPSv3-PlusPlus/checkpoints/hpsv3++.pth" \
  "${far_rl_root}/third_party/reference_repos/geneval/evaluation/evaluate_images.py" \
  "${far_rl_root}/third_party/reward-server/mmdetection/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py" \
  "${far_rl_root}/third_party/reference_repos/GenEval2/evaluation.py" \
  "${far_rl_root}/third_party/reference_repos/GenEval2/geneval2_data.jsonl"; do
  require_file "${path}" || status=1
done
require_dir "${far_rl_root}/third_party/reward-server/model/mask2former2" || status=1

if [[ "${status}" != "0" ]]; then
  echo "Benchmark setup is incomplete. See docs/INSTALL_FAR_CLUSTER.md." >&2
  exit 1
fi

"${main_python}" - <<'PY'
from benchmark_image import BENCHMARKS, expected_image_count

names = tuple(BENCHMARKS)
print(f"benchmark_image import OK: {names}")
print(f"full suite images per NFE: {expected_image_count(names)}")
PY

echo "benchmark_image cluster setup verified successfully."
