# FAR cluster installation

## One-command setup

The benchmark uses isolated scorer environments because PaddleOCR,
HPSv3++, GenEval, and LongText have incompatible CUDA and Transformers
dependencies. On EOS/NRT/Draco machines with the standard FAR shared
environments:

```bash
git clone https://github.com/JunyiWuCode/benchmark_image.git
cd benchmark_image
bash scripts/setup_far_cluster.sh
```

This command:

1. installs `benchmark_image` into `far-anyflow15`;
2. runs FAR-RL's public HPSv2/Aesthetic/GenEval asset setup;
3. downloads `hpsv3++.pth` when it is missing;
4. verifies all six Python interpreters, evaluator repositories, and weights;
5. imports the package and checks the full-suite image count.

Use non-default roots as follows:

```bash
FAR_RL_ROOT=/path/to/FAR-RL \
CONDA_ENV_ROOT=/path/to/conda/envs \
MAIN_PYTHON=/path/to/anyflow/bin/python \
bash scripts/setup_far_cluster.sh
```

For a read-only preflight:

```bash
bash scripts/setup_far_cluster.sh --verify-only
```

## Qwen-Image-Bench and Q-Judger

Q-Judger is a separate 27B multimodal model and intentionally uses its own
environment. Install the official source pinned at revision `8ab1fb47`, the
model, and scorer dependencies with:

```bash
cd benchmark_image
bash scripts/setup_qwen_image_bench.sh
```

The defaults are:

```text
environment:  $CONDA_ENV_ROOT/q_judger
source:       $FAR_RL_ROOT/third_party/reference_repos/Qwen-Image-Bench
model:        $FAR_RL_ROOT/third_party/reference_models/Qwen-Image-Bench
```

The official scorer uses ms-swift PtEngine. Optional accelerated scoring can
reuse the same model and generated images with these isolated environments:

```text
vLLM:   vllm 0.19.0, torch 2.10.0, torchvision 0.25.0
SGLang: sglang 0.5.17, torch 2.9.1+cu128,
        torchvision 0.24.1+cu128, torchaudio 2.9.1+cu128
```

Install this package into each engine environment after creating it:

```bash
$CONDA_ENV_ROOT/vllm/bin/python -m pip install --no-deps --upgrade .
$CONDA_ENV_ROOT/sglang/bin/python -m pip install --no-deps --upgrade .
```

Do not upgrade only `torch` inside an engine environment. vLLM and SGLang ship
compiled CUDA extensions tied to their tested torch stack. Run each backend in
a separate output directory while hardlinking or symlinking the same validated
generation directory; otherwise backend outputs cannot be compared exactly.

Override them with `Q_JUDGER_ENV_PREFIX`, `QWEN_IMAGE_BENCH_ROOT`, and
`Q_JUDGER_MODEL`. Setup downloads roughly the size of a 27B BF16 model, so run
it on a compute node with adequate storage, never at a training evaluation
boundary. The tested CUDA environment uses `torch 2.11.0+cu128`; the upstream
README's commented `torch 2.12.0` example is not currently available from the
official cu128 wheel index.

## Required environments

The following names and core package versions are the tested EOS layout:

| Environment | Purpose | Core tested packages |
| --- | --- | --- |
| `far-anyflow15` | AnyFlow generation, HPSv2, Aesthetic, CLIPScore | PyTorch 2.11, Transformers 5.5, HPSv2 1.2 |
| `hpsv3pp` | HPSv3++ at iter 0 | PyTorch 2.11, Transformers 4.57 |
| `paddleocr_gpu_official` | OCR and CVTG text recognition | PaddleOCR 3.3.3, PaddlePaddle GPU 3.3.1 |
| `longtext_ocr` | LongText and CVTG CLIP | PyTorch 2.11, Transformers 4.52, qwen-vl-utils 0.0.14 |
| `flow_opd_geneval_reward` | GenEval detector | PyTorch 2.1.2, MMEngine 0.10.7, MMDet 2.28.2 |
| `geneval2_official` | GenEval2 Soft-TIFA | PyTorch 2.11, Transformers 4.57, SciPy 1.15 |

The setup script deliberately does not rebuild an existing environment or
upgrade its CUDA stack. On a new cluster, create these isolated environments
from the official scorer repositories first, then run the one-command setup.
This prevents a benchmark setup from silently replacing the AnyFlow training
environment's PyTorch.

## Required weights and code

All paths are relative to `FAR_RL_ROOT`:

```text
hps_ckpt/HPS_v2.1_compressed.pt
hps_ckpt/open_clip_pytorch_model.bin
benchmark_assets/sac+logos+ava1-l14-linearMSE.pth
third_party/HPSv3-PlusPlus/checkpoints/hpsv3++.pth
third_party/HPSv3-PlusPlus/hpsv3/config/train_stage2.yaml
third_party/reference_repos/geneval/evaluation/evaluate_images.py
third_party/reward-server/model/mask2former2/
third_party/reference_repos/GenEval2/evaluation.py
third_party/reference_repos/GenEval2/geneval2_data.jsonl
```

The evaluator never downloads a weight during training. Setup and verification
happen before the experiment starts, so an `eval_freq` boundary cannot hang on
an unexpected model download.
