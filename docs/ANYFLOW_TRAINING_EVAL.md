# AnyFlow ZImage training-time benchmark guide

This guide installs `benchmark_image`, enables the full image suite in an
AnyFlow ZImage training YAML, and runs evaluation from the real training loop.
The tested FAR layout is:

```text
/path/to/workspace/
├── FAR-Anyflow1.5
├── FAR-RL
└── benchmark_image

/path/to/conda/envs/
├── far-anyflow15
├── hpsv3pp
├── paddleocr_gpu_official
├── longtext_ocr
├── flow_opd_geneval_reward
└── geneval2_official
```

The suite evaluates 10,894 generated images per NFE and reports:

- HPSv2, HPSv3++ at iter 0, Aesthetic, and CLIPScore;
- HPSv2.1 official average and its four style scores;
- GenEval overall and per-task scores;
- GenEval2 Soft-TIFA arithmetic and geometric means;
- OCR accuracy;
- CVTG word accuracy, NED, and CLIPScore;
- LongText EN text score.

HumanAES is intentionally excluded.

## 1. Clone the repositories

```bash
cd /path/to/workspace

git clone \
  --branch <anyflow-branch> \
  <training-repository-url> \
  FAR-Anyflow1.5

git clone \
  --branch <far-rl-branch> \
  <training-repository-url> \
  FAR-RL

git clone \
  https://github.com/JunyiWuCode/benchmark_image.git \
  benchmark_image
```

If a checkout already exists, inspect its branch and worktree instead of
cloning over it:

```bash
git -C /path/to/workspace/FAR-Anyflow1.5 status --short --branch
git -C /path/to/workspace/FAR-RL status --short --branch
git -C /path/to/workspace/benchmark_image status --short --branch
```

## 2. Install and verify the benchmark

Activate the Python environment used by AnyFlow:

```bash
source /path/to/conda/etc/profile.d/conda.sh
conda activate far-anyflow15
```

Run the cluster setup:

```bash
cd /path/to/workspace/benchmark_image
bash scripts/setup_far_cluster.sh
```

The script installs `benchmark_image` into `far-anyflow15`, retrieves missing
public assets through FAR-RL, and checks every isolated scorer environment and
weight. It deliberately does not replace the PyTorch/CUDA stack of an existing
environment.

Before every long training run, use the read-only preflight:

```bash
cd /path/to/workspace/benchmark_image
bash scripts/setup_far_cluster.sh --verify-only
```

Success ends with:

```text
benchmark_image cluster setup verified successfully.
```

For non-standard paths:

```bash
FAR_RL_ROOT=/path/to/FAR-RL \
CONDA_ENV_ROOT=/path/to/conda/envs \
MAIN_PYTHON=/path/to/anyflow/bin/python \
bash scripts/setup_far_cluster.sh
```

## 3. Configure the full suite

Add the benchmark dataset:

```yaml
datasets:
  val:
    type: ImageBenchmarkDataset
    benchmarks:
      [aesthetic_quality, hpsv2_official, geneval, geneval2, ocr, cvtg, longtext_en]
    dataloader_cfg:
      global_batch_size: 64
      num_workers: 4
      shuffle: false
      drop_last: false
```

Add the training-time evaluator:

```yaml
val:
  val_pipeline: ZImageAnyFlowPipeline
  skip_first_eval: true
  eval_freq: 1000
  benchmark_cfg:
    benchmarks:
      [aesthetic_quality, hpsv2_official, geneval, geneval2, ocr, cvtg, longtext_en]
    height: 512
    width: 512
    seed: 0
    guidance_scale: 0.0
    num_inference_steps: [2, 4, 8, 50]
    use_ema_for_metrics: true
    allow_partial: false
    scorers:
      far_rl_root: /path/to/workspace/FAR-RL
      conda_env_root: /path/to/conda/envs
      longtext_max_new_tokens: 256
```

Important settings:

- `allow_partial: false` is required for reportable numbers.
- `use_ema_for_metrics: true` evaluates the EMA weights used for deployment.
- One complete suite is generated independently for every configured NFE.
- `longtext_max_new_tokens: 256` avoids rare Qwen-VL repetition loops. Use
  `1024` only when strict X-Omni generation-parameter parity is required.
- Keep the benchmark list in `datasets.val` and `val.benchmark_cfg` identical.

The versioned reference configurations are:

```text
options/train/anyflow_v1.0_zimage/pretrain/
  zimage_stage1_v1_diffusion_0.5_consis_0.25_512px_b256_eos.yml
  zimage_stage1_v1_diffusion_0.5_consis_0.25_1024px_b64_eos.yml

options/train/anyflow_v1.5_zimage/pretrain/
  zimage_stage1_v15_diffusion_0.5_consis_0.25_512px_b256_eos.yml
  zimage_stage1_v15_diffusion_0.5_consis_0.25_1024px_b64_eos.yml
```

For 1024px, set both `height` and `width` to `1024`. Choose the validation
global batch from measured GPU memory rather than copying the training batch.

## 4. Launch training

On one eight-GPU node:

```bash
cd /path/to/workspace/FAR-Anyflow1.5
source /path/to/conda/etc/profile.d/conda.sh
conda activate far-anyflow15

torchrun \
  --nnodes 1 \
  --nproc_per_node 8 \
  --master_port 17643 \
  -m far.main \
  config_path=options/train/anyflow_v1.5_zimage/pretrain/zimage_stage1_v15_diffusion_0.5_consis_0.25_512px_b256_eos.yml
```

At steps divisible by `eval_freq`, the same training processes:

1. retain the training model, optimizer, and EMA state;
2. swap to EMA weights for generation;
3. generate all benchmark images for each NFE;
4. launch the isolated distributed scorers;
5. merge exact metric summaries;
6. log the metrics to the normal training logger and W&B;
7. restore training weights and continue optimization.

This is not a separate post-training evaluation process.

## 5. Run a real training-loop smoke

The following runs one optimization step, triggers the complete benchmark at
step 1, and evaluates only 4-step sampling:

```bash
torchrun \
  --nnodes 1 \
  --nproc_per_node 8 \
  --master_port 17651 \
  -m far.main \
  config_path=options/train/anyflow_v1.5_zimage/pretrain/zimage_stage1_v15_diffusion_0.5_consis_0.25_512px_b256_eos.yml \
  name=zimage_smoke_v15_512_train_fullbench_4step \
  train.total_iter=1 \
  val.eval_freq=1 \
  'val.benchmark_cfg.num_inference_steps=[4]' \
  logger.use_wandb=false
```

This smoke is intentionally full-sized. Do not add
`smoke_max_prompts_per_benchmark` when measuring production memory or time.

### EOS reference measurements

The following measurements were collected on 2026-07-30 with eight H100 80 GB
GPUs. Each run used the real Stage 1 training loop, one optimization step,
global validation batch 64, and one complete 4-step suite of 10,894 images.
Training model, optimizer, and EMA state remained resident during evaluation.

| Model | Resolution | Generation | Scoring | Benchmark total | End to end | Peak GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AnyFlow v1 | 512 | 10.86 min | 11.09 min | 21.95 min | 25.68 min | 48.1 GiB |
| AnyFlow v1.5 | 512 | 7.97 min | 11.12 min | 19.09 min | 22.60 min | 47.8 GiB |
| AnyFlow v1 | 1024 | 26.46 min | 10.65 min | 37.11 min | 39.30 min | 70.9 GiB |
| AnyFlow v1.5 | 1024 | 26.38 min | 10.57 min | 36.95 min | 39.18 min | 70.9 GiB |

The v1 512 generation pass included a roughly three-minute filesystem stall;
its steady-state generation throughput otherwise matched v1.5. The 1024 runs
prove that global validation batch 64 fits this Stage 1 configuration on H100
80 GB, but the roughly 9 GB remaining margin is not enough to assume that a
larger Stage 2 or on-policy trainer will also fit. Measure those trainers
separately.

With production NFE values `[2, 4, 8, 50]`, the full 10,894-image suite is
repeated four times. Schedule evaluation accordingly; the table above measures
only the requested 4-step pass.

## 6. Find outputs and timings

For evaluation at step 1000 and 4-step sampling:

```text
experiments/<name>/visualization/benchmark/iter_1000/step_4/
├── timing.json
├── _rank_manifests/
└── <benchmark>/
    ├── generation/
    │   ├── results.jsonl
    │   └── summary.json
    ├── scoring/
    │   └── summary.json
    └── timing.json
```

The root `timing.json` contains the distributed generation critical path,
scoring wall time, and total wall time. Each benchmark directory also has its
own scorer timing and raw outputs. `_rank_manifests` are implementation
artifacts: a distributed sampler may pad them with duplicate rows. The merged
`generation/results.jsonl` files are deduplicated by artifact ID and are the
canonical generated sets.

Verify completeness:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path(
    "experiments/<name>/visualization/benchmark/iter_1000/step_4"
)
timing = json.loads((root / "timing.json").read_text())
generation_summaries = [
    json.loads(path.read_text())
    for path in sorted(root.glob("*/generation/summary.json"))
]
scoring_summaries = sorted(root.glob("*/scoring/summary.json"))
images = sum(summary["num_images"] for summary in generation_summaries)
print("images:", images)
print("generation summaries:", len(generation_summaries))
print("scoring summaries:", len(scoring_summaries))
print("timing:", timing)
assert images == 10894
assert len(generation_summaries) == 7
assert len(scoring_summaries) == 7
assert all(summary["complete"] for summary in generation_summaries)
PY
```

## 7. Rerun scorers without regenerating images

Use this after a scorer dependency fix or to change a scorer-only setting:

```bash
torchrun \
  --nnodes 1 \
  --nproc_per_node 8 \
  --master_port 17653 \
  -m benchmark_image.score_existing \
  --root experiments/<name>/visualization/benchmark/iter_1000/step_4 \
  --benchmarks ocr,cvtg,longtext_en
```

This keeps the generated image set fixed and updates only the selected metric
summaries and scoring timings.

## 8. Operational checks

Before considering an evaluation successful, check all of the following:

```bash
grep -E "Begin evaluating|evaluation results|CUDA out of memory|Traceback" \
  experiments/<name>/train_*.log

find experiments/<name>/visualization/benchmark \
  -name metrics.json -o -name timing.json
```

A progress bar reaching 100% proves only image generation completed. The run is
complete only after every scorer returns, all seven scoring summaries are
written, the root timing contains `total_seconds`, and the training loop
proceeds to the next optimization step.
