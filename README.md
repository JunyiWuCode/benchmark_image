# benchmark_image

Reproducible, full-suite image benchmarks for training-time evaluation of
text-to-image models. Generation stays in the training repository; this package
owns prompt protocols, official output layouts, scorer orchestration, metric
summaries, and timing records.

## Suites

| YAML switch | Images | Reported metrics |
| --- | ---: | --- |
| `aesthetic_quality` | 1,024 | HPSv2, HPSv3++ at iter 0, Aesthetic, CLIPScore |
| `hpsv2_official` | 3,200 | HPSv2.1 official average and four style scores |
| `geneval` | 2,212 | GenEval overall and per-task scores |
| `geneval2` | 800 | GenEval2 Soft-TIFA arithmetic and geometric means |
| `ocr` | 1,018 | OCR accuracy |
| `cvtg` | 2,000 | CVTG word accuracy, NED, and CLIPScore |
| `longtext_en` | 640 | LongText EN text score |

The complete suite has 10,894 generated images for each inference-step setting.
HumanAES is intentionally not included.

## Installation

```bash
pip install "git+https://github.com/JunyiWuCode/benchmark_image.git"
```

For FAR clusters, environment and weight setup is one command:

```bash
git clone https://github.com/JunyiWuCode/benchmark_image.git
cd benchmark_image
bash scripts/setup_far_cluster.sh
```

See [`docs/INSTALL_FAR_CLUSTER.md`](docs/INSTALL_FAR_CLUSTER.md) for the tested
environment matrix, all required weights, path overrides, and a read-only
verification command.

Heavy and mutually conflicting scorer dependencies are intentionally excluded
from this package. `evaluate_generated_suite` starts each scorer with its
configured Python interpreter. On FAR clusters, two roots are sufficient:

```yaml
scorers:
  far_rl_root: /home/hcai/workspace/code/junyiwu/FAR-RL
  conda_env_root: /home/hcai/workspace/anaconda3/envs
```

Individual Python, checkpoint, model, and batch-size settings can override the
derived defaults.

## AnyFlow configuration

```yaml
datasets:
  val:
    type: ImageBenchmarkDataset
    benchmarks: [aesthetic_quality, hpsv2_official, geneval, geneval2, ocr, cvtg, longtext_en]
    dataloader_cfg:
      global_batch_size: 64
      num_workers: 4
      shuffle: false
      drop_last: false

val:
  eval_freq: 1000
  benchmark_cfg:
    benchmarks: [aesthetic_quality, hpsv2_official, geneval, geneval2, ocr, cvtg, longtext_en]
    height: 512
    width: 512
    seed: 0
    guidance_scale: 0.0
    num_inference_steps: [2, 4, 8, 50]
    use_ema_for_metrics: true
    allow_partial: false
    scorers:
      far_rl_root: /home/hcai/workspace/code/junyiwu/FAR-RL
      conda_env_root: /home/hcai/workspace/anaconda3/envs
```

Production runs must use `allow_partial: false`. The
`smoke_max_prompts_per_benchmark` dataset option and `allow_partial: true` exist
only for integration smoke tests and never produce official benchmark numbers.

Each benchmark writes a `timing.json` containing generation and scoring wall
times. Generation time is the distributed critical-path estimate for that
benchmark; scoring time is measured around the complete scorer stage.
