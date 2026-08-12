from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from importlib.resources import files
from pathlib import Path

from benchmark_image.protocols import normalize_benchmarks


WORKER_ROOT = files("benchmark_image").joinpath("workers")


def _resolved_config(config: dict | None) -> dict:
    resolved = dict(config or {})
    far_rl_root = Path(resolved.get("far_rl_root", "/home/hcai/workspace/code/junyiwu/FAR-RL"))
    conda_env_root = Path(
        resolved.get("conda_env_root", "/home/hcai/workspace/anaconda3/envs")
    )
    defaults = {
        "aesthetic_quality_python": conda_env_root / "far-anyflow15" / "bin" / "python",
        "hpsv2_python": conda_env_root / "far-anyflow15" / "bin" / "python",
        "hpsv3pp_python": conda_env_root / "hpsv3pp" / "bin" / "python",
        "ocr_python": conda_env_root / "paddleocr_gpu_official" / "bin" / "python",
        "cvtg_clip_python": conda_env_root / "longtext_ocr" / "bin" / "python",
        "longtext_python": conda_env_root / "longtext_ocr" / "bin" / "python",
        "geneval_python": conda_env_root / "flow_opd_geneval_reward" / "bin" / "python",
        "geneval2_python": conda_env_root / "geneval2_official" / "bin" / "python",
        "hpsv2_checkpoint_path": far_rl_root / "hps_ckpt" / "HPS_v2.1_compressed.pt",
        "hpsv2_open_clip_pretrained_path": far_rl_root / "hps_ckpt" / "open_clip_pytorch_model.bin",
        "aesthetic_checkpoint_path": far_rl_root / "benchmark_assets" / "sac+logos+ava1-l14-linearMSE.pth",
        "hpsv3pp_root": far_rl_root / "third_party" / "HPSv3-PlusPlus",
        "hpsv3pp_config_path": far_rl_root / "third_party" / "HPSv3-PlusPlus" / "hpsv3" / "config" / "train_stage2.yaml",
        "hpsv3pp_checkpoint_path": far_rl_root / "third_party" / "HPSv3-PlusPlus" / "checkpoints" / "hpsv3++.pth",
        "geneval_root": far_rl_root / "third_party" / "reference_repos" / "geneval",
        "geneval_model_config": far_rl_root / "third_party" / "reward-server" / "mmdetection" / "configs" / "mask2former" / "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py",
        "geneval_model_path": far_rl_root / "third_party" / "reward-server" / "model" / "mask2former2",
        "geneval2_root": far_rl_root / "third_party" / "reference_repos" / "GenEval2",
        "geneval2_benchmark_data": far_rl_root / "third_party" / "reference_repos" / "GenEval2" / "geneval2_data.jsonl",
        "q_judger_python": conda_env_root / "q_judger" / "bin" / "python",
        "q_judger_vllm_python": conda_env_root / "vllm" / "bin" / "python",
        "q_judger_sglang_python": conda_env_root / "sglang" / "bin" / "python",
        "qwen_image_bench_root": far_rl_root / "third_party" / "reference_repos" / "Qwen-Image-Bench",
        "q_judger_model": far_rl_root / "third_party" / "reference_models" / "Qwen-Image-Bench",
    }
    for key, value in defaults.items():
        resolved.setdefault(key, str(value))
    return resolved


def _dist_info() -> tuple[int, int, int]:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size(), int(os.environ.get("LOCAL_RANK", "0"))
    except ImportError:
        pass
    return 0, 1, 0


def _barrier() -> None:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except ImportError:
        return


def _broadcast(metrics):
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            payload = [metrics]
            dist.broadcast_object_list(payload, src=0)
            return payload[0]
    except ImportError:
        pass
    return metrics


def _remote_worker_layout(rank: int, world_size: int, remote_urls: str):
    urls = [url.strip() for url in remote_urls.split(",") if url.strip()]
    if not urls:
        raise ValueError("Remote Q-Judger scoring requires at least one URL.")
    worker_world_size = min(world_size, len(urls))
    return worker_world_size, rank, rank < worker_world_size


def _run_worker(
    python: str,
    worker: str,
    args: list[str],
    local_rank: int | None,
) -> None:
    env = os.environ.copy()
    if local_rank is not None:
        visible_devices = [
            device.strip()
            for device in env.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if device.strip()
        ]
        if visible_devices:
            if local_rank >= len(visible_devices):
                raise RuntimeError(
                    f"LOCAL_RANK={local_rank} is outside CUDA_VISIBLE_DEVICES="
                    f"{env['CUDA_VISIBLE_DEVICES']!r}."
                )
            env["CUDA_VISIBLE_DEVICES"] = visible_devices[local_rank]
        else:
            env["CUDA_VISIBLE_DEVICES"] = str(local_rank)
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    for name in (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "GROUP_WORLD_SIZE",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
    ):
        env.pop(name, None)
    subprocess.run([python, str(WORKER_ROOT.joinpath(worker)), *args], check=True, env=env)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _flatten(prefix: str, value, output: dict[str, float]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _flatten(f"{prefix}_{key}" if prefix else str(key), child, output)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        output[prefix] = float(value)


def evaluate_generated_suite(root: str | Path, benchmarks, config: dict | None = None) -> dict[str, float]:
    """Score one generated NFE directory and return broadcast scalar metrics."""

    root = Path(root)
    config = _resolved_config(config)
    names = normalize_benchmarks(benchmarks)
    rank, world_size, local_rank = _dist_info()
    allow_partial = bool(config.get("allow_partial", False))

    for benchmark in names:
        started_at = time.time()
        manifest = root / benchmark / "generation" / "results.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing generated manifest for {benchmark}: {manifest}")

        if benchmark == "aesthetic_quality":
            scoring = root / benchmark / "scoring"
            scoring.mkdir(parents=True, exist_ok=True)
            common = [
                "--results", str(manifest),
                "--output-dir", str(scoring),
                "--rank", str(rank),
                "--world-size", str(world_size),
            ]
            _run_worker(
                str(config.get("aesthetic_quality_python", sys.executable)),
                "score_public_metrics.py",
                [
                    *common,
                    "--batch-size", str(int(config.get("aesthetic_quality_batch_size", 16))),
                    "--hps-checkpoint-path", str(config["hpsv2_checkpoint_path"]),
                    "--hps-open-clip-pretrained", str(config.get("hpsv2_open_clip_pretrained", "laion2B-s32B-b79K")),
                    "--hps-open-clip-pretrained-path", str(config.get("hpsv2_open_clip_pretrained_path", "")),
                    "--aesthetic-checkpoint-path", str(config["aesthetic_checkpoint_path"]),
                ],
                local_rank,
            )
            _run_worker(
                str(config["hpsv3pp_python"]),
                "score_hpsv3pp.py",
                [
                    *common,
                    "--batch-size", str(int(config.get("hpsv3pp_batch_size", 8))),
                    "--hpsv3pp-root", str(config["hpsv3pp_root"]),
                    "--config-path", str(config["hpsv3pp_config_path"]),
                    "--checkpoint-path", str(config["hpsv3pp_checkpoint_path"]),
                ],
                local_rank,
            )
            _barrier()
            if rank == 0:
                merge_common = ["--merge", "--output-dir", str(scoring), "--world-size", str(world_size)]
                if allow_partial:
                    merge_common.append("--allow-partial")
                _run_worker(str(config.get("aesthetic_quality_python", sys.executable)), "score_public_metrics.py", merge_common, local_rank)
                _run_worker(str(config["hpsv3pp_python"]), "score_hpsv3pp.py", merge_common, local_rank)
            _barrier()

        elif benchmark == "hpsv2_official":
            scoring = root / benchmark / "scoring"
            scoring.mkdir(parents=True, exist_ok=True)
            _run_worker(
                str(config.get("hpsv2_python", sys.executable)),
                "score_hpsv2.py",
                [
                    "--results", str(manifest),
                    "--output-dir", str(scoring),
                    "--rank", str(rank),
                    "--world-size", str(world_size),
                    "--batch-size", str(int(config.get("hpsv2_batch_size", 16))),
                    "--checkpoint-path", str(config["hpsv2_checkpoint_path"]),
                    "--open-clip-pretrained", str(config.get("hpsv2_open_clip_pretrained", "laion2B-s32B-b79K")),
                    "--open-clip-pretrained-path", str(config.get("hpsv2_open_clip_pretrained_path", "")),
                ],
                local_rank,
            )
            _barrier()
            if rank == 0:
                args = [
                    "--output-dir", str(scoring),
                    "--world-size", str(world_size),
                ]
                if allow_partial:
                    args.append("--allow-partial")
                _run_worker(str(config.get("hpsv2_python", sys.executable)), "score_hpsv2.py", ["--merge", *args], local_rank)
            _barrier()

        elif benchmark == "ocr":
            scoring = root / benchmark / "scoring"
            scoring.mkdir(parents=True, exist_ok=True)
            args = [
                "--results", str(manifest),
                "--output-dir", str(scoring),
                "--rank", str(rank),
                "--world-size", str(world_size),
                "--batch-size", str(int(config.get("ocr_batch_size", 16))),
            ]
            if bool(config.get("ocr_require_paddleocr_3_3_3", True)):
                args.append("--require-paddleocr-3-3-3")
            _run_worker(str(config["ocr_python"]), "score_ocr.py", args, local_rank)
            _barrier()
            if rank == 0:
                merge_args = ["--merge", "--output-dir", str(scoring), "--world-size", str(world_size)]
                if allow_partial:
                    merge_args.append("--allow-partial")
                _run_worker(str(config["ocr_python"]), "score_ocr.py", merge_args, local_rank)
            _barrier()

        elif benchmark == "geneval":
            scoring = root / benchmark / "scoring"
            scoring.mkdir(parents=True, exist_ok=True)
            _run_worker(
                str(config["geneval_python"]),
                "score_geneval.py",
                [
                    "--image-root", str(root / benchmark / "generation" / "geneval_images"),
                    "--output-dir", str(scoring),
                    "--geneval-root", str(config["geneval_root"]),
                    "--model-config", str(config["geneval_model_config"]),
                    "--model-path", str(config["geneval_model_path"]),
                    "--rank", str(rank),
                    "--world-size", str(world_size),
                ],
                local_rank,
            )
            _barrier()
            if rank == 0:
                merge_args = [
                    "--merge",
                    "--output-dir", str(scoring),
                    "--world-size", str(world_size),
                ]
                if allow_partial:
                    merge_args.append("--allow-partial")
                _run_worker(
                    str(config["geneval_python"]),
                    "score_geneval.py",
                    merge_args,
                    local_rank,
                )
            _barrier()

        elif benchmark == "cvtg":
            scoring = root / benchmark / "scoring"
            scoring.mkdir(parents=True, exist_ok=True)
            _run_worker(
                str(config["ocr_python"]),
                "score_cvtg.py",
                [
                    "--stage", "ocr", "--results", str(manifest), "--output_dir", str(scoring),
                    "--num_shards", str(world_size), "--shard_id", str(rank),
                    "--ocr_batch", str(int(config.get("cvtg_ocr_batch_size", 32))),
                ],
                local_rank,
            )
            _run_worker(
                str(config.get("cvtg_clip_python", sys.executable)),
                "score_cvtg.py",
                [
                    "--stage", "clip", "--results", str(manifest), "--output_dir", str(scoring),
                    "--num_shards", str(world_size), "--shard_id", str(rank),
                    "--clip_batch", str(int(config.get("cvtg_clip_batch_size", 32))),
                ],
                local_rank,
            )
            _barrier()
            if rank == 0:
                _run_worker(
                    str(config.get("cvtg_clip_python", sys.executable)),
                    "score_cvtg.py",
                    ["--stage", "merge", "--results", str(manifest), "--output_dir", str(scoring)],
                    local_rank,
                )
            _barrier()

        elif benchmark == "longtext_en":
            scoring = root / benchmark / "scoring"
            scoring.mkdir(parents=True, exist_ok=True)
            common = [
                "--results", str(manifest), "--output_dir", str(scoring),
                "--world_size", str(world_size), "--rank", str(rank), "--mode", "en",
                "--batch_size", str(int(config.get("longtext_batch_size", 4))),
                "--max_new_tokens", str(int(config.get("longtext_max_new_tokens", 256))),
            ]
            _run_worker(str(config["longtext_python"]), "score_longtext.py", common, local_rank)
            _barrier()
            if rank == 0:
                merge_args = [
                    "--merge", "--results", str(manifest), "--output_dir", str(scoring),
                    "--world_size", str(world_size), "--mode", "en",
                ]
                if allow_partial:
                    merge_args.append("--allow_partial")
                _run_worker(str(config["longtext_python"]), "score_longtext.py", merge_args, local_rank)
            _barrier()

        elif benchmark == "geneval2":
            generation = root / benchmark / "generation"
            scoring = root / benchmark / "scoring"
            scoring.mkdir(parents=True, exist_ok=True)
            rows = [
                json.loads(line)
                for line in manifest.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            benchmark_data = Path(config["geneval2_benchmark_data"])
            if allow_partial:
                benchmark_data = generation / "benchmark_data_partial.jsonl"
                ordered_rows = sorted(rows, key=lambda row: int(row["prompt_index"]))
                if rank == 0:
                    benchmark_data.write_text(
                        "".join(
                            json.dumps(row["metadata"], ensure_ascii=False) + "\n"
                            for row in ordered_rows
                        ),
                        encoding="utf-8",
                    )
                _barrier()
            _run_worker(
                str(config["geneval2_python"]),
                "score_geneval2.py",
                [
                    "--results", str(manifest),
                    "--output-dir", str(scoring),
                    "--geneval2-root", str(config["geneval2_root"]),
                    "--benchmark-data", str(benchmark_data),
                    "--rank", str(rank),
                    "--world-size", str(world_size),
                ],
                local_rank,
            )
            _barrier()
            if rank == 0:
                merge_args = [
                    "--merge",
                    "--results", str(manifest),
                    "--output-dir", str(scoring),
                    "--benchmark-data", str(benchmark_data),
                    "--world-size", str(world_size),
                ]
                if allow_partial:
                    merge_args.append("--allow-partial")
                _run_worker(
                    str(config["geneval2_python"]),
                    "score_geneval2.py",
                    merge_args,
                    local_rank,
                )
            _barrier()

        elif benchmark == "qwen_image_bench":
            scoring = root / benchmark / "scoring"
            scoring.mkdir(parents=True, exist_ok=True)
            q_judger_backend = str(config.get("q_judger_backend", "pt")).lower()
            if q_judger_backend not in {"pt", "vllm", "sglang", "remote"}:
                raise ValueError(
                    f"Unsupported q_judger_backend: {q_judger_backend!r}"
                )
            backend_python = {
                "pt": config["q_judger_python"],
                "vllm": config["q_judger_vllm_python"],
                "sglang": config["q_judger_sglang_python"],
                "remote": config["q_judger_python"],
            }[q_judger_backend]
            tensor_parallel_size = int(
                config.get("q_judger_tensor_parallel_size", 1)
            )
            if tensor_parallel_size <= 0:
                raise ValueError(
                    "q_judger_tensor_parallel_size must be positive, got "
                    f"{tensor_parallel_size}."
                )
            if tensor_parallel_size > 1 and world_size != 1:
                raise ValueError(
                    "Tensor-parallel Q-Judger scoring must use one orchestrator "
                    "process rather than torchrun."
                )
            remote_urls = ""
            run_worker = True
            if q_judger_backend == "remote":
                remote_urls = str(
                    config.get("q_judger_remote_urls")
                    or os.environ.get("QJUDGER_REWARD_URL", "")
                )
                worker_world_size, worker_rank, run_worker = _remote_worker_layout(
                    rank, world_size, remote_urls
                )
            else:
                worker_world_size = 1 if tensor_parallel_size > 1 else world_size
                worker_rank = 0 if tensor_parallel_size > 1 else rank
            worker_local_rank = None if tensor_parallel_size > 1 else local_rank
            common = [
                "--results", str(manifest),
                "--output-dir", str(scoring),
                "--qwen-image-bench-root", str(config["qwen_image_bench_root"]),
                "--model", str(config["q_judger_model"]),
                "--world-size", str(worker_world_size),
                "--max-batch-size", str(int(config.get("q_judger_max_batch_size", 24))),
                "--max-new-tokens", str(int(config.get("q_judger_max_new_tokens", 4096))),
                "--max-model-len", str(int(config.get("q_judger_max_model_len", 8192))),
                "--tensor-parallel-size", str(tensor_parallel_size),
                "--backend", q_judger_backend,
                "--gpu-memory-utilization",
                str(float(config.get("q_judger_gpu_memory_utilization", 0.9))),
            ]
            if q_judger_backend == "remote":
                common.extend(
                    [
                        "--remote-urls",
                        remote_urls,
                    ]
                )
            if run_worker:
                _run_worker(
                    str(backend_python),
                    "score_qwen_image_bench.py",
                    [*common, "--rank", str(worker_rank)],
                    worker_local_rank,
                )
            _barrier()
            if rank == 0:
                merge_args = [*common, "--merge"]
                if allow_partial:
                    merge_args.append("--allow-partial")
                _run_worker(
                    str(backend_python),
                    "score_qwen_image_bench.py",
                    merge_args,
                    worker_local_rank,
                )
            _barrier()

        if rank == 0:
            timing_path = root / benchmark / "timing.json"
            timing = _read_json(timing_path) if timing_path.is_file() else {}
            timing.update(
                {
                    "benchmark": benchmark,
                    "scoring_seconds": time.time() - started_at,
                    "world_size": world_size,
                    "q_judger_backend": (
                        q_judger_backend if benchmark == "qwen_image_bench" else None
                    ),
                    "q_judger_tensor_parallel_size": (
                        tensor_parallel_size
                        if benchmark == "qwen_image_bench"
                        else None
                    ),
                }
            )
            timing_path.write_text(
                json.dumps(timing, indent=2),
                encoding="utf-8",
            )
        _barrier()

    metrics = {}
    if rank == 0:
        summaries = {
            "aesthetic_quality": root / "aesthetic_quality" / "scoring" / "summary.json",
            "hpsv2_official": root / "hpsv2_official" / "scoring" / "summary.json",
            "geneval": root / "geneval" / "scoring" / "summary.json",
            "ocr": root / "ocr" / "scoring" / "summary.json",
            "cvtg": root / "cvtg" / "scoring" / "summary.json",
            "longtext_en": root / "longtext_en" / "scoring" / "summary.json",
            "geneval2": root / "geneval2" / "scoring" / "summary.json",
            "qwen_image_bench": root / "qwen_image_bench" / "scoring" / "summary.json",
        }
        for name in names:
            _flatten(name, _read_json(summaries[name]), metrics)
            if name == "aesthetic_quality":
                _flatten(
                    name,
                    _read_json(root / "aesthetic_quality" / "scoring" / "hpsv3pp_summary.json"),
                    metrics,
                )
        canonical = {
            "ocr": "ocr_metrics_flowopd_ocr_acc",
            "cvtg_word_accuracy": "cvtg_word_accuracy",
            "cvtg_ned": "cvtg_ned",
            "cvtg_clipscore": "cvtg_clip_score",
            "longtext_en_text_score": "longtext_en_text_score",
            "hpsv2": "aesthetic_quality_hpsv2",
            "hpsv3pp_iter0": "aesthetic_quality_hpsv3pp_iter0",
            "aesthetic": "aesthetic_quality_aesthetic",
            "clipscore": "aesthetic_quality_clipscore",
            "hpsv2_1_official_average": "hpsv2_official_average",
            "geneval_overall": "geneval_overall",
            "geneval2_soft_tifa_am": "geneval2_soft_tifa_am",
            "geneval2_soft_tifa_gm": "geneval2_soft_tifa_gm",
            "qwen_image_bench_overall": "qwen_image_bench_overall",
            "qwen_image_bench_quality": "qwen_image_bench_quality",
            "qwen_image_bench_aesthetics": "qwen_image_bench_aesthetics",
            "qwen_image_bench_alignment": "qwen_image_bench_alignment",
            "qwen_image_bench_real_world_fidelity": "qwen_image_bench_real_world_fidelity",
            "qwen_image_bench_creative_generation": "qwen_image_bench_creative_generation",
        }
        for output_name, source_name in canonical.items():
            if source_name in metrics:
                metrics[output_name] = metrics[source_name]
    return _broadcast(metrics)
