from __future__ import annotations

import json
import os
import shutil
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


def _run_worker(python: str, worker: str, args: list[str], local_rank: int) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(local_rank)
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env.pop("RANK", None)
    env.pop("WORLD_SIZE", None)
    env.pop("LOCAL_RANK", None)
    subprocess.run([python, str(WORKER_ROOT.joinpath(worker)), *args], check=True, env=env)


def _scorer_layout(
    config: dict,
    *,
    global_rank: int,
    global_world_size: int,
    local_rank: int,
) -> tuple[bool, int, int, int]:
    """Map training ranks onto a smaller, explicitly assigned scorer pool."""

    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", global_world_size))
    if local_world_size <= 0 or global_world_size % local_world_size:
        raise ValueError(
            "scorer layout requires global_world_size divisible by LOCAL_WORLD_SIZE"
        )
    node_count = global_world_size // local_world_size
    node_rank = int(os.environ.get("GROUP_RANK", global_rank // local_world_size))
    workers_per_node = int(config.get("scorer_processes_per_node", local_world_size))
    if not 1 <= workers_per_node <= local_world_size:
        raise ValueError(
            "scorer_processes_per_node must be within [1, LOCAL_WORLD_SIZE], got "
            f"{workers_per_node} for LOCAL_WORLD_SIZE={local_world_size}"
        )

    configured_devices = config.get("scorer_cuda_devices")
    if configured_devices is None:
        scorer_devices = list(range(workers_per_node))
    else:
        scorer_devices = [int(device) for device in configured_devices]
        if len(scorer_devices) != workers_per_node:
            raise ValueError(
                "scorer_cuda_devices length must equal scorer_processes_per_node"
            )
        if len(set(scorer_devices)) != len(scorer_devices) or any(
            device < 0 for device in scorer_devices
        ):
            raise ValueError("scorer_cuda_devices must contain distinct non-negative IDs")

    active = local_rank < workers_per_node
    scorer_rank = node_rank * workers_per_node + min(local_rank, workers_per_node - 1)
    scorer_world_size = node_count * workers_per_node
    scorer_device = scorer_devices[min(local_rank, workers_per_node - 1)]
    return active, scorer_rank, scorer_world_size, scorer_device


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
    global_rank, global_world_size, local_rank = _dist_info()
    scorer_active, rank, world_size, scorer_device = _scorer_layout(
        config,
        global_rank=global_rank,
        global_world_size=global_world_size,
        local_rank=local_rank,
    )
    allow_partial = bool(config.get("allow_partial", False))
    stage_errors: list[str] = []

    def run_worker(python: str, worker: str, args: list[str]) -> None:
        if scorer_active and not stage_errors:
            try:
                _run_worker(python, worker, args, scorer_device)
            except Exception as exc:
                stage_errors.append(
                    f"global_rank={global_rank} scorer_rank={rank} "
                    f"worker={worker}: {type(exc).__name__}: {exc}"
                )

    def sync_stage(stage: str) -> None:
        """Make worker failures collective before any rank can leave a stage."""

        local_error = "; ".join(stage_errors) if stage_errors else None
        if global_world_size > 1:
            import torch.distributed as dist

            gathered = [None] * global_world_size
            dist.all_gather_object(gathered, local_error)
            failures = [error for error in gathered if error]
        else:
            failures = [local_error] if local_error else []
        stage_errors.clear()
        if failures:
            raise RuntimeError(f"benchmark scorer stage {stage!r} failed: " + " | ".join(failures))
        _barrier()

    summary_paths = {
        "aesthetic_quality": (
            root / "aesthetic_quality" / "scoring" / "summary.json",
            root / "aesthetic_quality" / "scoring" / "hpsv3pp_summary.json",
        ),
        "hpsv2_official": (root / "hpsv2_official" / "scoring" / "summary.json",),
        "geneval": (root / "geneval" / "scoring" / "summary.json",),
        "ocr": (root / "ocr" / "scoring" / "summary.json",),
        "cvtg": (root / "cvtg" / "scoring" / "summary.json",),
        "longtext_en": (root / "longtext_en" / "scoring" / "summary.json",),
        "geneval2": (root / "geneval2" / "scoring" / "summary.json",),
    }
    completed_benchmarks = {
        benchmark
        for benchmark in names
        if all(path.is_file() for path in summary_paths[benchmark])
    }

    # A previous failed attempt may have shards from a different scorer
    # topology. Preserve benchmark summaries that were already merged, reset
    # only incomplete scoring output, and always preserve expensive images.
    if global_rank == 0:
        try:
            for benchmark in names:
                if benchmark not in completed_benchmarks:
                    shutil.rmtree(root / benchmark / "scoring", ignore_errors=True)
        except Exception as exc:
            stage_errors.append(f"reset scoring directories: {type(exc).__name__}: {exc}")
    sync_stage("reset-scoring")

    for benchmark in names:
        if benchmark in completed_benchmarks:
            continue
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
            run_worker(
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
            )
            run_worker(
                str(config["hpsv3pp_python"]),
                "score_hpsv3pp.py",
                [
                    *common,
                    "--batch-size", str(int(config.get("hpsv3pp_batch_size", 8))),
                    "--hpsv3pp-root", str(config["hpsv3pp_root"]),
                    "--config-path", str(config["hpsv3pp_config_path"]),
                    "--checkpoint-path", str(config["hpsv3pp_checkpoint_path"]),
                ],
            )
            sync_stage(f"{benchmark}-shards")
            if scorer_active and rank == 0:
                merge_common = ["--merge", "--output-dir", str(scoring), "--world-size", str(world_size)]
                if allow_partial:
                    merge_common.append("--allow-partial")
                run_worker(str(config.get("aesthetic_quality_python", sys.executable)), "score_public_metrics.py", merge_common)
                run_worker(str(config["hpsv3pp_python"]), "score_hpsv3pp.py", merge_common)
            sync_stage(f"{benchmark}-merge")

        elif benchmark == "hpsv2_official":
            scoring = root / benchmark / "scoring"
            scoring.mkdir(parents=True, exist_ok=True)
            run_worker(
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
            )
            sync_stage(f"{benchmark}-shards")
            if scorer_active and rank == 0:
                args = [
                    "--output-dir", str(scoring),
                    "--world-size", str(world_size),
                ]
                if allow_partial:
                    args.append("--allow-partial")
                run_worker(str(config.get("hpsv2_python", sys.executable)), "score_hpsv2.py", ["--merge", *args])
            sync_stage(f"{benchmark}-merge")

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
            run_worker(str(config["ocr_python"]), "score_ocr.py", args)
            sync_stage(f"{benchmark}-shards")
            if scorer_active and rank == 0:
                merge_args = ["--merge", "--output-dir", str(scoring), "--world-size", str(world_size)]
                if allow_partial:
                    merge_args.append("--allow-partial")
                run_worker(str(config["ocr_python"]), "score_ocr.py", merge_args)
            sync_stage(f"{benchmark}-merge")

        elif benchmark == "geneval":
            scoring = root / benchmark / "scoring"
            scoring.mkdir(parents=True, exist_ok=True)
            run_worker(
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
            )
            sync_stage(f"{benchmark}-shards")
            if scorer_active and rank == 0:
                merge_args = [
                    "--merge",
                    "--output-dir", str(scoring),
                    "--world-size", str(world_size),
                ]
                if allow_partial:
                    merge_args.append("--allow-partial")
                run_worker(
                    str(config["geneval_python"]),
                    "score_geneval.py",
                    merge_args,
                )
            sync_stage(f"{benchmark}-merge")

        elif benchmark == "cvtg":
            scoring = root / benchmark / "scoring"
            scoring.mkdir(parents=True, exist_ok=True)
            run_worker(
                str(config["ocr_python"]),
                "score_cvtg.py",
                [
                    "--stage", "ocr", "--results", str(manifest), "--output_dir", str(scoring),
                    "--num_shards", str(world_size), "--shard_id", str(rank),
                    "--ocr_batch", str(int(config.get("cvtg_ocr_batch_size", 32))),
                ],
            )
            run_worker(
                str(config.get("cvtg_clip_python", sys.executable)),
                "score_cvtg.py",
                [
                    "--stage", "clip", "--results", str(manifest), "--output_dir", str(scoring),
                    "--num_shards", str(world_size), "--shard_id", str(rank),
                    "--clip_batch", str(int(config.get("cvtg_clip_batch_size", 32))),
                ],
            )
            sync_stage(f"{benchmark}-shards")
            if scorer_active and rank == 0:
                run_worker(
                    str(config.get("cvtg_clip_python", sys.executable)),
                    "score_cvtg.py",
                    ["--stage", "merge", "--results", str(manifest), "--output_dir", str(scoring)],
                )
            sync_stage(f"{benchmark}-merge")

        elif benchmark == "longtext_en":
            scoring = root / benchmark / "scoring"
            scoring.mkdir(parents=True, exist_ok=True)
            common = [
                "--results", str(manifest), "--output_dir", str(scoring),
                "--world_size", str(world_size), "--rank", str(rank), "--mode", "en",
                "--batch_size", str(int(config.get("longtext_batch_size", 4))),
                "--max_new_tokens", str(int(config.get("longtext_max_new_tokens", 256))),
            ]
            run_worker(str(config["longtext_python"]), "score_longtext.py", common)
            sync_stage(f"{benchmark}-shards")
            if scorer_active and rank == 0:
                merge_args = [
                    "--merge", "--results", str(manifest), "--output_dir", str(scoring),
                    "--world_size", str(world_size), "--mode", "en",
                ]
                if allow_partial:
                    merge_args.append("--allow_partial")
                run_worker(str(config["longtext_python"]), "score_longtext.py", merge_args)
            sync_stage(f"{benchmark}-merge")

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
                if global_rank == 0:
                    benchmark_data.write_text(
                        "".join(
                            json.dumps(row["metadata"], ensure_ascii=False) + "\n"
                            for row in ordered_rows
                        ),
                        encoding="utf-8",
                    )
                sync_stage(f"{benchmark}-partial-manifest")
            run_worker(
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
            )
            sync_stage(f"{benchmark}-shards")
            if scorer_active and rank == 0:
                merge_args = [
                    "--merge",
                    "--results", str(manifest),
                    "--output-dir", str(scoring),
                    "--benchmark-data", str(benchmark_data),
                    "--world-size", str(world_size),
                ]
                if allow_partial:
                    merge_args.append("--allow-partial")
                run_worker(
                    str(config["geneval2_python"]),
                    "score_geneval2.py",
                    merge_args,
                )
            sync_stage(f"{benchmark}-merge")

        if global_rank == 0:
            try:
                timing_path = root / benchmark / "timing.json"
                timing = _read_json(timing_path) if timing_path.is_file() else {}
                timing.update(
                    {
                        "benchmark": benchmark,
                        "scoring_seconds": time.time() - started_at,
                        "world_size": world_size,
                    }
                )
                timing_path.write_text(
                    json.dumps(timing, indent=2),
                    encoding="utf-8",
                )
            except Exception as exc:
                stage_errors.append(
                    f"write {benchmark} timing: {type(exc).__name__}: {exc}"
                )
        sync_stage(f"{benchmark}-timing")

    metrics = {}
    if global_rank == 0:
        try:
            for name in names:
                _flatten(name, _read_json(summary_paths[name][0]), metrics)
                if name == "aesthetic_quality":
                    _flatten(name, _read_json(summary_paths[name][1]), metrics)
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
            }
            for output_name, source_name in canonical.items():
                if source_name in metrics:
                    metrics[output_name] = metrics[source_name]
        except Exception as exc:
            stage_errors.append(f"load summaries: {type(exc).__name__}: {exc}")
    sync_stage("load-summaries")
    return _broadcast(metrics)
