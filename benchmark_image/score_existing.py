from __future__ import annotations

import argparse
import json
import os

from benchmark_image.evaluator import evaluate_generated_suite


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score an existing benchmark generation directory."
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--benchmarks", required=True)
    parser.add_argument(
        "--far-rl-root",
        default="/home/hcai/workspace/code/junyiwu/FAR-RL",
    )
    parser.add_argument(
        "--conda-env-root",
        default="/home/hcai/workspace/anaconda3/envs",
    )
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        import torch
        import torch.distributed as dist

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    names = [name.strip() for name in args.benchmarks.split(",") if name.strip()]
    metrics = evaluate_generated_suite(
        args.root,
        names,
        {
            "far_rl_root": args.far_rl_root,
            "conda_env_root": args.conda_env_root,
            "allow_partial": args.allow_partial,
        },
    )
    if local_rank == 0:
        print(json.dumps(metrics, indent=2, sort_keys=True))

    if distributed:
        import torch.distributed as dist

        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
