#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def score(args):
    import torch
    import trl

    if not hasattr(trl, "get_quantization_config"):
        trl.get_quantization_config = lambda model_config: None
    if not hasattr(trl, "get_kbit_device_map"):
        trl.get_kbit_device_map = lambda: None
    sys.path.insert(0, args.hpsv3pp_root)
    from hpsv3.inference import HPSv3RewardInferencer

    inferencer = HPSv3RewardInferencer(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        device="cuda:0",
    )
    rows = read_rows(args.results)[args.rank :: args.world_size]
    output = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        values = inferencer.reward(
            prompts=[row["prompt"] for row in batch],
            image_paths=[row["image_path"] for row in batch],
            iter_step=0.0,
        ).detach().float().cpu()
        if values.ndim > 1:
            values = values[:, 0]
        for row, value in zip(batch, values.tolist()):
            output.append({"artifact_id": row["artifact_id"], "hpsv3pp_mu": float(value)})
    path = Path(args.output_dir) / f"hpsv3pp_rank_{args.rank:05d}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in output) + ("\n" if output else ""), encoding="utf-8")


def merge(args):
    rows = {}
    for rank in range(args.world_size):
        for row in read_rows(Path(args.output_dir) / f"hpsv3pp_rank_{rank:05d}.jsonl"):
            rows.setdefault(row["artifact_id"], row)
    if not args.allow_partial and len(rows) != 1024:
        raise RuntimeError(f"Incomplete HPSv3++ benchmark: {len(rows)}/1024.")
    summary = {
        "hpsv3pp_iter0": sum(row["hpsv3pp_mu"] for row in rows.values()) / len(rows),
        "iter_step": 0.0,
        "num_inputs": len(rows),
        "complete": len(rows) == 1024,
    }
    path = Path(args.output_dir) / "hpsv3pp_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--results")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hpsv3pp-root", default="")
    parser.add_argument("--config-path", default="")
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    merge(args) if args.merge else score(args)


if __name__ == "__main__":
    main()
