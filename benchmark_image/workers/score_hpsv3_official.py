#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def score(args) -> None:
    sys.path.insert(0, str(Path(args.hpsv3_root).resolve()))
    from hpsv3 import HPSv3RewardInferencer

    rows = [row for row in _rows(Path(args.results)) if row["benchmark"] == "hpsv3_official"]
    selected = rows[args.rank :: args.world_size]
    inferencer = HPSv3RewardInferencer(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        device="cuda",
    )
    scored = []
    for start in range(0, len(selected), args.batch_size):
        batch = selected[start : start + args.batch_size]
        rewards = inferencer.reward(
            prompts=[row["prompt"] for row in batch],
            image_paths=[row["image_path"] for row in batch],
        ).detach().float().cpu()
        for row, reward in zip(batch, rewards):
            values = reward.flatten().tolist()
            scored.append(
                {
                    "artifact_id": row["artifact_id"],
                    "category": row["metadata"]["category"],
                    "prompt_index": row["prompt_index"],
                    "image_path": row["image_path"],
                    "prompt": row["prompt"],
                    "reward": values[0],
                    "raw_model_output": values,
                }
            )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"rank_{args.rank:05d}.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in scored),
        encoding="utf-8",
    )


def merge(args) -> dict:
    output = Path(args.output_dir)
    rows = []
    for rank in range(args.world_size):
        path = output / f"rank_{rank:05d}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.extend(_rows(path))
    ids = [row["artifact_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate HPSv3 artifact IDs")
    if len(rows) != args.expected_count:
        raise RuntimeError(f"HPSv3 coverage mismatch: {len(rows)} != {args.expected_count}")
    by_category = defaultdict(list)
    for row in rows:
        reward = float(row["reward"])
        if not math.isfinite(reward):
            raise RuntimeError(f"Non-finite HPSv3 reward for {row['artifact_id']}: {reward}")
        by_category[str(row["category"])].append(reward)
    categories = {
        category: {"count": len(values), "mean": sum(values) / len(values)}
        for category, values in sorted(by_category.items())
    }
    overall = sum(value["mean"] for value in categories.values()) / len(categories)
    summary = {
        "benchmark": "hpsv3_official",
        "scorer": "MizzenAI/HPSv3 HPSv3RewardInferencer",
        "model_revision": args.model_revision,
        "count": len(rows),
        "category_count": len(categories),
        "categories": categories,
        "overall": overall,
        "failure_count": 0,
    }
    rows.sort(key=lambda row: int(row["prompt_index"]))
    (output / "results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--results")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hpsv3-root")
    parser.add_argument("--config-path")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--model-revision", default="4f81e3e09edd82fe3c5f636444c721b592a735ca")
    args = parser.parse_args()
    if args.merge:
        print(json.dumps(merge(args), indent=2))
    else:
        required = (args.results, args.hpsv3_root)
        if any(value is None for value in required):
            parser.error("scoring requires --results and --hpsv3-root")
        score(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
