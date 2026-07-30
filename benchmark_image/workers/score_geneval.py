#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _numeric_folders(path: Path) -> list[Path]:
    return sorted(
        (child for child in path.iterdir() if child.is_dir() and child.name.isdigit()),
        key=lambda child: int(child.name),
    )


def score(args) -> None:
    source = Path(args.image_root)
    output = Path(args.output_dir)
    shard_root = output / "shards" / f"rank_{args.rank:05d}" / "images"
    shard_root.mkdir(parents=True, exist_ok=True)
    folders = _numeric_folders(source)
    selected = folders[args.rank :: args.world_size]
    for folder in selected:
        link = shard_root / folder.name
        target = folder.resolve()
        if link.is_symlink():
            if link.resolve() != target:
                raise RuntimeError(f"Stale GenEval shard link: {link}")
        elif link.exists():
            raise RuntimeError(f"GenEval shard path is not a symlink: {link}")
        else:
            link.symlink_to(target, target_is_directory=True)

    result_path = output / f"rank_{args.rank:05d}.jsonl"
    subprocess.run(
        [
            sys.executable,
            str(Path(args.geneval_root) / "evaluation" / "evaluate_images.py"),
            str(shard_root),
            "--outfile",
            str(result_path),
            "--model-config",
            args.model_config,
            "--model-path",
            args.model_path,
        ],
        check=True,
    )


def merge(args) -> None:
    output = Path(args.output_dir)
    merged = []
    for rank in range(args.world_size):
        path = output / f"rank_{rank:05d}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        merged.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if not args.allow_partial and len(merged) != 2212:
        raise RuntimeError(f"Incomplete GenEval coverage: {len(merged)}/2212.")
    merged.sort(key=lambda row: row["filename"])
    result_path = output / "geneval_official_results.jsonl"
    result_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in merged),
        encoding="utf-8",
    )
    summarize = [
        sys.executable,
        str(Path(__file__).with_name("summarize_geneval.py")),
        "--results",
        str(result_path),
        "--output",
        str(output / "summary.json"),
    ]
    if args.allow_partial:
        summarize.append("--allow-partial")
    subprocess.run(summarize, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--image-root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--geneval-root")
    parser.add_argument("--model-config")
    parser.add_argument("--model-path")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.merge:
        merge(args)
    else:
        required = (
            args.image_root,
            args.geneval_root,
            args.model_config,
            args.model_path,
        )
        if any(value is None for value in required):
            parser.error("scoring requires image/model paths")
        score(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
