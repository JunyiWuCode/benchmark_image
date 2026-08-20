#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def select_benchmark_rows(result_rows: list[dict], benchmark_path: Path) -> list[dict]:
    """Resolve smoke/full rows by official source index and reject source drift."""
    official_rows = read_rows(benchmark_path)
    selected = []
    seen_indices = set()
    for row in result_rows:
        index = int(row["prompt_index"])
        if index in seen_indices:
            raise RuntimeError(f"Duplicate GenEval2 prompt_index: {index}")
        if index < 0 or index >= len(official_rows):
            raise RuntimeError(f"GenEval2 prompt_index out of range: {index}")
        seen_indices.add(index)
        official = official_rows[index]
        if row["metadata"] != official:
            raise RuntimeError(f"GenEval2 source metadata drift at prompt_index {index}")
        selected.append(official)
    return selected


def score(args) -> None:
    rows = sorted(read_rows(Path(args.results)), key=lambda row: int(row["prompt_index"]))
    selected = rows[args.rank :: args.world_size]
    output = Path(args.output_dir)
    shard_root = output / "shards" / f"rank_{args.rank:05d}"
    shard_root.mkdir(parents=True, exist_ok=True)
    benchmark_path = shard_root / "benchmark.jsonl"
    mapping_path = shard_root / "image_paths.json"
    benchmark_path.write_text(
        "".join(
            json.dumps(row["metadata"], ensure_ascii=False) + "\n"
            for row in selected
        ),
        encoding="utf-8",
    )
    mapping_path.write_text(
        json.dumps(
            {
                row["prompt"]: str(Path(row["image_path"]).resolve())
                for row in selected
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            str(Path(args.evaluator_script) if args.evaluator_script else Path(args.geneval2_root) / "evaluation.py"),
            "--benchmark_data",
            str(benchmark_path),
            "--image_filepath_data",
            str(mapping_path),
            "--method",
            "soft_tifa_gm",
            "--output_file",
            str(output / f"score_lists_rank_{args.rank:05d}.json"),
        ],
        check=True,
    )


def merge(args) -> None:
    rows = sorted(read_rows(Path(args.results)), key=lambda row: int(row["prompt_index"]))
    benchmark_rows = select_benchmark_rows(rows, Path(args.benchmark_data))
    merged = [None] * len(rows)
    output = Path(args.output_dir)
    for rank in range(args.world_size):
        path = output / f"score_lists_rank_{rank:05d}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        values = json.loads(path.read_text(encoding="utf-8"))
        positions = list(range(rank, len(rows), args.world_size))
        if len(values) != len(positions):
            raise RuntimeError(
                f"GenEval2 rank {rank} returned {len(values)} rows, "
                f"expected {len(positions)}."
            )
        for position, value in zip(positions, values):
            merged[position] = value
    if any(value is None for value in merged):
        raise RuntimeError("Incomplete GenEval2 merged score list.")
    if not args.allow_partial and len(merged) != 800:
        raise RuntimeError(f"Incomplete GenEval2 coverage: {len(merged)}/800.")

    score_path = output / "score_lists.json"
    score_path.write_text(json.dumps(merged), encoding="utf-8")
    selected_benchmark_path = output / "benchmark_selected.jsonl"
    selected_benchmark_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in benchmark_rows),
        encoding="utf-8",
    )
    summary_command = [
        sys.executable,
        str(Path(__file__).with_name("summarize_geneval2.py")),
        "--benchmark_data",
        str(selected_benchmark_path),
        "--score_data",
        str(score_path),
        "--output_json",
        str(output / "summary.json"),
    ]
    if args.allow_partial:
        summary_command.append("--allow_partial")
    subprocess.run(summary_command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--results", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--geneval2-root")
    parser.add_argument("--evaluator-script")
    parser.add_argument("--benchmark-data", required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.merge:
        merge(args)
    else:
        if args.geneval2_root is None:
            parser.error("scoring requires --geneval2-root")
        score(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
