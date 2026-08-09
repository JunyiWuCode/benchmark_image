from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


EXPECTED_PROMPTS = 1000
LEVEL1_TO_KEY = {
    "Quality": "quality",
    "Aesthetics": "aesthetics",
    "Alignment": "alignment",
    "Real-world Fidelity": "real_world_fidelity",
    "Creative Generation": "creative_generation",
}


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_official(root: Path):
    required = ("judge.py", "score_utils.py", "checklists.py")
    missing = [name for name in required if not root.joinpath(name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete Qwen-Image-Bench checkout at {root}: missing={missing}"
        )
    sys.path.insert(0, str(root))
    judge = importlib.import_module("judge")
    score_utils = importlib.import_module("score_utils")
    if score_utils.SCORE_MAP != {0: 0.0, 1: 60.0, 2: 100.0}:
        raise RuntimeError(
            "Unexpected Qwen-Image-Bench score mapping: "
            f"{score_utils.SCORE_MAP!r}"
        )
    return judge


def _metadata_from_rows(rows: list[dict]) -> pd.DataFrame:
    metadata = []
    for row in rows:
        source = row.get("metadata", {})
        metadata.append(
            {
                "ID": int(source["ID"]),
                "dims_en": str(source["dims_en"]),
            }
        )
    return pd.DataFrame(metadata).drop_duplicates(subset=["ID"])


def score_shard(args) -> None:
    all_rows = _read_jsonl(Path(args.results))
    shard_rows = [
        row for index, row in enumerate(all_rows) if index % args.world_size == args.rank
    ]
    input_rows = [
        {
            "ID": int(row["metadata"]["ID"]),
            "prompt": str(row["prompt"]),
            "image_path": str(row["image_path"]),
            "artifact_id": str(row["artifact_id"]),
        }
        for row in shard_rows
    ]
    official = _load_official(Path(args.qwen_image_bench_root))
    inference_args = SimpleNamespace(
        model=args.model,
        max_batch_size=args.max_batch_size,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.max_batch_size,
    )
    results, parse_failures, parsed_scores, image_failures = (
        official.run_ms_swift_inference(
            inference_args,
            pd.DataFrame(input_rows),
            _metadata_from_rows(shard_rows),
        )
    )

    rank_root = Path(args.output_dir) / f"rank_{args.rank:05d}"
    _write_jsonl(rank_root / "judged.jsonl", results)
    _write_jsonl(
        rank_root / "parsed_scores.jsonl",
        [
            {"ID": input_rows[index]["ID"], "scores": scores}
            for index, scores in enumerate(parsed_scores)
        ],
    )
    (rank_root / "summary.json").write_text(
        json.dumps(
            {
                "rank": args.rank,
                "world_size": args.world_size,
                "num_rows": len(input_rows),
                "parse_failures": int(parse_failures),
                "image_failures": int(image_failures),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def merge_shards(args) -> None:
    output_dir = Path(args.output_dir)
    manifest_rows = _read_jsonl(Path(args.results))
    if not args.allow_partial and len(manifest_rows) != EXPECTED_PROMPTS:
        raise RuntimeError(
            f"Qwen-Image-Bench needs {EXPECTED_PROMPTS} generated images, "
            f"got {len(manifest_rows)}."
        )

    judged_by_id = {}
    parsed_by_id = {}
    shard_summaries = []
    for rank in range(args.world_size):
        rank_root = output_dir / f"rank_{rank:05d}"
        required = (
            rank_root / "judged.jsonl",
            rank_root / "parsed_scores.jsonl",
            rank_root / "summary.json",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing Q-Judger rank outputs: {missing}")
        for row in _read_jsonl(required[0]):
            judged_by_id[int(row["ID"])] = row
        for row in _read_jsonl(required[1]):
            parsed_by_id[int(row["ID"])] = row["scores"]
        shard_summaries.append(json.loads(required[2].read_text(encoding="utf-8")))

    expected_ids = {int(row["metadata"]["ID"]) for row in manifest_rows}
    if set(judged_by_id) != expected_ids or set(parsed_by_id) != expected_ids:
        raise RuntimeError(
            "Q-Judger merge IDs do not match generation manifest: "
            f"expected={len(expected_ids)}, judged={len(judged_by_id)}, "
            f"parsed={len(parsed_by_id)}."
        )

    ordered_ids = sorted(expected_ids)
    _write_jsonl(
        output_dir / "judged_results.jsonl",
        [judged_by_id[row_id] for row_id in ordered_ids],
    )
    official = _load_official(Path(args.qwen_image_bench_root))
    scores = official.compute_bench_scores(
        [parsed_by_id[row_id] for row_id in ordered_ids]
    )
    details = {
        "official_aggregation": scores,
        "protocol": {
            "language": "cn",
            "raw_scores": [0, 1, 2, "N/A"],
            "normalized_scores": {"0": 0, "1": 60, "2": 100},
            "q_judger_model": str(args.model),
            "max_batch_size": args.max_batch_size,
            "max_new_tokens": args.max_new_tokens,
        },
    }
    (output_dir / "scores_detail.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "overall": scores["total"],
        **{
            output_key: scores["level1"].get(level1)
            for level1, output_key in LEVEL1_TO_KEY.items()
        },
        "num_images": len(ordered_ids),
        "expected_images": EXPECTED_PROMPTS,
        "complete": len(ordered_ids) == EXPECTED_PROMPTS,
        "parse_failures": sum(row["parse_failures"] for row in shard_summaries),
        "image_failures": sum(row["image_failures"] for row in shard_summaries),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Official Qwen-Image-Bench scorer")
    parser.add_argument("--results", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--qwen-image-bench-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--max-batch-size", type=int, default=24)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.world_size <= 0 or not 0 <= args.rank < args.world_size:
        raise ValueError(
            f"Invalid rank/world_size: rank={args.rank}, world_size={args.world_size}"
        )
    if args.merge:
        merge_shards(args)
    else:
        score_shard(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
