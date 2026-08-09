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


def _rank_output_complete(
    rank_root: Path,
    *,
    rank: int,
    world_size: int,
    expected_ids: set[int],
    backend: str,
) -> bool:
    judged_path = rank_root / "judged.jsonl"
    parsed_path = rank_root / "parsed_scores.jsonl"
    summary_path = rank_root / "summary.json"
    if not all(path.is_file() for path in (judged_path, parsed_path, summary_path)):
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        judged_ids = {int(row["ID"]) for row in _read_jsonl(judged_path)}
        parsed_ids = {int(row["ID"]) for row in _read_jsonl(parsed_path)}
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return (
        int(summary.get("rank", -1)) == rank
        and int(summary.get("world_size", -1)) == world_size
        and int(summary.get("num_rows", -1)) == len(expected_ids)
        and str(summary.get("backend", "pt")) == backend
        and judged_ids == expected_ids
        and parsed_ids == expected_ids
    )


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
    rank_root = Path(args.output_dir) / f"rank_{args.rank:05d}"
    expected_ids = {row["ID"] for row in input_rows}
    if _rank_output_complete(
        rank_root,
        rank=args.rank,
        world_size=args.world_size,
        expected_ids=expected_ids,
        backend=args.backend,
    ):
        print(
            f"Q-Judger rank {args.rank}/{args.world_size} is already complete; "
            "reusing its validated shard outputs."
        )
        return
    if not input_rows:
        _write_jsonl(rank_root / "judged.jsonl", [])
        _write_jsonl(rank_root / "parsed_scores.jsonl", [])
        (rank_root / "summary.json").write_text(
            json.dumps(
                {
                    "rank": args.rank,
                    "world_size": args.world_size,
                    "num_rows": 0,
                    "parse_failures": 0,
                    "image_failures": 0,
                    "backend": args.backend,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return

    official = _load_official(Path(args.qwen_image_bench_root))
    inference_args = SimpleNamespace(
        model=args.model,
        max_batch_size=args.max_batch_size,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.max_batch_size,
    )
    if args.backend == "pt":
        results, parse_failures, parsed_scores, image_failures = (
            official.run_ms_swift_inference(
                inference_args,
                pd.DataFrame(input_rows),
                _metadata_from_rows(shard_rows),
            )
        )
    else:
        from benchmark_image.q_judger_backends import build_judge

        judge = build_judge(
            args.backend,
            model_path=args.model,
            max_batch_size=args.max_batch_size,
            max_new_tokens=args.max_new_tokens,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        results, parse_failures, parsed_scores, image_failures = (
            official._run_batch_inference(
                judge,
                inference_args,
                pd.DataFrame(input_rows),
                _metadata_from_rows(shard_rows),
                desc=f"{args.backend} batch inference",
            )
        )

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
                "backend": args.backend,
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

    shard_backends = {str(row.get("backend", "pt")) for row in shard_summaries}
    if shard_backends != {args.backend}:
        raise RuntimeError(
            "Q-Judger shard backends do not match requested backend: "
            f"requested={args.backend!r}, found={sorted(shard_backends)!r}."
        )

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
            "backend": args.backend,
            "max_batch_size": args.max_batch_size,
            "max_new_tokens": args.max_new_tokens,
            "max_model_len": args.max_model_len,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
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
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--backend", choices=("pt", "vllm", "sglang"), default="pt")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.world_size <= 0 or not 0 <= args.rank < args.world_size:
        raise ValueError(
            f"Invalid rank/world_size: rank={args.rank}, world_size={args.world_size}"
        )
    if args.tensor_parallel_size <= 0:
        raise ValueError(
            f"tensor_parallel_size must be positive, got {args.tensor_parallel_size}."
        )
    if args.backend == "pt" and args.tensor_parallel_size != 1:
        raise ValueError("PtEngine supports tensor_parallel_size=1 only.")
    if args.merge:
        merge_shards(args)
    else:
        score_shard(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
