from __future__ import annotations

import json
import os
from pathlib import Path

from benchmark_image.protocols import BENCHMARKS, normalize_benchmarks


def output_path_for_record(root: str | Path, record: dict) -> Path:
    root = Path(root)
    benchmark = str(record["benchmark"])
    prompt_index = int(record["prompt_index"])
    sample_index = int(record["sample_index"])
    if benchmark == "geneval":
        return root / benchmark / "generation" / "geneval_images" / f"{prompt_index:05d}" / "samples" / f"{sample_index:04d}.png"
    if benchmark == "hpsv2_official":
        metadata = json.loads(record["metadata_json"]) if isinstance(record.get("metadata_json"), str) else record["metadata"]
        return root / benchmark / "generation" / "hpsv2_images" / str(metadata["style"]) / f"{int(metadata['style_index']):05d}.jpg"
    if benchmark in {
        "ocr",
        "aesthetic_quality",
        "cvtg",
        "longtext_en",
        "geneval2",
        "qwen_image_bench",
    }:
        return root / benchmark / "generation" / "images" / f"{prompt_index:05d}_{sample_index:04d}.png"
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def write_rank_manifest(root: str | Path, rank: int, rows: list[dict]) -> Path:
    path = Path(root) / "_rank_manifests" / f"rank_{int(rank):05d}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)
    return path


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def merge_generation_manifests(
    root: str | Path,
    benchmarks,
    *,
    expected_world_size: int,
    allow_partial: bool = False,
) -> dict[str, dict]:
    root = Path(root)
    names = normalize_benchmarks(benchmarks)
    rank_paths = [
        root / "_rank_manifests" / f"rank_{rank:05d}.jsonl"
        for rank in range(int(expected_world_size))
    ]
    missing_rank_paths = [str(path) for path in rank_paths if not path.is_file()]
    if missing_rank_paths:
        raise FileNotFoundError("Missing rank generation manifests:\n" + "\n".join(missing_rank_paths))

    rows_by_id = {}
    for path in rank_paths:
        for row in _read_jsonl(path):
            rows_by_id.setdefault(str(row["artifact_id"]), row)
    rows = sorted(
        rows_by_id.values(),
        key=lambda row: (
            names.index(str(row["benchmark"])),
            int(row["prompt_index"]),
            int(row["sample_index"]),
        ),
    )

    summaries = {}
    for benchmark in names:
        benchmark_rows = [row for row in rows if row["benchmark"] == benchmark]
        missing_images = [
            str(row["image_path"])
            for row in benchmark_rows
            if not Path(row["image_path"]).is_file()
        ]
        expected = BENCHMARKS[benchmark].image_count
        if not allow_partial and len(benchmark_rows) != expected:
            raise RuntimeError(
                f"Incomplete {benchmark} generation: {len(benchmark_rows)} images, expected {expected}."
            )
        if missing_images:
            raise RuntimeError(
                f"{benchmark} manifest references {len(missing_images)} missing images; "
                f"first={missing_images[0]}"
            )
        output = root / benchmark / "generation"
        output.mkdir(parents=True, exist_ok=True)
        results = output / "results.jsonl"
        with results.open("w", encoding="utf-8") as handle:
            for row in benchmark_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary = {
            "benchmark": benchmark,
            "num_images": len(benchmark_rows),
            "expected_images": expected,
            "complete": len(benchmark_rows) == expected,
            "results_path": str(results),
        }
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        summaries[benchmark] = summary
    return summaries
