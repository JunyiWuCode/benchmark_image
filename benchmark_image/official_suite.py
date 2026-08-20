"""Opt-in, source-locked protocols for reportable Z-Image English benchmarks.

This module is deliberately separate from :mod:`benchmark_image.dataset`.
The latter is used synchronously during training and its resolution and sample
defaults are a compatibility contract.  Nothing in this file is activated by
the training-time evaluator unless a caller explicitly imports it.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Iterable, Mapping


ASSET_ROOT = files("benchmark_image").joinpath("assets")
LOCK_PATH = ASSET_ROOT.joinpath(
    "source_locks", "zimage_base_english_official_v1.json"
)
PROFILES = ("official_report", "training_monitor")


@dataclass(frozen=True)
class OfficialProtocol:
    name: str
    prompts: int
    samples_per_prompt: int
    resolution_policy: str
    primary_metric: str

    @property
    def images(self) -> int:
        return self.prompts * self.samples_per_prompt


OFFICIAL_PROTOCOLS = {
    "geneval": OfficialProtocol("geneval", 553, 4, "fallback_1024", "overall"),
    "geneval2": OfficialProtocol("geneval2", 800, 1, "fallback_1024", "soft_tifa_am_gm"),
    "dpgbench": OfficialProtocol("dpgbench", 1065, 4, "fallback_1024", "dpg_score"),
    "tiif_short": OfficialProtocol("tiif_short", 2538, 1, "fallback_1024", "tiif_short"),
    "tiif_long": OfficialProtocol("tiif_long", 2538, 1, "fallback_1024", "tiif_long"),
    "cvtg": OfficialProtocol("cvtg", 2000, 1, "fallback_1024", "word_accuracy_ned_clipscore"),
    "longtext_en": OfficialProtocol("longtext_en", 160, 4, "fallback_1024", "text_score"),
    "oneig_en": OfficialProtocol("oneig_en", 1120, 4, "fallback_1024", "oneig_score"),
    "qwen_image_bench_en": OfficialProtocol("qwen_image_bench_en", 1000, 1, "fallback_1024", "q_judger_overall"),
    "bizgeneval": OfficialProtocol("bizgeneval", 400, 1, "biz_dynamic_original", "gemini_score"),
    "t2i_corebench": OfficialProtocol("t2i_corebench", 1080, 4, "fallback_1024", "corebench_score"),
    "hpsv3_official": OfficialProtocol("hpsv3_official", 12000, 1, "hps_aspect_1024", "hpsv3"),
}


def load_source_lock() -> dict:
    return json.loads(LOCK_PATH.read_text(encoding="utf-8"))


def expected_image_count(benchmarks: Iterable[str] | None = None) -> int:
    names = normalize_benchmarks(benchmarks)
    return sum(OFFICIAL_PROTOCOLS[name].images for name in names)


def normalize_benchmarks(benchmarks: Iterable[str] | None = None) -> tuple[str, ...]:
    names = tuple(benchmarks or OFFICIAL_PROTOCOLS)
    unknown = sorted(set(names) - set(OFFICIAL_PROTOCOLS))
    if unknown:
        raise ValueError(f"Unknown official benchmark(s): {unknown}")
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate official benchmark(s): {names}")
    return names


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    paths = sorted(
        item for item in path.rglob("*")
        if item.is_file() and item.name != ".DS_Store"
    )
    for item in paths:
        relative = item.relative_to(path).as_posix().encode()
        data = item.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest(), len(paths)


def _git_head(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _source_file(source_root: Path, relative: str) -> Path:
    if source_root.is_file():
        return source_root
    candidate = source_root / relative
    if relative == "benchmark" and not candidate.exists() and any(source_root.glob("benchmark_*.json")):
        return source_root
    if not candidate.exists():
        raise FileNotFoundError(f"Locked source path is missing: {candidate}")
    return candidate


def verify_sources(source_dirs: Mapping[str, str | Path]) -> dict:
    """Verify every external source and both packaged prompt assets.

    ``source_dirs`` maps lock source names to checkout roots.  The CVTG and
    Qwen prompt files are shipped by this package, so those two keys are not
    required.  A source checkout may be newer only if its locked file bytes
    remain identical; the result records both the requested and observed HEAD.
    """
    lock = load_source_lock()
    results = {}
    packaged = {
        "cvtg": Path(str(ASSET_ROOT.joinpath("cvtg", "cvtg_prompts.jsonl"))),
        "qwen_image_bench_en": Path(str(ASSET_ROOT.joinpath("qwen_image_bench", "prompts_cn.jsonl"))),
    }
    for name, spec in lock["sources"].items():
        if name in packaged:
            target = packaged[name]
            root = None
        else:
            if name not in source_dirs:
                raise KeyError(f"Missing --source mapping for {name!r}")
            root = Path(source_dirs[name]).resolve()
            target = _source_file(root, spec["path"])
        if target.is_dir():
            observed, file_count = _tree_sha256(target)
            expected = spec["tree_sha256"]
            if file_count != int(spec["files"]):
                raise RuntimeError(
                    f"{name} file count mismatch: {file_count} != {spec['files']}"
                )
        else:
            observed, expected = _sha256(target), spec["sha256"]
            file_count = 1
        if observed != expected:
            raise RuntimeError(
                f"{name} source hash mismatch: observed={observed}, expected={expected}, path={target}"
            )
        results[name] = {
            "path": str(target),
            "sha256": observed,
            "files": file_count,
            "locked_revision": spec["revision"],
            "observed_revision": _git_head(root if root and root.is_dir() else root.parent) if root else "packaged",
        }
    return results


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _round_stride(value: int, stride: int) -> int:
    # Python's round is intentional: this matches BizGenEval upstream.
    return max(stride, round(value / stride) * stride)


def biz_dynamic_original(
    reference_wh: str | None,
    *,
    aspect_ratio: str = "1:1",
    max_pixels: int = 2048 * 2048,
    stride: int = 32,
) -> tuple[int, int]:
    """Return ``(height, width)`` matching BizGenEval dynamic_original."""
    parts = re.split(r"[xX*×]", str(reference_wh).strip()) if reference_wh else []
    if len(parts) == 2:
        width, height = (int(part.strip()) for part in parts)
    else:
        aspect_ratio = aspect_ratio or "1:1"
        try:
            ratio_width, ratio_height = (float(part) for part in aspect_ratio.split(":"))
            ratio = ratio_width / ratio_height
        except (TypeError, ValueError, ZeroDivisionError) as error:
            raise ValueError(
                f"Invalid BizGenEval resolution: reference={reference_wh!r}, aspect_ratio={aspect_ratio!r}"
            ) from error
        height = max(1, int(math.sqrt(max_pixels / ratio)))
        width = int(height * ratio)
    if width * height > max_pixels:
        scale = math.sqrt(max_pixels / (width * height))
        width, height = max(1, int(width * scale)), max(1, int(height * scale))
    width, height = _round_stride(width, stride), _round_stride(height, stride)
    while width * height > max_pixels and (width > stride or height > stride):
        if width >= height:
            width = max(stride, width - stride)
        else:
            height = max(stride, height - stride)
        width, height = _round_stride(width, stride), _round_stride(height, stride)
    return height, width


def hps_aspect_1024(aspect_ratio: float) -> tuple[int, int]:
    """Return HPSv3 official ``(height, width)`` rounded down to 64."""
    ratio = float(aspect_ratio)
    if ratio <= 0:
        raise ValueError(f"HPSv3 aspect_ratio must be positive, got {ratio}")
    height = int(1024 / math.sqrt(ratio) // 64 * 64)
    width = int(1024 * math.sqrt(ratio) // 64 * 64)
    return max(64, height), max(64, width)


def _record(
    benchmark: str,
    prompt_index: int,
    sample_index: int,
    prompt: str,
    metadata: dict,
    height: int = 1024,
    width: int = 1024,
) -> dict:
    return {
        "benchmark": benchmark,
        "prompt_index": int(prompt_index),
        "sample_index": int(sample_index),
        "artifact_id": f"{benchmark}:{prompt_index}:{sample_index}",
        "prompt": str(prompt),
        "height": int(height),
        "width": int(width),
        "resolution_policy": OFFICIAL_PROTOCOLS[benchmark].resolution_policy,
        "metadata": metadata,
    }


def _expand(rows: list[tuple[str, dict, int, int]], benchmark: str, samples: int) -> list[dict]:
    records = []
    for index, (prompt, metadata, height, width) in enumerate(rows):
        for sample in range(samples):
            records.append(_record(benchmark, index, sample, prompt, metadata, height, width))
    return records


def _load_simple_jsonl(path: Path, benchmark: str, samples: int, prompt_key: str = "prompt") -> list[dict]:
    rows = [(str(row[prompt_key]), row, 1024, 1024) for row in _jsonl(path)]
    return _expand(rows, benchmark, samples)


def _load_records(name: str, sources: Mapping[str, Path], samples: int) -> list[dict]:
    if name == "geneval":
        return _load_simple_jsonl(sources["geneval"] / "prompts/evaluation_metadata.jsonl", name, samples)
    if name == "geneval2":
        return _load_simple_jsonl(sources["geneval2"] / "geneval2_data.jsonl", name, samples)
    if name == "dpgbench":
        data = json.loads((sources[name] / "src/eval/dpgbench/eval_prompts/dpgbench_prompts.json").read_text(encoding="utf-8"))
        rows = [(prompt, {"source_id": source_id}, 1024, 1024) for source_id, prompt in data.items()]
        return _expand(rows, name, samples)
    if name in {"tiif_short", "tiif_long"}:
        rows = []
        prompt_key = "short_description" if name.endswith("short") else "long_description"
        for path in sorted((sources["tiif"] / "data/test_prompts").glob("*.jsonl")):
            for local_index, row in enumerate(_jsonl(path)):
                metadata = dict(row)
                metadata.update({"dimension": path.stem.removesuffix("_prompts"), "local_index": local_index, "variant": prompt_key})
                rows.append((str(row[prompt_key]), metadata, 1024, 1024))
        return _expand(rows, name, samples)
    if name == "cvtg":
        path = Path(str(ASSET_ROOT.joinpath("cvtg", "cvtg_prompts.jsonl")))
        return _load_simple_jsonl(path, name, samples)
    if name == "longtext_en":
        return _load_simple_jsonl(sources[name] / "textbench/text_prompts.jsonl", name, samples)
    if name == "oneig_en":
        with (sources[name] / "OneIG-Bench.csv").open(encoding="utf-8", newline="") as handle:
            data = list(csv.DictReader(handle))
        rows = [(str(row["prompt_en"]), row, 1024, 1024) for row in data]
        return _expand(rows, name, samples)
    if name == "qwen_image_bench_en":
        path = Path(str(ASSET_ROOT.joinpath("qwen_image_bench", "prompts_cn.jsonl")))
        rows = [(str(row["prompt_en"]), row, 1024, 1024) for row in _jsonl(path)]
        return _expand(rows, name, samples)
    if name == "bizgeneval":
        data = _jsonl(sources[name] / "assets/bizgeneval.jsonl")
        rows = []
        for row in data:
            height, width = biz_dynamic_original(
                row.get("reference_image_wh"), aspect_ratio=row.get("aspect_ratio", "1:1")
            )
            rows.append((str(row["prompt"]), row, height, width))
        return _expand(rows, name, samples)
    if name == "t2i_corebench":
        rows = []
        for path in sorted((sources[name] / "data").glob("*.json")):
            for item_id, row in json.loads(path.read_text(encoding="utf-8")).items():
                metadata = dict(row)
                metadata.update({"item_id": item_id, "subset": path.stem})
                rows.append((str(row["Prompt"]), metadata, 1024, 1024))
        return _expand(rows, name, samples)
    if name == "hpsv3_official":
        rows = []
        source = sources[name]
        benchmark_dir = source / "benchmark" if (source / "benchmark").is_dir() else source
        for path in sorted(benchmark_dir.glob("benchmark_*.json")):
            for category_index, row in enumerate(json.loads(path.read_text(encoding="utf-8"))):
                metadata = dict(row)
                metadata["category_index"] = category_index
                height, width = hps_aspect_1024(row["aspect_ratio"])
                rows.append((str(row["caption"]), metadata, height, width))
        return _expand(rows, name, samples)
    raise AssertionError(name)


def build_records(
    source_dirs: Mapping[str, str | Path],
    benchmarks: Iterable[str] | None = None,
    *,
    profile: str = "official_report",
    smoke_max_prompts_per_benchmark: int | None = None,
) -> list[dict]:
    """Build deterministic image-level records for generation.

    ``training_monitor`` is explicitly non-reportable and reduces every
    benchmark to one image per prompt.  Smoke limiting preserves official
    sample multiplicity so layout/scorer integration is exercised faithfully.
    """
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {PROFILES}, got {profile!r}")
    names = normalize_benchmarks(benchmarks)
    sources = {name: Path(path).resolve() for name, path in source_dirs.items()}
    output = []
    for name in names:
        protocol = OFFICIAL_PROTOCOLS[name]
        samples = protocol.samples_per_prompt if profile == "official_report" else 1
        records = _load_records(name, sources, samples)
        if len(records) != protocol.prompts * samples:
            raise RuntimeError(
                f"{name} expanded to {len(records)} images; expected {protocol.prompts * samples}"
            )
        if smoke_max_prompts_per_benchmark is not None:
            limit = int(smoke_max_prompts_per_benchmark)
            if limit <= 0:
                raise ValueError("smoke_max_prompts_per_benchmark must be positive")
            records = [row for row in records if row["prompt_index"] < limit]
        output.extend(records)
    for global_index, row in enumerate(output):
        row["index"] = global_index
        row["profile"] = profile
        row["reportable"] = profile == "official_report" and smoke_max_prompts_per_benchmark is None
    return output


def _greedy_cover(rows: list[dict], labels, *, minimum: int = 0) -> set[int]:
    remaining = set().union(*(set(labels(row)) for row in rows))
    selected: set[int] = set()
    while remaining:
        candidate = max(
            (row for row in rows if row["prompt_index"] not in selected),
            key=lambda row: len(set(labels(row)) & remaining),
        )
        covered = set(labels(candidate)) & remaining
        if not covered:
            raise RuntimeError(f"Unable to cover smoke labels: {sorted(remaining)}")
        selected.add(int(candidate["prompt_index"]))
        remaining -= covered
    for row in rows:
        if len(selected) >= minimum:
            break
        selected.add(int(row["prompt_index"]))
    return selected


def _dpg_source_family(source_id: str) -> str:
    match = re.match(r"[A-Za-z_]+", source_id)
    return match.group(0).lower() if match else "numeric"


def select_coverage_smoke(records: Iterable[Mapping]) -> list[dict]:
    """Select the plan's minimum coverage smoke while retaining all samples."""
    all_rows = [dict(row) for row in records]
    first_samples = {
        name: [row for row in all_rows if row["benchmark"] == name and int(row["sample_index"]) == 0]
        for name in OFFICIAL_PROTOCOLS
    }
    selected: dict[str, set[int]] = {}
    selected["geneval"] = _greedy_cover(
        first_samples["geneval"], lambda row: {str(row["metadata"]["tag"])}
    )
    selected["geneval2"] = _greedy_cover(
        first_samples["geneval2"],
        lambda row: {
            *(f"skill:{skill}" for skill in row["metadata"].get("skills", [])),
            f"atoms:{min(int(row['metadata'].get('atom_count', 0)), 5)}",
        },
        minimum=8,
    )
    selected["dpgbench"] = _greedy_cover(
        first_samples["dpgbench"],
        lambda row: {"source:" + _dpg_source_family(row["metadata"]["source_id"])},
        minimum=8,
    )
    tiif_indices = _greedy_cover(
        first_samples["tiif_short"],
        lambda row: {str(row["metadata"]["dimension"])},
        minimum=4,
    )
    # Full dimension coverage is useful but unnecessarily expensive for a VLM
    # smoke.  Keep four deterministic matched pairs spanning the sorted suite.
    tiif_indices = set(sorted(tiif_indices)[:4])
    selected["tiif_short"] = tiif_indices
    selected["tiif_long"] = set(tiif_indices)
    selected["cvtg"] = _greedy_cover(
        first_samples["cvtg"],
        lambda row: {
            f"regions:{min(int(row['metadata'].get('region_count', 0)), 4)}",
            f"style:{bool(row['metadata'].get('style'))}",
            f"text_bin:{min(sum(len(str(item)) for item in row['metadata'].get('text', [])) // 10, 3)}",
        },
        minimum=8,
    )
    selected["longtext_en"] = _greedy_cover(
        first_samples["longtext_en"],
        lambda row: {
            f"category:{row['metadata'].get('category')}",
            f"length:{row['metadata'].get('length')}",
        },
        minimum=8,
    )
    selected["oneig_en"] = _greedy_cover(
        first_samples["oneig_en"], lambda row: {str(row["metadata"]["category"])}
    )
    selected["qwen_image_bench_en"] = _greedy_cover(
        first_samples["qwen_image_bench_en"],
        lambda row: {
            item.split("/")[0].strip()
            for item in str(row["metadata"].get("dims_en", "")).split(";")
            if item.strip()
        },
        minimum=10,
    )
    selected["bizgeneval"] = _greedy_cover(
        first_samples["bizgeneval"],
        lambda row: {
            f"domain:{row['metadata'].get('domain')}",
            f"dimension:{row['metadata'].get('dimension')}",
        },
    )
    selected["t2i_corebench"] = _greedy_cover(
        first_samples["t2i_corebench"], lambda row: {str(row["metadata"]["subset"])}
    )
    selected["hpsv3_official"] = _greedy_cover(
        first_samples["hpsv3_official"], lambda row: {str(row["metadata"]["category"])}
    )
    smoke = [
        row for row in all_rows
        if int(row["prompt_index"]) in selected[str(row["benchmark"])]
    ]
    for index, row in enumerate(smoke):
        row["index"] = index
        row["reportable"] = False
        row["smoke_selection"] = "coverage_v1"
    return smoke


def output_path_for_record(root: str | Path, record: Mapping) -> Path:
    """Canonical lossless generation path; scorer layouts are derived later."""
    root = Path(root)
    benchmark = str(record["benchmark"])
    prompt_index = int(record["prompt_index"])
    sample_index = int(record["sample_index"])
    return root / benchmark / "raw_images" / f"{prompt_index:06d}_{sample_index:02d}.png"


def write_records(path: str | Path, records: Iterable[Mapping]) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        for row in records:
            line = json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            handle.write(line)
            digest.update(line.encode())
            count += 1
    return {"path": str(path), "records": count, "sha256": digest.hexdigest()}
