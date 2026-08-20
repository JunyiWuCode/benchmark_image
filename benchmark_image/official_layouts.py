from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

from PIL import Image

from benchmark_image.official_suite import output_path_for_record


ONEIG_FOLDERS = {
    "Anime_Stylization": "anime",
    "Portrait": "human",
    "General_Object": "object",
    "Text_Rendering": "text",
    "Knowledge_Reasoning": "reasoning",
}


def read_records(path: str | Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def audit_raw_images(records: Iterable[Mapping], image_root: str | Path) -> dict:
    rows = list(records)
    failures = []
    observed = defaultdict(int)
    for row in rows:
        path = output_path_for_record(image_root, row)
        try:
            with Image.open(path) as image:
                image.load()
                if image.size != (int(row["width"]), int(row["height"])):
                    failures.append(
                        {"artifact_id": row["artifact_id"], "error": f"size={image.size}", "path": str(path)}
                    )
                if image.format != "PNG":
                    failures.append(
                        {"artifact_id": row["artifact_id"], "error": f"format={image.format}", "path": str(path)}
                    )
        except (FileNotFoundError, OSError) as error:
            failures.append({"artifact_id": row["artifact_id"], "error": str(error), "path": str(path)})
        observed[str(row["benchmark"])] += 1
    return {
        "expected_images": len(rows),
        "audited_images": len(rows) - len(failures),
        "counts": dict(observed),
        "failure_count": len(failures),
        "failures": failures,
        "complete": not failures,
    }


def _link(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    relative = Path(os.path.relpath(source.resolve(), destination.parent.resolve()))
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise RuntimeError(f"Stale layout link: {destination}")
        return
    if destination.exists():
        raise RuntimeError(f"Layout destination already exists and is not a symlink: {destination}")
    destination.symlink_to(relative)


def _grid(sources: list[Path], destination: Path, *, format_name: str) -> None:
    if len(sources) != 4:
        raise RuntimeError(f"A 2x2 official grid requires four images, got {len(sources)}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.open(path).convert("RGB") for path in sources]
    try:
        size = images[0].size
        if any(image.size != size for image in images):
            raise RuntimeError(f"Grid sources have inconsistent sizes: {[image.size for image in images]}")
        expected_size = (size[0] * 2, size[1] * 2)
        if destination.is_file():
            with Image.open(destination) as existing:
                if existing.size == expected_size:
                    return
            raise RuntimeError(f"Existing grid has wrong size: {destination}")
        canvas = Image.new("RGB", expected_size)
        for index, image in enumerate(images):
            canvas.paste(image, ((index % 2) * size[0], (index // 2) * size[1]))
        temporary = destination.with_name(f".{destination.stem}.tmp{destination.suffix}")
        save_kwargs = {"lossless": True} if format_name == "WEBP" else {}
        canvas.save(temporary, format=format_name, **save_kwargs)
        os.replace(temporary, destination)
    finally:
        for image in images:
            image.close()


def materialize_official_layouts(
    records: Iterable[Mapping],
    image_root: str | Path,
    layout_root: str | Path,
    *,
    model_name: str = "Z-Image-Base-cfg4-nfe50",
) -> dict:
    rows = list(records)
    image_root, layout_root = Path(image_root), Path(layout_root)
    prompt_groups = defaultdict(list)
    for row in rows:
        prompt_groups[(str(row["benchmark"]), int(row["prompt_index"]))].append(row)

    counts = defaultdict(int)
    geneval2_mapping = {}
    tiif_indices = defaultdict(set)
    for (benchmark, prompt_index), group in prompt_groups.items():
        group.sort(key=lambda row: int(row["sample_index"]))
        sources = [output_path_for_record(image_root, row) for row in group]
        metadata = group[0]["metadata"]
        if benchmark == "geneval":
            folder = layout_root / "geneval" / f"{prompt_index:05d}"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "metadata.jsonl").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
            for row, source in zip(group, sources):
                _link(source, folder / "samples" / f"{int(row['sample_index']):04d}.png")
        elif benchmark == "geneval2":
            geneval2_mapping[str(group[0]["prompt"])] = str(sources[0].resolve())
        elif benchmark == "dpgbench":
            _grid(sources, layout_root / "dpgbench" / Path(metadata["source_id"]).with_suffix(".png"), format_name="PNG")
        elif benchmark in {"tiif_short", "tiif_long"}:
            variant = "short_description" if benchmark.endswith("short") else "long_description"
            dimension = str(metadata["dimension"])
            local_index = int(metadata["local_index"])
            tiif_indices[dimension].add(local_index)
            _link(sources[0], layout_root / "tiif" / dimension / model_name / variant / f"{local_index}.png")
        elif benchmark in {"cvtg", "longtext_en"}:
            for row, source in zip(group, sources):
                _link(source, layout_root / benchmark / "images" / f"{prompt_index:05d}_{int(row['sample_index']):04d}.png")
        elif benchmark == "oneig_en":
            folder = ONEIG_FOLDERS[str(metadata["category"])]
            _grid(sources, layout_root / "oneig_en" / folder / model_name / f"{metadata['id']}.webp", format_name="WEBP")
        elif benchmark == "qwen_image_bench_en":
            _link(sources[0], layout_root / benchmark / "images" / f"{int(metadata['ID']):06d}.png")
        elif benchmark == "bizgeneval":
            filename = f"{metadata.get('domain', '')}_{metadata.get('dimension', '')}_{metadata['id']}.png"
            _link(sources[0], layout_root / benchmark / "images" / filename)
        elif benchmark == "t2i_corebench":
            for row, source in zip(group, sources):
                filename = f"{metadata['item_id']}-{int(row['sample_index'])}.png"
                _link(source, layout_root / benchmark / model_name / metadata["subset"] / filename)
        elif benchmark == "hpsv3_official":
            category = str(metadata["category"])
            stem = f"{int(metadata['category_index']):05d}"
            destination = layout_root / benchmark / category / f"{stem}.png"
            _link(sources[0], destination)
            destination.with_suffix(".txt").write_text(str(group[0]["prompt"]), encoding="utf-8")
        else:
            raise ValueError(f"No official layout adapter for {benchmark}")
        counts[benchmark] += len(group)

    if geneval2_mapping:
        path = layout_root / "geneval2" / "image_paths.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(geneval2_mapping, indent=2, ensure_ascii=False), encoding="utf-8")
    if tiif_indices:
        path = layout_root / "tiif" / "sample_indices.json"
        path.write_text(
            json.dumps({key: sorted(value) for key, value in sorted(tiif_indices.items())}, indent=2),
            encoding="utf-8",
        )
    summary = {
        "layout_root": str(layout_root.resolve()),
        "model_name": model_name,
        "raw_image_count": len(rows),
        "benchmark_image_counts": dict(counts),
        "geneval2_mapping_count": len(geneval2_mapping),
        "tiif_dimension_count": len(tiif_indices),
    }
    layout_root.mkdir(parents=True, exist_ok=True)
    (layout_root / "layout_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def collect_generation_manifests(
    records: Iterable[Mapping], image_root: str | Path
) -> dict:
    rows = [dict(row) for row in records]
    image_root = Path(image_root)
    expected = {str(row["artifact_id"]): row for row in rows}
    observed = {}
    protocol_paths = sorted((image_root / "manifests").glob("shard_*_of_*/protocol.json"))
    if not protocol_paths:
        raise FileNotFoundError(f"No completed shard protocols under {image_root / 'manifests'}")
    protocols = [json.loads(path.read_text(encoding="utf-8")) for path in protocol_paths]
    shard_count = int(protocols[0]["num_shards"])
    shard_ids = {int(protocol["shard_id"]) for protocol in protocols}
    if shard_ids != set(range(shard_count)):
        raise RuntimeError(f"Incomplete shard set: observed={sorted(shard_ids)}, expected=0..{shard_count - 1}")
    invariant_keys = (
        "pipeline_backend", "num_inference_steps", "guidance_scale",
        "scheduler_shift", "seed", "records_sha256", "suite_record_count",
        "num_shards", "model",
    )
    baseline = {key: protocols[0][key] for key in invariant_keys}
    for protocol in protocols[1:]:
        current = {key: protocol[key] for key in invariant_keys}
        if current != baseline:
            raise RuntimeError(f"Shard protocol mismatch: {current} != {baseline}")
    for protocol_path in protocol_paths:
        results = protocol_path.with_name("results.jsonl")
        if not results.is_file():
            raise FileNotFoundError(results)
        for row in read_records(results):
            artifact_id = str(row["artifact_id"])
            if artifact_id in observed:
                raise RuntimeError(f"Duplicate generated artifact: {artifact_id}")
            observed[artifact_id] = row
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    if missing or extra:
        raise RuntimeError(
            f"Generation coverage mismatch: missing={len(missing)}, extra={len(extra)}, "
            f"first_missing={missing[:1]}, first_extra={extra[:1]}"
        )
    for artifact_id, source in expected.items():
        generated = observed[artifact_id]
        for key in ("benchmark", "prompt_index", "sample_index", "height", "width"):
            if generated[key] != source[key]:
                raise RuntimeError(
                    f"Manifest field mismatch for {artifact_id} {key}: {generated[key]} != {source[key]}"
                )
    audit = audit_raw_images(rows, image_root)
    if not audit["complete"]:
        raise RuntimeError(f"Generated image audit failed: {audit['failure_count']} failures")
    merged_path = image_root / "generation_manifest.jsonl"
    with merged_path.open("w", encoding="utf-8") as handle:
        for source in rows:
            handle.write(json.dumps(observed[str(source["artifact_id"])], ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "complete": True,
        "record_count": len(rows),
        "shard_count": shard_count,
        "manifest": str(merged_path.resolve()),
        "protocol": baseline,
        "image_audit": audit,
    }
    (image_root / "generation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
