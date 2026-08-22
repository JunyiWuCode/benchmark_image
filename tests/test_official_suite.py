from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from benchmark_image.official_layouts import audit_raw_images, materialize_official_layouts
from benchmark_image.official_suite import (
    OFFICIAL_PROTOCOLS,
    biz_dynamic_original,
    build_records,
    expected_image_count,
    hps_aspect_1024,
    output_path_for_record,
    select_coverage_smoke,
    write_records,
)


def test_official_total_and_training_protocol_is_not_mutated():
    assert len(OFFICIAL_PROTOCOLS) == 12
    assert expected_image_count() == 37188
    lock = __import__("benchmark_image.official_suite", fromlist=["load_source_lock"]).load_source_lock()
    assert set(lock["protocols"]) == set(OFFICIAL_PROTOCOLS)
    for name, protocol in OFFICIAL_PROTOCOLS.items():
        assert lock["protocols"][name] == {
            "prompts": protocol.prompts,
            "samples_per_prompt": protocol.samples_per_prompt,
            "expected_images": protocol.images,
            "resolution_policy": protocol.resolution_policy,
            "primary_metric": protocol.primary_metric,
        }
    from benchmark_image.protocols import BENCHMARKS

    assert BENCHMARKS["geneval"].image_count == 2212
    assert "dpgbench" not in BENCHMARKS


def test_official_lock_and_record_handoff_do_not_use_content_hashes(tmp_path: Path):
    module = Path(__import__("benchmark_image.official_suite", fromlist=["__file__"]).__file__)
    lock = module.parent / "assets/source_locks/zimage_base_english_official_v1.json"
    assert "sha256" not in module.read_text(encoding="utf-8").lower()
    assert "sha256" not in lock.read_text(encoding="utf-8").lower()
    summary = write_records(tmp_path / "records.jsonl", [{"prompt": "one"}])
    assert summary == {"path": str(tmp_path / "records.jsonl"), "records": 1}


@pytest.mark.parametrize(
    ("reference", "expected"),
    [("2400x1800", (1760, 2368)), ("1024x1024", (1024, 1024))],
)
def test_biz_resolution_matches_upstream(reference, expected):
    assert biz_dynamic_original(reference) == expected


def test_biz_missing_reference_matches_upstream_aspect_fallback():
    assert biz_dynamic_original(None, aspect_ratio="16:9") == (1536, 2720)


@pytest.mark.parametrize(
    ("aspect", "expected"),
    [(1.0, (1024, 1024)), (16 / 9, (768, 1344)), (9 / 16, (1344, 768))],
)
def test_hps_resolution_matches_upstream(aspect, expected):
    assert hps_aspect_1024(aspect) == expected


def test_smoke_keeps_official_sample_multiplicity(tmp_path: Path):
    root = tmp_path / "geneval"
    prompts = root / "prompts"
    prompts.mkdir(parents=True)
    rows = [{"prompt": f"prompt {index}"} for index in range(553)]
    (prompts / "evaluation_metadata.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    records = build_records(
        {"geneval": root},
        ["geneval"],
        smoke_max_prompts_per_benchmark=2,
    )
    assert len(records) == 8
    assert {row["sample_index"] for row in records} == {0, 1, 2, 3}
    assert all(not row["reportable"] for row in records)
    assert output_path_for_record(tmp_path, records[0]).name == "000000_00.png"


def test_fallback_resolution_override_is_explicit_and_scoped(tmp_path: Path):
    root = tmp_path / "geneval"
    prompts = root / "prompts"
    prompts.mkdir(parents=True)
    (prompts / "evaluation_metadata.jsonl").write_text(
        "".join(json.dumps({"prompt": str(index)}) + "\n" for index in range(553)),
        encoding="utf-8",
    )
    records = build_records(
        {"geneval": root},
        ["geneval"],
        fallback_resolution=512,
    )
    assert {(row["height"], row["width"]) for row in records} == {(512, 512)}
    assert {row["resolution_policy"] for row in records} == {"fallback_512"}


@pytest.mark.parametrize("value", (0, 510))
def test_fallback_resolution_override_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="positive multiple of 16"):
        build_records({}, ["qwen_image_bench_en"], fallback_resolution=value)


def test_training_monitor_reduces_to_one_image_per_prompt(tmp_path: Path):
    root = tmp_path / "geneval2"
    root.mkdir()
    (root / "geneval2_data.jsonl").write_text(
        "".join(json.dumps({"prompt": str(index)}) + "\n" for index in range(800)),
        encoding="utf-8",
    )
    records = build_records({"geneval2": root}, ["geneval2"], profile="training_monitor")
    assert len(records) == 800
    assert all(row["sample_index"] == 0 for row in records)
    assert all(not row["reportable"] for row in records)


def test_official_qwen_english_records_carry_language_contract():
    records = build_records({}, ["qwen_image_bench_en"])
    assert len(records) == 1000
    assert {row["metadata"]["language"] for row in records} == {"en"}


def test_oneig_coverage_smoke_includes_style_scorable_anime(tmp_path: Path):
    root = tmp_path / "oneig"
    (root / "scripts/style").mkdir(parents=True)
    categories = (
        [("Anime_Stylization", 245), ("Portrait", 244), ("General_Object", 206),
         ("Text_Rendering", 200), ("Knowledge_Reasoning", 225)]
    )
    rows = []
    for category, count in categories:
        rows.extend(
            {"category": category, "id": f"{index:03d}", "prompt_en": f"{category} {index}"}
            for index in range(count)
        )
    import csv
    with (root / "OneIG-Bench.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("category", "id", "prompt_en"))
        writer.writeheader()
        writer.writerows(rows)
    with (root / "scripts/style/style.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("id", "class"))
        writer.writeheader()
        writer.writerows({"id": f"{index:03d}", "class": "fauvism" if index == 49 else ""} for index in range(245))
    records = build_records({"oneig_en": root}, ["oneig_en"])
    smoke = select_coverage_smoke(records)
    anime = [row for row in smoke if row["metadata"]["category"] == "Anime_Stylization"]
    assert {row["metadata"]["id"] for row in anime} == {"049"}
    assert {row["sample_index"] for row in anime} == {0, 1, 2, 3}


def test_official_grid_layouts_are_lossless_and_auditable(tmp_path: Path):
    image_root = tmp_path / "raw"
    records = []
    for benchmark, metadata in (
        ("dpgbench", {"source_id": "partiprompts1.txt"}),
        ("oneig_en", {"category": "Portrait", "id": "007"}),
    ):
        for sample in range(4):
            row = {
                "benchmark": benchmark,
                "prompt_index": 0,
                "sample_index": sample,
                "artifact_id": f"{benchmark}:0:{sample}",
                "prompt": "prompt",
                "height": 32,
                "width": 48,
                "metadata": metadata,
            }
            path = output_path_for_record(image_root, row)
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (48, 32), (sample * 20, 0, 0)).save(path)
            records.append(row)
    assert audit_raw_images(records, image_root)["complete"]
    layout = tmp_path / "layouts"
    materialize_official_layouts(records, image_root, layout, model_name="model")
    with Image.open(layout / "dpgbench" / "partiprompts1.png") as image:
        assert image.size == (96, 64)
        assert image.format == "PNG"
    with Image.open(layout / "oneig_en" / "human" / "model" / "007.webp") as image:
        assert image.size == (96, 64)
        assert image.format == "WEBP"
