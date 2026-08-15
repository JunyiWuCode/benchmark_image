import importlib.util
import json
from pathlib import Path


WORKER = (
    Path(__file__).parents[1]
    / "benchmark_image"
    / "workers"
    / "score_qwen_image_bench.py"
)
SPEC = importlib.util.spec_from_file_location("score_qwen_image_bench", WORKER)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_level1_metric_names_are_stable():
    assert MODULE.LEVEL1_TO_KEY == {
        "Quality": "quality",
        "Aesthetics": "aesthetics",
        "Alignment": "alignment",
        "Real-world Fidelity": "real_world_fidelity",
        "Creative Generation": "creative_generation",
    }


def test_metadata_for_judge_preserves_official_ids_and_dims():
    rows = [
        {
            "metadata": {
                "ID": 7,
                "dims_en": "Quality / Detail / Naturalness",
            }
        }
    ]
    frame = MODULE._metadata_from_rows(rows)
    assert frame.to_dict(orient="records") == [
        {"ID": 7, "dims_en": "Quality / Detail / Naturalness"}
    ]


def test_manifest_language_is_derived_from_generated_rows():
    assert MODULE._manifest_language(
        [{"metadata": {"language": "en"}}, {"metadata": {"language": "en"}}]
    ) == "en"


def test_manifest_rejects_mixed_languages():
    try:
        MODULE._manifest_language(
            [{"metadata": {"language": "cn"}}, {"metadata": {"language": "en"}}]
        )
    except RuntimeError as error:
        assert "one supported language" in str(error)
    else:
        raise AssertionError("Expected a mixed-language manifest to fail")


def test_completed_rank_outputs_can_be_resumed(tmp_path):
    rank_root = tmp_path / "rank_00003"
    rank_root.mkdir()
    rows = [{"ID": 7}, {"ID": 11}]
    for name in ("judged.jsonl", "parsed_scores.jsonl"):
        (rank_root / name).write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
    (rank_root / "summary.json").write_text(
        json.dumps({"rank": 3, "world_size": 8, "num_rows": 2, "input_fingerprint": "abc"}),
        encoding="utf-8",
    )

    assert MODULE._rank_output_complete(
        rank_root,
        rank=3,
        world_size=8,
        expected_ids={7, 11},
        backend="pt",
        input_fingerprint="abc",
    )


def test_incomplete_rank_outputs_are_recomputed(tmp_path):
    rank_root = tmp_path / "rank_00003"
    rank_root.mkdir()
    (rank_root / "judged.jsonl").write_text('{"ID": 7}\n', encoding="utf-8")
    (rank_root / "parsed_scores.jsonl").write_text(
        '{"ID": 7}\n', encoding="utf-8"
    )
    (rank_root / "summary.json").write_text(
        json.dumps({"rank": 3, "world_size": 8, "num_rows": 2}),
        encoding="utf-8",
    )

    assert not MODULE._rank_output_complete(
        rank_root,
        rank=3,
        world_size=8,
        expected_ids={7, 11},
        backend="pt",
        input_fingerprint="abc",
    )


def test_completed_rank_output_is_not_reused_by_another_backend(tmp_path):
    rank_root = tmp_path / "rank_00000"
    rank_root.mkdir()
    for name in ("judged.jsonl", "parsed_scores.jsonl"):
        (rank_root / name).write_text('{"ID": 7}\n', encoding="utf-8")
    (rank_root / "summary.json").write_text(
        json.dumps(
            {
                "rank": 0,
                "world_size": 1,
                "num_rows": 1,
                "backend": "vllm",
            }
        ),
        encoding="utf-8",
    )
    assert not MODULE._rank_output_complete(
        rank_root,
        rank=0,
        world_size=1,
        expected_ids={7},
        backend="sglang",
        input_fingerprint="abc",
    )


def test_completed_rank_output_is_not_reused_for_different_images(tmp_path):
    rank_root = tmp_path / "rank_00000"
    rank_root.mkdir()
    for name in ("judged.jsonl", "parsed_scores.jsonl"):
        (rank_root / name).write_text('{"ID": 7}\n', encoding="utf-8")
    (rank_root / "summary.json").write_text(
        json.dumps(
            {
                "rank": 0,
                "world_size": 1,
                "num_rows": 1,
                "backend": "remote",
                "input_fingerprint": "old-images",
            }
        ),
        encoding="utf-8",
    )
    assert not MODULE._rank_output_complete(
        rank_root,
        rank=0,
        world_size=1,
        expected_ids={7},
        backend="remote",
        input_fingerprint="new-images",
    )


def test_input_fingerprint_tracks_image_contents(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"first")
    rows = [
        {
            "ID": 7,
            "prompt": "a prompt",
            "artifact_id": "qwen_image_bench_0007",
            "image_path": str(image),
        }
    ]
    first = MODULE._input_fingerprint(rows)
    image.write_bytes(b"second")
    assert MODULE._input_fingerprint(rows) != first
