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
