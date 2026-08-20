from argparse import Namespace
import json

from benchmark_image.workers.score_hpsv3_official import merge


def test_hpsv3_merge_uses_unweighted_category_average(tmp_path):
    rows = [
        {"artifact_id": "a", "category": "A", "prompt_index": 0, "reward": 1.0},
        {"artifact_id": "b", "category": "A", "prompt_index": 1, "reward": 3.0},
        {"artifact_id": "c", "category": "B", "prompt_index": 2, "reward": 10.0},
    ]
    (tmp_path / "rank_00000.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    summary = merge(
        Namespace(
            output_dir=str(tmp_path),
            world_size=1,
            expected_count=3,
            model_revision="revision",
        )
    )
    assert summary["categories"]["A"]["mean"] == 2.0
    assert summary["categories"]["B"]["mean"] == 10.0
    assert summary["overall"] == 6.0
