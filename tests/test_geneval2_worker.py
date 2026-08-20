import json
from pathlib import Path

import pytest

from benchmark_image.workers.score_geneval2 import select_benchmark_rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_select_benchmark_rows_uses_official_indices_for_partial_smoke(tmp_path: Path):
    official = [
        {"prompt": f"prompt {index}", "atom_count": 3, "skills": [], "vqa_list": []}
        for index in range(5)
    ]
    source = tmp_path / "geneval2.jsonl"
    _write_jsonl(source, official)
    result_rows = [
        {"prompt_index": 1, "metadata": official[1]},
        {"prompt_index": 4, "metadata": official[4]},
    ]
    assert select_benchmark_rows(result_rows, source) == [official[1], official[4]]


def test_select_benchmark_rows_rejects_metadata_drift(tmp_path: Path):
    official = [{"prompt": "official"}]
    source = tmp_path / "geneval2.jsonl"
    _write_jsonl(source, official)
    with pytest.raises(RuntimeError, match="metadata drift"):
        select_benchmark_rows(
            [{"prompt_index": 0, "metadata": {"prompt": "changed"}}],
            source,
        )
