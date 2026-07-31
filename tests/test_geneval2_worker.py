import argparse
import json

from benchmark_image.workers import score_geneval2


def test_empty_geneval2_shard_writes_mergeable_score_file(
    tmp_path,
    monkeypatch,
):
    results = tmp_path / "results.jsonl"
    results.write_text(
        json.dumps(
            {
                "prompt_index": 0,
                "prompt": "one object",
                "image_path": str(tmp_path / "image.png"),
                "metadata": {"prompt": "one object"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "scoring"
    args = argparse.Namespace(
        results=str(results),
        output_dir=str(output),
        geneval2_root=str(tmp_path / "geneval2"),
        rank=3,
        world_size=8,
    )

    def unexpected_subprocess(*unused_args, **unused_kwargs):
        raise AssertionError("empty shards must not launch the official evaluator")

    monkeypatch.setattr(score_geneval2.subprocess, "run", unexpected_subprocess)

    score_geneval2.score(args)

    assert json.loads(
        (output / "score_lists_rank_00003.json").read_text(encoding="utf-8")
    ) == []
    assert (
        output / "shards" / "rank_00003" / "benchmark.jsonl"
    ).read_text(encoding="utf-8") == ""
    assert json.loads(
        (
            output / "shards" / "rank_00003" / "image_paths.json"
        ).read_text(encoding="utf-8")
    ) == {}
