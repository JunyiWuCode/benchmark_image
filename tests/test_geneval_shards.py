from types import SimpleNamespace

from benchmark_image.workers import score_geneval


def test_geneval_score_rebuilds_stale_rank_workspace(tmp_path, monkeypatch):
    image_root = tmp_path / "images"
    output_dir = tmp_path / "scoring"
    for index in range(4):
        (image_root / str(index)).mkdir(parents=True)

    stale_root = output_dir / "shards" / "rank_00000" / "images"
    stale_root.mkdir(parents=True)
    (stale_root / "1").symlink_to((image_root / "1").resolve(), target_is_directory=True)

    monkeypatch.setattr(score_geneval.subprocess, "run", lambda *args, **kwargs: None)
    score_geneval.score(
        SimpleNamespace(
            image_root=str(image_root),
            output_dir=str(output_dir),
            rank=0,
            world_size=2,
            geneval_root=str(tmp_path / "geneval"),
            model_config="config.py",
            model_path="model.pth",
        )
    )

    assert sorted(path.name for path in stale_root.iterdir()) == ["0", "2"]
