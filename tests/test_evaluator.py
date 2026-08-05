from benchmark_image import evaluator


def test_worker_bounds_threads_and_removes_distributed_environment(monkeypatch):
    captured = {}

    monkeypatch.setenv("RANK", "7")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "64")

    def fake_run(command, *, check, env):
        captured.update(command=command, check=check, env=env)

    monkeypatch.setattr(evaluator.subprocess, "run", fake_run)
    evaluator._run_worker("/env/bin/python", "worker.py", ["--flag"], 2)

    assert captured["command"][0] == "/env/bin/python"
    assert captured["check"] is True
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "2"
    assert captured["env"]["OPENBLAS_NUM_THREADS"] == "1"
    assert captured["env"]["OMP_NUM_THREADS"] == "1"
    assert captured["env"]["MKL_NUM_THREADS"] == "1"
    assert "RANK" not in captured["env"]
    assert "WORLD_SIZE" not in captured["env"]
    assert "LOCAL_RANK" not in captured["env"]


def test_worker_preserves_parent_cuda_visibility_mapping(monkeypatch):
    captured = {}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,2,3,4,5,6,7")

    def fake_run(command, *, check, env):
        captured.update(command=command, check=check, env=env)

    monkeypatch.setattr(evaluator.subprocess, "run", fake_run)
    evaluator._run_worker("/env/bin/python", "worker.py", [], 2)

    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "3"
