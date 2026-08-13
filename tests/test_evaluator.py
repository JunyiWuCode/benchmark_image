from benchmark_image import evaluator


def test_worker_bounds_threads_and_removes_distributed_environment(monkeypatch):
    captured = {}

    monkeypatch.setenv("RANK", "7")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")
    monkeypatch.setenv("GROUP_RANK", "0")
    monkeypatch.setenv("GROUP_WORLD_SIZE", "1")
    monkeypatch.setenv("ROLE_RANK", "7")
    monkeypatch.setenv("ROLE_WORLD_SIZE", "8")
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "stale-generation-group")
    monkeypatch.setenv("TORCHELASTIC_USE_AGENT_STORE", "True")
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
    assert "LOCAL_WORLD_SIZE" not in captured["env"]
    assert "MASTER_ADDR" not in captured["env"]
    assert "MASTER_PORT" not in captured["env"]
    assert "GROUP_RANK" not in captured["env"]
    assert "GROUP_WORLD_SIZE" not in captured["env"]
    assert "ROLE_RANK" not in captured["env"]
    assert "ROLE_WORLD_SIZE" not in captured["env"]
    assert "TORCHELASTIC_RUN_ID" not in captured["env"]
    assert "TORCHELASTIC_USE_AGENT_STORE" not in captured["env"]


def test_worker_preserves_parent_cuda_visibility_mapping(monkeypatch):
    captured = {}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,2,3,4,5,6,7")

    def fake_run(command, *, check, env):
        captured.update(command=command, check=check, env=env)

    monkeypatch.setattr(evaluator.subprocess, "run", fake_run)
    evaluator._run_worker("/env/bin/python", "worker.py", [], 2)

    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "3"


def test_tensor_parallel_worker_preserves_all_visible_devices(monkeypatch):
    captured = {}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")

    def fake_run(command, *, check, env):
        captured.update(command=command, check=check, env=env)

    monkeypatch.setattr(evaluator.subprocess, "run", fake_run)
    evaluator._run_worker("/env/bin/python", "worker.py", [], None)

    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"


def test_q_judger_defaults_are_isolated_from_training_environment(tmp_path):
    config = evaluator._resolved_config(
        {"far_rl_root": str(tmp_path / "FAR-RL"), "conda_env_root": str(tmp_path / "envs")}
    )
    assert config["q_judger_python"].endswith("envs/q_judger/bin/python")
    assert config["q_judger_vllm_python"].endswith("envs/vllm/bin/python")
    assert config["q_judger_sglang_python"].endswith("envs/sglang/bin/python")
    assert config["qwen_image_bench_root"].endswith(
        "FAR-RL/third_party/reference_repos/Qwen-Image-Bench"
    )
    assert config["q_judger_model"].endswith(
        "FAR-RL/third_party/reference_models/Qwen-Image-Bench"
    )
