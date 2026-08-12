import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from benchmark_image.evaluator import (
    _post_remote_control,
    _remote_worker_layout,
    _should_sleep_remote_qjudger,
    _wait_for_sync_markers,
    _write_sync_marker,
)


def test_remote_qjudger_uses_one_distributed_worker_per_replica():
    urls = "http://node0:18094,http://node1:18094"
    assert _remote_worker_layout(0, 16, urls) == (2, 0, True)
    assert _remote_worker_layout(1, 16, urls) == (2, 1, True)
    assert _remote_worker_layout(2, 16, urls) == (2, 2, False)
    assert _remote_worker_layout(15, 16, urls) == (2, 15, False)


def test_remote_qjudger_rejects_an_empty_replica_list():
    with pytest.raises(ValueError, match="at least one URL"):
        _remote_worker_layout(0, 16, "")


def test_remote_qjudger_filesystem_sync(tmp_path):
    markers = [tmp_path / "worker-0.json", tmp_path / "worker-1.json"]
    _write_sync_marker(markers[0], {"rank": 0, "ok": True})
    _write_sync_marker(markers[1], {"rank": 1, "ok": True})

    assert _wait_for_sync_markers(markers, timeout_seconds=0.1) == [
        {"rank": 0, "ok": True},
        {"rank": 1, "ok": True},
    ]


def test_remote_qjudger_filesystem_sync_times_out(tmp_path):
    with pytest.raises(TimeoutError, match="worker-0.json"):
        _wait_for_sync_markers(
            [tmp_path / "worker-0.json"],
            timeout_seconds=0.01,
            poll_seconds=0.001,
        )


def test_remote_control_posts_to_every_replica(monkeypatch):
    seen = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b'{"status": "sleeping"}'

    def urlopen(request, timeout):
        seen.append((request.full_url, request.method, timeout))
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    results = _post_remote_control(
        ["http://node0:18094", "http://node1:18094/"],
        "sleep",
        12,
    )
    assert results == [{"status": "sleeping"}, {"status": "sleeping"}]
    assert sorted(seen) == [
        ("http://node0:18094/sleep", "POST", 12),
        ("http://node1:18094/sleep", "POST", 12),
    ]


def test_remote_qjudger_stays_resident_by_default():
    assert not _should_sleep_remote_qjudger(
        {},
        ["qwen_image_bench", "aesthetic_quality"],
        ["http://node0:18094"],
    )


def test_remote_qjudger_sleep_requires_explicit_opt_in():
    assert _should_sleep_remote_qjudger(
        {"q_judger_sleep_during_other_benchmarks": True},
        ["qwen_image_bench", "aesthetic_quality"],
        ["http://node0:18094"],
    )
