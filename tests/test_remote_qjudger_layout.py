import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from benchmark_image.evaluator import _remote_worker_layout


def test_remote_qjudger_uses_one_distributed_worker_per_replica():
    urls = "http://node0:18094,http://node1:18094"
    assert _remote_worker_layout(0, 16, urls) == (2, 0, True)
    assert _remote_worker_layout(1, 16, urls) == (2, 1, True)
    assert _remote_worker_layout(2, 16, urls) == (2, 2, False)
    assert _remote_worker_layout(15, 16, urls) == (2, 15, False)


def test_remote_qjudger_rejects_an_empty_replica_list():
    with pytest.raises(ValueError, match="at least one URL"):
        _remote_worker_layout(0, 16, "")
