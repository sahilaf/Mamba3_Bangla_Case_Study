"""CPU tests for the sharded-token dataset (numpy only)."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bangla_ssm.data import ShardedTokens


def make_shards(tmp_path: Path):
    rng = np.random.default_rng(0)
    a = rng.integers(0, 32000, 5000, dtype=np.uint16)
    b = rng.integers(0, 32000, 3000, dtype=np.uint16)
    a.tofile(tmp_path / "train_0000.bin")
    b.tofile(tmp_path / "train_0001.bin")
    (tmp_path / "meta.json").write_text(json.dumps({"vocab_size": 32000}))
    return np.concatenate([a, b])


def test_window_crosses_shard_boundary(tmp_path):
    ref = make_shards(tmp_path)
    ds = ShardedTokens(tmp_path, "train")
    assert ds.total == 8000
    w = ds.window(4990, 30)  # spans the 5000 boundary
    assert np.array_equal(w, ref[4990:5020])


def test_sample_batch_deterministic_resume(tmp_path):
    make_shards(tmp_path)
    ds = ShardedTokens(tmp_path, "train")
    x1, y1 = ds.sample_batch(step=7, batch_size=4, seq_len=64, seed=42)
    x2, y2 = ds.sample_batch(step=7, batch_size=4, seq_len=64, seed=42)
    assert np.array_equal(x1, x2) and np.array_equal(y1, y2)
    # y is x shifted by one
    assert np.array_equal(x1[:, 1:], y1[:, :-1])
    # different steps -> different batches
    x3, _ = ds.sample_batch(step=8, batch_size=4, seq_len=64, seed=42)
    assert not np.array_equal(x1, x3)
