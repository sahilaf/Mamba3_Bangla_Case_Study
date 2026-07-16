"""Pretokenized-shard dataset with deterministic, resumable sampling.

Data layout produced by scripts/pretokenize.py:
    data_dir/
      train_0000.bin, train_0001.bin, ...   (uint16 token ids, flat)
      eval.bin                              (uint16 token ids, flat)
      meta.json                             {"vocab_size": ..., "train_tokens": ...}

Training batches are random windows sampled with an RNG seeded by
(seed, step), so resuming at step S reproduces exactly the batches that
would have been drawn in an uninterrupted run — no dataloader state to save.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class ShardedTokens:
    def __init__(self, data_dir: str | Path, split: str = "train"):
        self.data_dir = Path(data_dir)
        if split == "train":
            paths = sorted(self.data_dir.glob("train_*.bin"))
        else:
            p = self.data_dir / "eval.bin"
            paths = [p] if p.exists() else []
        if not paths:
            raise FileNotFoundError(f"No {split} shards in {self.data_dir}")
        self.shards = [np.memmap(p, dtype=np.uint16, mode="r") for p in paths]
        self.sizes = np.array([len(s) for s in self.shards], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)])
        self.total = int(self.offsets[-1])

    def window(self, start: int, length: int) -> np.ndarray:
        """Read a contiguous window [start, start+length) across shard bounds."""
        out = np.empty(length, dtype=np.uint16)
        filled = 0
        while filled < length:
            pos = start + filled
            si = int(np.searchsorted(self.offsets, pos, side="right") - 1)
            local = pos - int(self.offsets[si])
            take = min(length - filled, int(self.sizes[si]) - local)
            out[filled : filled + take] = self.shards[si][local : local + take]
            filled += take
        return out

    def sample_batch(self, step: int, batch_size: int, seq_len: int, seed: int = 0):
        """Deterministic batch for a given step: (x, y) int64 arrays (B, L)."""
        rng = np.random.default_rng((seed, step))
        max_start = self.total - seq_len - 1
        starts = rng.integers(0, max_start, size=batch_size)
        x = np.stack([self.window(int(s), seq_len) for s in starts]).astype(np.int64)
        y = np.stack(
            [self.window(int(s) + 1, seq_len) for s in starts]
        ).astype(np.int64)
        return x, y

    def sequential_windows(self, seq_len: int, limit_tokens: int | None = None):
        """Non-overlapping windows for eval perplexity."""
        end = self.total - 1
        if limit_tokens is not None:
            end = min(end, limit_tokens)
        for start in range(0, end - seq_len, seq_len):
            x = self.window(start, seq_len).astype(np.int64)
            y = self.window(start + 1, seq_len).astype(np.int64)
            yield x, y


def load_meta(data_dir: str | Path) -> dict:
    return json.loads((Path(data_dir) / "meta.json").read_text(encoding="utf-8"))
