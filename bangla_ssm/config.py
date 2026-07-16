"""Model/experiment config loading.

Configs are plain JSON files (see configs/). Required keys:
  arch: "transformer" | "mamba3"
  vocab_size, d_model, n_layer, seq_len
Transformer extras: n_head, intermediate_size, rope_theta
Mamba-3 extras:     d_state, expand, headdim, is_mimo, mimo_rank
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExpConfig:
    arch: str
    vocab_size: int
    d_model: int
    n_layer: int
    seq_len: int
    tie_embeddings: bool = True
    # transformer
    n_head: int = 8
    intermediate_size: int = 1376
    rope_theta: float = 10000.0
    # mamba3
    d_state: int = 128
    expand: int = 2
    headdim: int = 64
    is_mimo: bool = False
    mimo_rank: int = 4
    # hybrid: which layer indices are attention (rest are Mamba-3)
    attn_layer_idx: list = field(default_factory=list)
    # bookkeeping
    name: str = ""
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @staticmethod
    def load(path: str | Path) -> "ExpConfig":
        path = Path(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        known = {f for f in ExpConfig.__dataclass_fields__ if f != "raw"}
        kwargs = {k: v for k, v in raw.items() if k in known}
        cfg = ExpConfig(raw=raw, **kwargs)
        if not cfg.name:
            cfg.name = path.stem
        if cfg.arch not in ("transformer", "mamba3", "hybrid"):
            raise ValueError(f"Unknown arch: {cfg.arch!r}")
        return cfg
