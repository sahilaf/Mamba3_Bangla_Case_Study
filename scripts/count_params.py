"""Verify the two architectures are parameter-matched (non-embedding).

  python scripts/count_params.py configs/transformer_53m.json configs/mamba3_53m.json

Adjust n_layer in the Mamba-3 config until non_embedding counts are within ~2%.
Runs on CPU; no GPU needed (Mamba-3 module construction does not require CUDA,
only its forward pass does).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bangla_ssm.config import ExpConfig  # noqa: E402
from bangla_ssm.models import build_model, count_params  # noqa: E402


def main():
    counts = {}
    for cfg_path in sys.argv[1:]:
        cfg = ExpConfig.load(cfg_path)
        model = build_model(cfg)
        pc = count_params(model)
        counts[cfg.name] = pc
        print(f"{cfg.name:24s} total={pc['total']/1e6:8.2f}M  "
              f"emb={pc['embedding']/1e6:6.2f}M  non_emb={pc['non_embedding']/1e6:8.2f}M")
    if len(counts) == 2:
        (a, pa), (b, pb) = counts.items()
        diff = abs(pa["non_embedding"] - pb["non_embedding"]) / max(
            pa["non_embedding"], pb["non_embedding"])
        print(f"non-embedding mismatch: {diff*100:.1f}%  "
              f"({'OK (<2%)' if diff < 0.02 else 'ADJUST n_layer'})")


if __name__ == "__main__":
    main()
