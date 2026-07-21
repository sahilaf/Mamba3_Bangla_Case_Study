"""Held-out perplexity on the pretokenized eval split (eval.bin).

  python -m bangla_ssm.eval_ppl --config configs/mamba3_53m.json \
      --ckpt /content/drive/MyDrive/runs/mamba3_53m/ckpt_XXXXXXXX.pt \
      --data_dir /content/drive/MyDrive/bn_data --limit_tokens 20000000
"""

from __future__ import annotations

import argparse
import json
import math

import torch
import torch.nn.functional as F

from .config import ExpConfig
from .data import ShardedTokens
from .models import build_model


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--limit_tokens", type=int, default=None)
    ap.add_argument("--out", default=None, help="optional JSON path to write {ppl, nll, tokens}")
    args = ap.parse_args()

    cfg = ExpConfig.load(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg, device=device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    data = ShardedTokens(args.data_dir, "eval")
    amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

    total_nll, total_tok = 0.0, 0
    xs, ys = [], []

    def flush(xs, ys):
        nonlocal total_nll, total_tok
        x = torch.from_numpy(__import__("numpy").stack(xs)).to(device)
        y = torch.from_numpy(__import__("numpy").stack(ys)).to(device)
        with torch.autocast(device, dtype=amp_dtype, enabled=device == "cuda"):
            logits = model(x).logits
        nll = F.cross_entropy(
            logits.float().view(-1, logits.size(-1)), y.view(-1), reduction="sum"
        )
        total_nll += nll.item()
        total_tok += y.numel()

    for x, y in data.sequential_windows(cfg.seq_len, args.limit_tokens):
        xs.append(x)
        ys.append(y)
        if len(xs) == args.batch_size:
            flush(xs, ys)
            xs, ys = [], []
    if xs:
        flush(xs, ys)

    nll = total_nll / total_tok
    ppl = math.exp(nll)
    print(f"eval tokens={total_tok}  nll/token={nll:.4f}  ppl={ppl:.2f}")
    if args.out:
        from pathlib import Path
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps({"ckpt": args.ckpt, "tokens": total_tok,
                        "nll": round(nll, 6), "ppl": round(ppl, 4)}, indent=2),
            encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
