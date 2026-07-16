"""Token-budget training loop with atomic checkpointing and exact resume.

Designed for Colab sessions that die without warning:
  - checkpoints are written atomically (tmp file + os.replace) every
    --ckpt_interval steps, keeping the last 2;
  - on start, the newest checkpoint in --out_dir is loaded automatically;
  - data order is a pure function of (seed, step), so no dataloader state.

Usage (see notebooks/colab_driver.ipynb):
  python -m bangla_ssm.train --config configs/mamba3_53m.json \
      --data_dir /content/drive/MyDrive/bn_data \
      --out_dir  /content/drive/MyDrive/runs/mamba3_53m \
      --token_budget 1000000000 --batch_size 32 --grad_accum 2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import ExpConfig
from .data import ShardedTokens
from .models import build_model, count_params


def cosine_lr(step: int, total_steps: int, base_lr: float, warmup: int, min_ratio: float = 0.1):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    t = (step - warmup) / max(1, total_steps - warmup)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))


def make_optimizer(model, lr: float, weight_decay: float):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or getattr(p, "_no_weight_decay", False) or "norm" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )


def latest_ckpt(out_dir: Path):
    cks = sorted(out_dir.glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    return cks[-1] if cks else None


def save_ckpt(out_dir: Path, step: int, model, opt, scaler, args_dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f".tmp_ckpt_{step}.pt"
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "args": args_dict,
        },
        tmp,
    )
    os.replace(tmp, out_dir / f"ckpt_{step:08d}.pt")
    # keep last 2
    cks = sorted(out_dir.glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    for old in cks[:-2]:
        old.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--token_budget", type=int, required=True)
    ap.add_argument("--batch_size", type=int, default=32, help="sequences per micro-batch")
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--warmup_frac", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--ckpt_interval", type=int, default=250, help="optimizer steps between checkpoints")
    ap.add_argument("--log_interval", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=0, help="stop early after N optimizer steps (0 = full budget)")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA GPU required (Colab: Runtime > Change runtime type)"
    device = "cuda"
    torch.manual_seed(args.seed)

    cfg = ExpConfig.load(args.config)
    out_dir = Path(args.out_dir)
    data = ShardedTokens(args.data_dir, "train")

    tokens_per_step = args.batch_size * args.grad_accum * cfg.seq_len
    total_steps = args.token_budget // tokens_per_step
    warmup = max(10, int(total_steps * args.warmup_frac))

    use_bf16 = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = None if use_bf16 else torch.amp.GradScaler("cuda")

    model = build_model(cfg, device=device)
    pc = count_params(model)
    print(f"[{cfg.name}] params total={pc['total']/1e6:.2f}M non_emb={pc['non_embedding']/1e6:.2f}M")
    print(f"steps={total_steps} tokens/step={tokens_per_step} warmup={warmup} amp={amp_dtype}")

    opt = make_optimizer(model, args.lr, args.weight_decay)

    start_step = 0
    ck = latest_ckpt(out_dir)
    if ck is not None:
        state = torch.load(ck, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        if scaler is not None and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        start_step = state["step"]
        print(f"resumed from {ck.name} at step {start_step}")

    log_path = out_dir / "train_log.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.train()
    loss_ema, t0, tokens_since = None, time.time(), 0

    end_step = total_steps if args.max_steps == 0 else min(total_steps, start_step + args.max_steps)
    for step in range(start_step, end_step):
        lr = cosine_lr(step, total_steps, args.lr, warmup)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        for micro in range(args.grad_accum):
            x, y = data.sample_batch(step * args.grad_accum + micro, args.batch_size, cfg.seq_len, args.seed)
            x = torch.from_numpy(x).to(device, non_blocking=True)
            y = torch.from_numpy(y).to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=amp_dtype):
                logits = model(x).logits
                loss = F.cross_entropy(
                    logits.float().view(-1, logits.size(-1)), y.view(-1)
                ) / args.grad_accum
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if scaler is not None:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if scaler is not None:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()

        loss_val = loss.item() * args.grad_accum
        loss_ema = loss_val if loss_ema is None else 0.98 * loss_ema + 0.02 * loss_val
        tokens_since += tokens_per_step

        if (step + 1) % args.log_interval == 0:
            dt = time.time() - t0
            tps = tokens_since / dt
            rec = {
                "step": step + 1,
                "loss": round(loss_val, 4),
                "loss_ema": round(loss_ema, 4),
                "lr": lr,
                "tok_per_sec": int(tps),
                "tokens": (step + 1) * tokens_per_step,
            }
            print(json.dumps(rec))
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            t0, tokens_since = time.time(), 0

        if (step + 1) % args.ckpt_interval == 0 or (step + 1) == end_step:
            save_ckpt(out_dir, step + 1, model, opt, scaler, vars(args))

    print(f"done at step {end_step}/{total_steps} "
          f"({end_step * tokens_per_step / 1e9:.2f}B / {args.token_budget / 1e9:.2f}B tokens)")


if __name__ == "__main__":
    main()
