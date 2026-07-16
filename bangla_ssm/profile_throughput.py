"""Measure real tokens/sec for a config on the currently assigned GPU,
then back-calculate an affordable token budget.

Run this BEFORE committing to a token budget (plan §6):
  python -m bangla_ssm.profile_throughput --config configs/mamba3_53m.json \
      --batch_size 32 --grad_accum 2 --steps 30
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from .config import ExpConfig
from .models import build_model, count_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup_steps", type=int, default=5)
    args = ap.parse_args()

    assert torch.cuda.is_available()
    cfg = ExpConfig.load(args.config)
    gpu = torch.cuda.get_device_name(0)
    use_bf16 = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = None if use_bf16 else torch.amp.GradScaler("cuda")

    model = build_model(cfg, device="cuda")
    pc = count_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    rng = np.random.default_rng(0)
    tokens_per_step = args.batch_size * args.grad_accum * cfg.seq_len

    def one_step():
        opt.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            x = torch.from_numpy(
                rng.integers(0, cfg.vocab_size, (args.batch_size, cfg.seq_len))
            ).to("cuda")
            with torch.autocast("cuda", dtype=amp_dtype):
                logits = model(x).logits
                loss = F.cross_entropy(
                    logits.float().view(-1, logits.size(-1)), x.view(-1)
                ) / args.grad_accum
            (scaler.scale(loss) if scaler else loss).backward()
        if scaler:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()

    for _ in range(args.warmup_steps):
        one_step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(args.steps):
        one_step()
    torch.cuda.synchronize()
    dt = time.time() - t0

    tps = args.steps * tokens_per_step / dt
    mem = torch.cuda.max_memory_allocated() / 2**30
    print(f"GPU: {gpu}  amp={amp_dtype}  peak_mem={mem:.1f}GiB")
    print(f"model: {cfg.name}  non_emb={pc['non_embedding']/1e6:.1f}M  seq={cfg.seq_len}")
    print(f"throughput: {tps/1e3:.1f}k tokens/sec")
    for hours in (2, 4, 8):
        print(f"  {hours}h session  -> {tps*3600*hours/1e9:.2f}B tokens")
    for budget in (0.5e9, 1e9, 2e9):
        print(f"  {budget/1e9:.1f}B tokens -> {budget/tps/3600:.1f} GPU-hours")


if __name__ == "__main__":
    main()
