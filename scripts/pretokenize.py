"""Stream FineWeb-2 Bengali, tokenize, and write uint16 shards.

Run once for train (with a token target) and once for eval:
  python scripts/pretokenize.py --sp_model .../bn_bpe32k.model \
      --out_dir /content/drive/MyDrive/bn_data --target_tokens 2500000000
  python scripts/pretokenize.py --sp_model .../bn_bpe32k.model \
      --out_dir /content/drive/MyDrive/bn_data --split test --target_tokens 30000000

uint16 requires vocab_size <= 65535. Documents are separated by EOS.
Restartable: skips work if the target is already met (checks meta.json).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SHARD_TOKENS = 100_000_000  # 200 MB per shard at uint16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sp_model", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--target_tokens", type=int, required=True)
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-2")
    ap.add_argument("--subset", default="ben_Beng")
    args = ap.parse_args()

    import sentencepiece as spm
    from datasets import load_dataset

    sp = spm.SentencePieceProcessor(model_file=args.sp_model)
    assert sp.vocab_size() <= 65535, "uint16 storage requires vocab <= 65535"
    eos = sp.eos_id() if sp.eos_id() >= 0 else 0

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    done_key = f"{args.split}_tokens"
    if meta.get(done_key, 0) >= args.target_tokens:
        print(f"{done_key} already at {meta[done_key]} >= target; nothing to do")
        return

    ds = load_dataset(args.dataset, args.subset, split=args.split, streaming=True)
    if args.split == "train":
        ds = ds.shuffle(seed=1234, buffer_size=10_000)

    written = 0
    shard_idx = 0
    buf: list[int] = []

    def flush(final=False):
        nonlocal buf, shard_idx, written
        while len(buf) >= SHARD_TOKENS or (final and buf):
            chunk, buf = buf[:SHARD_TOKENS], buf[SHARD_TOKENS:]
            if args.split == "train":
                path = out / f"train_{shard_idx:04d}.bin"
            else:
                path = out / "eval.bin"
            arr = np.array(chunk, dtype=np.uint16)
            if path.exists() and args.split != "train":
                # eval is a single file; append
                with open(path, "ab") as f:
                    arr.tofile(f)
            else:
                arr.tofile(path)
            written += len(chunk)
            shard_idx += 1
            print(f"wrote {path.name} (+{len(chunk)} tok, total {written/1e6:.0f}M)")
            if final and not buf:
                break

    batch = []
    for ex in ds:
        batch.append(ex["text"])
        if len(batch) == 512:
            for ids in sp.encode(batch):
                buf.extend(ids)
                buf.append(eos)
            batch = []
            flush()
            if written + len(buf) >= args.target_tokens:
                break
    if batch:
        for ids in sp.encode(batch):
            buf.extend(ids)
            buf.append(eos)
    # trim to target and finish
    over = written + len(buf) - args.target_tokens
    if over > 0:
        buf = buf[: max(0, len(buf) - over)]
    flush(final=True)

    meta[done_key] = written
    meta["vocab_size"] = sp.vocab_size()
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"done: {written} {args.split} tokens; meta.json updated")


if __name__ == "__main__":
    main()
