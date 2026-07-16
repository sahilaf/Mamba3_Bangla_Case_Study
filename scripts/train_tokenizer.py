"""Train a 32k SentencePiece BPE tokenizer on a FineWeb-2 Bengali sample.

  python scripts/train_tokenizer.py --out_prefix /content/drive/MyDrive/bn_data/bn_bpe32k \
      --sample_docs 400000
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--vocab_size", type=int, default=32768)
    ap.add_argument("--sample_docs", type=int, default=400_000)
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-2")
    ap.add_argument("--subset", default="ben_Beng")
    args = ap.parse_args()

    import sentencepiece as spm
    from datasets import load_dataset

    Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(args.dataset, args.subset, split="train", streaming=True)
    ds = ds.shuffle(seed=42, buffer_size=10_000)

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt",
                                     delete=False) as f:
        tmp_path = f.name
        for i, ex in enumerate(ds):
            if i >= args.sample_docs:
                break
            text = ex["text"].replace("\n", " ").strip()
            if text:
                f.write(text + "\n")
    print(f"wrote sample corpus to {tmp_path}")

    spm.SentencePieceTrainer.train(
        input=tmp_path,
        model_prefix=args.out_prefix,
        vocab_size=args.vocab_size,
        model_type="bpe",
        character_coverage=0.9995,
        byte_fallback=True,
        input_sentence_size=2_000_000,
        shuffle_input_sentence=True,
        pad_id=3,
        train_extremely_large_corpus=False,
    )
    print(f"tokenizer written: {args.out_prefix}.model / .vocab")


if __name__ == "__main__":
    main()
