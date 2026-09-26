"""Tokenization sanity check (camera-ready addition, Reviewer 1).

Reviewer 1: "Because grammatical and ungrammatical verb forms may tokenize
differently, it would be useful to show that the reported agreement effects
are not driven by systematic subword-length differences."

Scoring is total (non-length-normalized) log-probability, so a pair where the
grammatical sentence tokenizes shorter than the ungrammatical one is
mechanically advantaged. This script:

  1. Tokenizes every released minimal pair with the actual trained tokenizer
     and records the signed subword-count delta between sen/wrong_sen.
  2. Cross-tabulates that delta against per-pair correctness (the released
     seed-1 dumps in results/per_pair/ -- the only per-pair granularity we have)
     to quantify how much the confound inflates raw accuracy.
  3. Recomputes the headline SVA-by-distance table restricted to
     length-matched pairs only, to check whether the reported dissociation
     survives once the confound is removed.

Requires `sentencepiece` and network access to fetch the released tokenizer
from the HF checkpoints repo the first time (then cached locally).

  python paper/tokenization_check.py
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

import sentencepiece as spm
from huggingface_hub import hf_hub_download

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROBES = ["sva", "attraction", "honorific", "discourse"]
ARCHS = {"transformer": "tf", "mamba3": "m3", "hybrid": "hybrid"}


def load_tokenizer() -> spm.SentencePieceProcessor:
    path = hf_hub_download("sahilfarib/mamba3-bangla-case-study", "tokenizer/bn_bpe32k.model")
    return spm.SentencePieceProcessor(model_file=path)


def load_tsv(probe: str) -> dict[str, dict]:
    rows = {}
    with open(ROOT / "data" / "probes" / f"{probe}.tsv", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rows[row["id"]] = row
    return rows


def load_correct(arch_tag: str, probe: str) -> dict[str, int]:
    path = ROOT / "results" / "per_pair" / f"{arch_tag}_s1_{probe}.csv"
    out = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["id"]] = int(row["correct"])
    return out


def token_delta(sp: spm.SentencePieceProcessor, sen: str, wrong_sen: str) -> int:
    a = sp.encode(sen, out_type=int)
    b = sp.encode(wrong_sen, out_type=int)
    return len(a) - len(b)


def main():
    sp = load_tokenizer()

    print("== Part 1: token-count delta between grammatical/ungrammatical forms ==\n")
    all_deltas: dict[str, dict[str, int]] = {}
    for probe in PROBES:
        rows = load_tsv(probe)
        deltas = {pid: token_delta(sp, r["sen"], r["wrong_sen"]) for pid, r in rows.items()}
        all_deltas[probe] = deltas
        n = len(deltas)
        n_shorter = sum(1 for d in deltas.values() if d < 0)
        n_longer = sum(1 for d in deltas.values() if d > 0)
        n_equal = n - n_shorter - n_longer
        mean_d = sum(deltas.values()) / n
        print(f"{probe:12s} n={n:5d}  grammatical-shorter={n_shorter:5d} ({100*n_shorter/n:5.1f}%)"
              f"  grammatical-longer={n_longer:5d} ({100*n_longer/n:5.1f}%)"
              f"  equal={n_equal:5d} ({100*n_equal/n:5.1f}%)  mean_delta={mean_d:+.3f}")

    print("\n== Part 2: model accuracy by token-length-delta bucket (seed 1) ==\n")
    for arch, tag in ARCHS.items():
        print(f"-- {arch} --")
        for probe in PROBES:
            correct = load_correct(tag, probe)
            deltas = all_deltas[probe]
            buckets = defaultdict(list)
            for pid, d in deltas.items():
                if pid not in correct:
                    continue
                key = "shorter" if d < 0 else ("longer" if d > 0 else "equal")
                buckets[key].append(correct[pid])
            parts = []
            for key in ["shorter", "equal", "longer"]:
                vals = buckets.get(key, [])
                acc = f"{100*sum(vals)/len(vals):5.1f}% (n={len(vals)})" if vals else "n/a"
                parts.append(f"{key}={acc}")
            print(f"  {probe:12s} " + "  ".join(parts))
        print()

    print("== Part 3: SVA-by-distance, length-matched pairs only vs. all pairs (seed 1) ==\n")
    rows = load_tsv("sva")
    deltas = all_deltas["sva"]
    for arch, tag in ARCHS.items():
        correct = load_correct(tag, "sva")
        print(f"-- {arch} --")
        by_bin_matched = defaultdict(list)
        by_bin_all = defaultdict(list)
        for pid, r in rows.items():
            if pid not in correct:
                continue
            by_bin_all[r["distance"]].append(correct[pid])
            if deltas[pid] == 0:
                by_bin_matched[r["distance"]].append(correct[pid])
        for b in ["none", "short", "medium", "long"]:
            va, vm = by_bin_all[b], by_bin_matched[b]
            acc_a = 100 * sum(va) / len(va)
            acc_m = 100 * sum(vm) / len(vm)
            print(f"  {b:7s} all-pairs={acc_a:5.1f}% (n={len(va):4d})   "
                  f"length-matched-only={acc_m:5.1f}% (n={len(vm):4d})")
        print()


if __name__ == "__main__":
    main()
