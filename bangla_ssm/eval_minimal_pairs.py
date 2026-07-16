"""Score minimal-pair TSVs (sen vs wrong_sen) with a causal LM.

Correct = full-sentence log-prob(sen) > log-prob(wrong_sen)  (BLiMP standard).
Reports overall accuracy and per-condition breakdowns (distance, tense,
subj_person) — the distance breakdown is the paper's headline analysis.

Works with:
  - our checkpoints:  --config configs/X.json --ckpt ckpt.pt --sp_model tok.model
  - any HF causal LM: --hf_model <repo_or_path>       (for probe validation)

Also accepts MultiBLiMP-style TSVs (only sen/wrong_sen columns required).

  python -m bangla_ssm.eval_minimal_pairs --tsv data/probes/sva.tsv \
      --hf_model flax-community/gpt2-bengali --out results/gpt2bn_sva.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


def load_pairs(tsv: str) -> list[dict]:
    with open(tsv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for r in rows:
        assert r.get("sen") and r.get("wrong_sen"), "TSV needs sen/wrong_sen columns"
    return rows


class OurScorer:
    def __init__(self, config: str, ckpt: str, sp_model: str, device: str):
        import sentencepiece as spm

        from .config import ExpConfig
        from .models import build_model

        self.cfg = ExpConfig.load(config)
        self.model = build_model(self.cfg, device=device)
        state = torch.load(ckpt, map_location=device, weights_only=False)
        self.model.load_state_dict(state["model"])
        self.model.eval()
        self.sp = spm.SentencePieceProcessor(model_file=sp_model)
        self.device = device
        self.bos = self.sp.bos_id() if self.sp.bos_id() >= 0 else None

    @torch.no_grad()
    def logprob(self, text: str) -> float:
        ids = self.sp.encode(text)
        if self.bos is not None:
            ids = [self.bos] + ids
        x = torch.tensor([ids], device=self.device)
        logits = self.model(x[:, :-1]).logits.float()
        lp = F.log_softmax(logits, dim=-1)
        tgt = x[:, 1:]
        return lp.gather(-1, tgt.unsqueeze(-1)).sum().item()


class HFScorer:
    def __init__(self, name_or_path: str, device: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(name_or_path).to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def logprob(self, text: str) -> float:
        enc = self.tok(text, return_tensors="pt").to(self.device)
        ids = enc["input_ids"]
        if ids.size(1) < 2:
            return float("-inf")
        logits = self.model(ids[:, :-1]).logits.float()
        lp = F.log_softmax(logits, dim=-1)
        tgt = ids[:, 1:]
        return lp.gather(-1, tgt.unsqueeze(-1)).sum().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True, nargs="+")
    ap.add_argument("--out", default="")
    # ours
    ap.add_argument("--config")
    ap.add_argument("--ckpt")
    ap.add_argument("--sp_model")
    # external
    ap.add_argument("--hf_model")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.hf_model:
        scorer = HFScorer(args.hf_model, device)
        model_name = args.hf_model
    else:
        assert args.config and args.ckpt and args.sp_model, \
            "need --config/--ckpt/--sp_model or --hf_model"
        scorer = OurScorer(args.config, args.ckpt, args.sp_model, device)
        model_name = args.ckpt

    all_results = {}
    for tsv in args.tsv:
        rows = load_pairs(tsv)
        n_correct = 0
        by = {k: defaultdict(lambda: [0, 0]) for k in ("distance", "tense", "subj_person")}
        for r in rows:
            good = scorer.logprob(r["sen"])
            bad = scorer.logprob(r["wrong_sen"])
            ok = good > bad
            n_correct += ok
            for k in by:
                if r.get(k):
                    by[k][r[k]][0] += ok
                    by[k][r[k]][1] += 1
        res = {
            "n": len(rows),
            "accuracy": round(n_correct / len(rows), 4),
            "breakdown": {
                k: {c: {"acc": round(v[0] / v[1], 4), "n": v[1]} for c, v in d.items()}
                for k, d in by.items() if d
            },
        }
        all_results[Path(tsv).name] = res
        print(f"{tsv}: acc={res['accuracy']:.4f} (n={res['n']})")
        for k, d in res["breakdown"].items():
            parts = ", ".join(f"{c}={v['acc']:.3f}" for c, v in sorted(d.items()))
            print(f"  by {k}: {parts}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps({"model": model_name, "results": all_results}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
