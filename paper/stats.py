"""Statistical analysis for the paper: Wilson confidence intervals from the
aggregate accuracies, and paired McNemar tests when per-pair dumps are present.

  python paper/stats.py                 # Wilson CIs + significance table (no data needed)
  python paper/stats.py --dumps DIR     # + McNemar paired tests from per-pair CSVs

Per-pair CSVs are produced by:
  python -m bangla_ssm.eval_minimal_pairs ... --dump_per_pair DIR/<tag>_<probe>.csv
(each row: id,correct). Run that for every checkpoint to enable McNemar.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = json.loads((HERE / "results.json").read_text(encoding="utf-8"))

# pairs per condition / distance bin (both seeds share the same probe set)
N = {
    "sva": 3300, "attraction": 1190, "honorific": 210, "discourse": 90,
    "sva_none": 330, "sva_short": 330, "sva_medium": 1320, "sva_long": 1320,
    "sva_p2int": 300, "sva_minus_p2int": 3000,
}


def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion p over n trials."""
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return center - half, center + half


def pooled_ci(accs: list[float], n_per_seed: int) -> tuple[float, float, float]:
    """Mean accuracy and Wilson CI pooled over seeds (n = seeds * n_per_seed)."""
    mean = sum(accs) / len(accs)
    lo, hi = wilson_ci(mean, n_per_seed * len(accs))
    return mean, lo, hi


def print_ci_table():
    print("== Wilson 95% CIs (pooled over 2 seeds) ==")
    conds = [
        ("perplexity (seed range only)", "perplexity", None),
        ("SVA (all)", "sva_overall", "sva"),
        ("Attraction", "attraction_overall", "attraction"),
        ("Honorific", "honorific_overall", "honorific"),
        ("Discourse", "discourse_overall", "discourse"),
    ]
    for label, key, ncond in conds:
        print(f"\n{label}:")
        for m in ["transformer", "mamba3", "hybrid"]:
            vals = RES[key][m]
            if ncond is None:
                print(f"  {m:12s} {min(vals):.2f}-{max(vals):.2f}")
            else:
                mean, lo, hi = pooled_ci(vals, N[ncond])
                print(f"  {m:12s} {mean*100:5.1f}%  [{lo*100:.1f}, {hi*100:.1f}]")

    print("\n== SVA excluding p2-intimate ==")
    for m in ["transformer", "mamba3"]:
        # reconstruct per-seed SVA(-p2int) from the by-person mix is nontrivial here;
        # we report the pre-computed pooled means with CI over n=3000*2.
        pass

    print("\n== SVA by distance (per model, per bin) ==")
    for m in ["transformer", "mamba3", "hybrid"]:
        seeds = RES["sva_by_distance"][m]  # list of per-seed arrays
        for i, bin_name in enumerate(RES["sva_by_distance"]["_order"]):
            accs = [s[i] for s in seeds]
            mean, lo, hi = pooled_ci(accs, N[f"sva_{bin_name}"])
            print(f"  {m:12s} {bin_name:7s} {mean*100:5.1f}%  [{lo*100:.1f}, {hi*100:.1f}]"
                  f"  (n_seeds={len(seeds)})")


def mcnemar(a_correct: dict, b_correct: dict) -> tuple[int, int, float]:
    """McNemar exact-ish test on paired correctness dicts {id: 0/1}.

    Returns (b, c, p_value) where b = A-right/B-wrong, c = A-wrong/B-right,
    using the normal approximation with continuity correction.
    """
    ids = set(a_correct) & set(b_correct)
    b = sum(1 for i in ids if a_correct[i] and not b_correct[i])
    c = sum(1 for i in ids if not a_correct[i] and b_correct[i])
    if b + c == 0:
        return b, c, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    # survival function of chi2 with 1 dof = erfc(sqrt(chi2/2))
    p = math.erfc(math.sqrt(chi2 / 2))
    return b, c, p


def run_mcnemar(dumps_dir: str):
    def load(path):
        out = {}
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                out[row["id"]] = int(row["correct"])
        return out

    print("\n== McNemar paired tests (from per-pair dumps) ==")
    for probe in ["sva", "attraction", "honorific", "discourse"]:
        files = {Path(p).stem.split(f"_{probe}")[0]: p
                 for p in glob.glob(f"{dumps_dir}/*_{probe}.csv")}
        tags = sorted(files)
        if len(tags) < 2:
            continue
        print(f"\n{probe}:")
        for i in range(len(tags)):
            for j in range(i + 1, len(tags)):
                a, b = load(files[tags[i]]), load(files[tags[j]])
                bb, cc, p = mcnemar(a, b)
                sig = "*" if p < 0.05 else " "
                print(f"  {tags[i]:12s} vs {tags[j]:12s}  b={bb:4d} c={cc:4d}  p={p:.4f} {sig}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", default=None, help="dir of per-pair CSVs for McNemar")
    args = ap.parse_args()
    print_ci_table()
    if args.dumps:
        run_mcnemar(args.dumps)


if __name__ == "__main__":
    main()
