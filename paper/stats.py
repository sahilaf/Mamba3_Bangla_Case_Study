"""Statistical analysis for the paper: Wilson confidence intervals from the
aggregate accuracies, and paired McNemar tests when per-pair dumps are present.

  python paper/stats.py                             # Wilson CIs + Welch tests
  python paper/stats.py --dumps results/per_pair    # + seed-1 McNemar paired tests

Per-pair CSVs are produced by:
  python -m bangla_ssm.eval_minimal_pairs ... --dump_per_pair DIR/<tag>_<probe>.csv
(each row: id,correct).
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

# pairs per condition / distance bin (all seeds share the same probe set)
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


def welch_t(a: list[float], b: list[float]):
    """Welch's t-test across seeds (proper between-architecture test)."""
    na, nb = len(a), len(b)
    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return ma - mb, float("inf"), 0.0
    t = (ma - mb) / se
    df = (va / na + vb / nb) ** 2 / (
        (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    # two-sided p = I_{df/(df+t^2)}(df/2, 1/2)
    p = _betai(df / 2.0, 0.5, df / (df + t * t))
    return ma - mb, t, p


def _betacf(a, b, x):
    """Continued fraction for the incomplete beta (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (tiny if abs(d) < tiny else d)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((qam + m2) * (a + m2)),
                   -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1.0 + aa * d
            d = 1.0 / (tiny if abs(d) < tiny else d)
            c = 1.0 + aa / c
            c = tiny if abs(c) < tiny else c
            h *= d * c
        if abs(d * c - 1.0) < 3e-14:
            break
    return h


def _betai(a, b, x):
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                  + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1 - x) / b


def print_between_arch():
    print("\n== Between-architecture Welch t-test ACROSS SEEDS (proper test) ==")
    conds = [("perplexity", "perplexity", False), ("SVA", "sva_overall", True),
             ("attraction", "attraction_overall", True),
             ("honorific", "honorific_overall", True),
             ("discourse", "discourse_overall", True)]
    pairs = [("transformer", "mamba3"), ("transformer", "hybrid"), ("mamba3", "hybrid")]
    for label, key, higher_better in conds:
        print(f"\n{label}:")
        for a, b in pairs:
            va, vb = RES[key][a], RES[key][b]
            diff, t, p = welch_t(va, vb)
            sig = "**" if p < 0.05 else ("*" if p < 0.10 else "ns")
            sc = 1 if key == "perplexity" else 100
            print(f"  {a:11s} vs {b:11s}  diff={diff*sc:+6.2f}  t={t:+.2f}  p={p:.3f}  {sig}")


def print_ci_table():
    n_seeds = len(RES["sva_overall"]["transformer"])
    print(f"== Wilson 95% CIs (pooled over {n_seeds} seeds) ==")
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
    print_between_arch()
    if args.dumps:
        run_mcnemar(args.dumps)


if __name__ == "__main__":
    main()
