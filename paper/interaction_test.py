"""Architecture x distance interaction test (camera-ready addition, Reviewer 1).

Reviewer 1 asked for a direct test of whether the Transformer's and Mamba-3's
SVA-accuracy-vs-distance slopes differ from each other, rather than inferring
a dissociation from "significant for one, not for the other." This fits a
per-seed OLS slope (distance encoded ordinally: none=0, short=1, medium=2,
long=3) and Welch-t's the five per-seed slopes between architectures, reusing
the same welch_t() the paper already uses for between-architecture claims.
No retraining or new eval runs: reads paper/results.json only.

  python paper/interaction_test.py
"""
from __future__ import annotations

import json
from pathlib import Path

from stats import _betai, welch_t

HERE = Path(__file__).resolve().parent
RES = json.loads((HERE / "results.json").read_text(encoding="utf-8"))

X = [0, 1, 2, 3]  # none, short, medium, long


def ols_slope(x: list[float], y: list[float]) -> float:
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den = sum((xi - mx) ** 2 for xi in x)
    return num / den


def main():
    order = RES["sva_by_distance"]["_order"]
    assert order == ["none", "short", "medium", "long"]

    slopes = {}
    for arch in ["transformer", "mamba3", "hybrid"]:
        seeds = RES["sva_by_distance"][arch]  # 5 x [none,short,medium,long]
        slopes[arch] = [ols_slope(X, seed_accs) for seed_accs in seeds]

    print("== Per-seed SVA-accuracy-vs-distance slopes (pp per bin step) ==")
    for arch in ["transformer", "mamba3", "hybrid"]:
        vals = [round(s * 100, 2) for s in slopes[arch]]
        mean = sum(slopes[arch]) / len(slopes[arch]) * 100
        print(f"  {arch:12s} seeds={vals}  mean={mean:+.2f} pp/step")

    print("\n== Welch t-test on per-seed slopes: architecture x distance interaction ==")
    pairs = [("transformer", "mamba3"), ("transformer", "hybrid"), ("mamba3", "hybrid")]
    for a, b in pairs:
        diff, t, p = welch_t(slopes[a], slopes[b])
        sig = "**" if p < 0.05 else ("*" if p < 0.10 else "ns")
        print(f"  {a:11s} vs {b:11s}  slope_diff={diff*100:+.2f} pp/step  t={t:+.2f}  p={p:.4f}  {sig}")

    print("\n== Seed-aware within-model decline: paired t, none vs. long, across 5 seeds ==")
    for arch in ["transformer", "mamba3", "hybrid"]:
        seeds = RES["sva_by_distance"][arch]
        diffs = [s[0] - s[3] for s in seeds]
        n = len(diffs)
        md = sum(diffs) / n
        sd = (sum((d - md) ** 2 for d in diffs) / (n - 1)) ** 0.5
        t = md / (sd / n ** 0.5)
        df = n - 1
        p = _betai(df / 2.0, 0.5, df / (df + t * t))
        print(f"  {arch:12s} none-long per seed={[round(d*100, 1) for d in diffs]} pp  "
              f"mean={md*100:.1f} pp  t={t:.2f}  p={p:.4f}")


if __name__ == "__main__":
    main()
