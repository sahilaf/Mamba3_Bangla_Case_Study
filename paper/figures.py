"""Generate all paper figures from paper/results.json.

  python paper/figures.py

Writes PNG (300 dpi, for GitHub/preview) and PDF (vector, for LaTeX) into
paper/figures/. No data or GPU needed — reads the final scored numbers only.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

# consistent, colour-blind-safe palette + distinct markers/hatches per model
MODELS = ["transformer", "mamba3", "hybrid"]
LABEL = {"transformer": "Transformer", "mamba3": "Mamba-3", "hybrid": "Hybrid"}
COLOR = {"transformer": "#2a78d6", "mamba3": "#1baf7a", "hybrid": "#4a3aa7"}
MARKER = {"transformer": "o", "mamba3": "D", "hybrid": "^"}
HATCH = {"transformer": "", "mamba3": "//", "hybrid": ".."}

plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e1e0d9",
    "grid.linewidth": 0.8,
    "figure.dpi": 300,
})


def _mean(xs):
    return sum(xs) / len(xs)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name + ".png/.pdf")


def fig_distance():
    """Fig 1 — the headline: SVA accuracy vs subject-verb distance."""
    order = RES["sva_by_distance"]["_order"]
    x = range(len(order))
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for m in MODELS:
        d = RES["sva_by_distance"][m]
        s1, s2 = d["s1"], d["s2"]
        mean = [(_mean([s1[i], s2[i]])) * 100 for i in range(len(order))]
        lo = [min(s1[i], s2[i]) * 100 for i in range(len(order))]
        hi = [max(s1[i], s2[i]) * 100 for i in range(len(order))]
        ax.fill_between(x, lo, hi, color=COLOR[m], alpha=0.15, linewidth=0)
        ax.plot(x, mean, color=COLOR[m], marker=MARKER[m], markersize=6,
                linewidth=2, label=LABEL[m])
    ax.set_xticks(list(x))
    ax.set_xticklabels(order)
    ax.set_xlabel("subject–verb distance")
    ax.set_ylabel("agreement accuracy (%)")
    ax.set_ylim(75, 95)
    ax.legend(frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.13), columnspacing=1.2, handletextpad=0.5)
    save(fig, "fig1_distance")


def fig_probes():
    """Fig 2 — accuracy by probe type, grouped bars with seed range as error."""
    probes = ["sva_overall", "attraction_overall", "honorific_overall", "discourse_overall"]
    names = ["SVA", "Attraction", "Honorific", "Discourse"]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    w = 0.26
    for j, m in enumerate(MODELS):
        means, errs = [], []
        for p in probes:
            vals = RES[p][m]
            mn = _mean(vals) * 100
            means.append(mn)
            errs.append((max(vals) - min(vals)) / 2 * 100)
        xs = [i + (j - 1) * w for i in range(len(probes))]
        ax.bar(xs, means, width=w, color=COLOR[m], label=LABEL[m],
               hatch=HATCH[m], edgecolor="white", linewidth=0.6,
               yerr=errs, capsize=3, error_kw={"elinewidth": 1, "ecolor": "#444441"})
    ax.axhline(50, color="#888780", linestyle=":", linewidth=1)
    ax.text(3.35, 51, "chance", fontsize=8, color="#888780", va="bottom", ha="right")
    ax.set_xticks(range(len(probes)))
    ax.set_xticklabels(names)
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    save(fig, "fig2_probes")


def fig_perplexity():
    """Fig 3 — held-out perplexity (lower is better)."""
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    for i, m in enumerate(MODELS):
        vals = RES["perplexity"][m]
        mn = _mean(vals)
        err = (max(vals) - min(vals)) / 2
        ax.bar(i, mn, width=0.6, color=COLOR[m], hatch=HATCH[m],
               edgecolor="white", linewidth=0.6, yerr=err, capsize=4,
               error_kw={"elinewidth": 1, "ecolor": "#444441"})
        ax.text(i, mn + 0.15, f"{mn:.1f}", ha="center", fontsize=10)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([LABEL[m] for m in MODELS], rotation=12)
    ax.set_ylabel("perplexity (↓)")
    ax.set_ylim(38, 43)
    save(fig, "fig3_perplexity")


if __name__ == "__main__":
    fig_distance()
    fig_probes()
    fig_perplexity()
    print("all figures written to", OUT)
