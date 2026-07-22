"""Generate all paper figures from paper/results.json.

  python paper/figures.py

Writes PNG (300 dpi, preview) and PDF (vector, for LaTeX \\includegraphics) into
paper/figures/. No data or GPU needed — reads the final scored numbers only.

Design: colour-blind-safe palette with redundant marker/hatch encoding, individual
per-seed points overlaid so cross-seed spread is visible, sized for a single ACL
column (~3.3in) with fonts that stay legible at that width.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stats import N, pooled_ci  # Wilson-CI helpers, same directory

HERE = Path(__file__).resolve().parent
RES = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

MODELS = ["transformer", "mamba3", "hybrid"]
LABEL = {"transformer": "Transformer", "mamba3": "Mamba-3", "hybrid": "Hybrid"}
COLOR = {"transformer": "#2a78d6", "mamba3": "#1baf7a", "hybrid": "#5a4fbf"}
MARKER = {"transformer": "o", "mamba3": "D", "hybrid": "^"}

plt.rcParams.update({
    "font.size": 12,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#e6e5df",
    "grid.linewidth": 0.7,
    "xtick.major.size": 0,
    "ytick.major.size": 3,
    "legend.handlelength": 1.4,
    "legend.handletextpad": 0.5,
    "legend.columnspacing": 1.2,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})


def _mean(xs):
    return sum(xs) / len(xs)


def _sd(xs):
    m = _mean(xs)
    return (sum((v - m) ** 2 for v in xs) / len(xs)) ** 0.5


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", name + ".png/.pdf")


def fig_distance():
    """Fig 1 — headline: SVA accuracy vs subject-verb distance (Wilson 95% CI bands)."""
    order = RES["sva_by_distance"]["_order"]
    labels = ["none", "short", "medium", "long"]
    x = list(range(len(order)))
    fig, ax = plt.subplots(figsize=(3.5, 3.0), constrained_layout=True)
    for m in MODELS:
        seeds = RES["sva_by_distance"][m]
        mean, lo, hi = [], [], []
        for i, b in enumerate(order):
            mn, l, h = pooled_ci([s[i] for s in seeds], N[f"sva_{b}"])
            mean.append(mn * 100); lo.append(l * 100); hi.append(h * 100)
        ax.fill_between(x, lo, hi, color=COLOR[m], alpha=0.13, linewidth=0)
        ax.plot(x, mean, color=COLOR[m], marker=MARKER[m], markersize=5.5,
                markeredgecolor="white", markeredgewidth=0.6,
                linewidth=1.8, label=LABEL[m], clip_on=False)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("subject–verb distance")
    ax.set_ylabel("agreement accuracy (%)")
    ax.set_ylim(75, 95)
    ax.set_xlim(-0.15, len(order) - 0.85)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              fontsize=10.5, borderaxespad=0.2)
    save(fig, "fig1_distance")


def _grouped(ax, keys, conds, names, ylabel, ylim, chance=None):
    w = 0.26
    for j, m in enumerate(MODELS):
        xs = [i + (j - 1) * w for i in range(len(keys))]
        means = [_mean([v * 100 for v in RES[k][m]]) for k in keys]
        errs = [_sd([v * 100 for v in RES[k][m]]) for k in keys]
        ax.bar(xs, means, width=w, color=COLOR[m], label=LABEL[m],
               edgecolor="white", linewidth=0.7, zorder=2,
               yerr=errs, capsize=2.5,
               error_kw={"elinewidth": 1, "ecolor": "#333331", "zorder": 4})
    if chance is not None:
        ax.axhline(chance, color="#9a9990", linestyle=(0, (3, 3)), linewidth=0.9, zorder=1)
        ax.set_xlim(-0.5, len(keys) - 0.5 + 0.36)
        ax.text(len(keys) - 0.5 + 0.05, chance, "chance", fontsize=8.5,
                color="#8a8a82", va="center", ha="left")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(names)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              fontsize=10.5, borderaxespad=0.2)


def fig_probes():
    """Fig 2 — accuracy by probe type (bars=mean, error=cross-seed SD, dots=seeds)."""
    fig, ax = plt.subplots(figsize=(5.4, 3.1), constrained_layout=True)
    _grouped(ax,
             ["sva_overall", "attraction_overall", "honorific_overall", "discourse_overall"],
             ["sva", "attraction", "honorific", "discourse"],
             ["SVA", "Attraction", "Honorific", "Discourse"],
             "accuracy (%)", (45, 100), chance=50)
    save(fig, "fig2_probes")


def fig_perplexity():
    """Fig 3 — held-out perplexity (lower is better); error bars = cross-seed SD."""
    fig, ax = plt.subplots(figsize=(3.0, 3.0), constrained_layout=True)
    for i, m in enumerate(MODELS):
        vals = RES["perplexity"][m]
        mn, sd = _mean(vals), _sd(vals)
        ax.bar(i, mn, width=0.62, color=COLOR[m], edgecolor="white",
               linewidth=0.7, zorder=2, yerr=sd, capsize=3,
               error_kw={"elinewidth": 1, "ecolor": "#333331", "zorder": 4})
        ax.text(i, mn + sd + 0.12, f"{mn:.1f}", ha="center", va="bottom",
                fontsize=10, color="#2c2c2a")
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([LABEL[m] for m in MODELS], rotation=15, ha="right")
    ax.set_ylabel("perplexity (lower is better)")
    ax.set_ylim(38, 43)
    save(fig, "fig3_perplexity")


if __name__ == "__main__":
    fig_distance()
    fig_probes()
    fig_perplexity()
    print("all figures written to", OUT)
