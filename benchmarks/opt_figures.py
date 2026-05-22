"""Generate slide figures for the packed-schema optimization deck.

Numbers are the measured results recorded in docs/schema-optimization-results.md and docs/orc-comparison.md (N=5000 unless noted). Encoded here
so the deck is reproducible without re-running every backend (the C backend needs
a custom HDF5 on LD_LIBRARY_PATH).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures" / "slides"
OUT.mkdir(parents=True, exist_ok=True)

PACKED = "#9467bd"
PACKED_LT = "#c5b0d5"
ORANGE = "#ff7f0e"
GREEN = "#2ca02c"
RED = "#d62728"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.35,
    "figure.dpi": 150,
})


def save(fig, name):
    p = OUT / f"{name}.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p.relative_to(ROOT)}")


def fig_batch_scaling():
    batches = [1, 10, 25, 100]
    text = [414, 3801, 8764, 25870]
    array = [174, 397, 431, 424]
    x = np.arange(len(batches))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    b1 = ax.bar(x - w/2, text, w, label="text-only", color=PACKED)
    b2 = ax.bar(x + w/2, array, w, label="array-heavy", color=PACKED_LT)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([f"batch={b}" for b in batches])
    ax.set_ylabel("write throughput (turns/sec, log)")
    ax.set_title("Batched-write buffer: throughput vs. batch size (N=5000)")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()*1.05,
                    f"{int(b.get_height()):,}", ha="center", va="bottom", fontsize=8)
    ax.legend()
    ax.set_ylim(100, 60000)
    save(fig, "opt_batch_scaling")


def fig_cumulative():
    stages = ["baseline", "+libver", "+pre-alloc", "+zlib-ng", "+batch=100"]
    tps = [296, 319, 409, 489, 25870]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    colors = ["#8a8a8a", "#6f8fb0", "#4f7fb0", "#2f6fb0", PACKED]
    bars = ax.bar(stages, tps, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("write throughput (turns/sec, log)")
    ax.set_title("Cumulative effect of optimizations (text-only, N=5000)")
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()*1.05,
                f"{int(b.get_height()):,}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(100, 60000)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    save(fig, "opt_cumulative")


def fig_read_calls():
    """Embedding consolidation: h5py call counts before/after."""
    cats = ["array\nlast-20", "array\nfull-scan", "text\nlast-20"]
    before = [45, 10005, 25]
    after = [8, 8, 6]
    x = np.arange(len(cats)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    b1 = ax.bar(x - w/2, before, w, label="per-UUID embeddings", color=RED)
    b2 = ax.bar(x + w/2, after, w, label="consolidated (N, dim)", color=GREEN)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylabel("HDF5 read calls (log)")
    ax.set_title("Embedding consolidation: read call counts (N=5000)")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()*1.08,
                    f"{int(b.get_height()):,}", ha="center", va="bottom", fontsize=8)
    ax.legend(); ax.set_ylim(1, 20000)
    save(fig, "opt_read_calls")


def fig_orc():
    """ORC per-turn rewrite cliff vs HDF5 packed batched (array-heavy)."""
    n = [50, 200, 500]
    orc = [132, 98, 55]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(n, orc, "o-", color=ORANGE, label="ORC (per-turn rewrite)")
    ax.axhline(422, ls="--", color=PACKED, label="HDF5-packed (batch=100)")
    ax.set_xlabel("turns written (N)")
    ax.set_ylabel("write throughput (turns/sec)")
    ax.set_title("ORC live-append cliff (array-heavy)")
    for xi, yi in zip(n, orc):
        ax.text(xi, yi-7, f"{yi}", ha="center", va="top", fontsize=9)
    ax.legend(); ax.set_ylim(0, 460)
    save(fig, "opt_orc_cliff")


if __name__ == "__main__":
    print("Generating optimization figures...")
    fig_batch_scaling()
    fig_cumulative()
    fig_read_calls()
    fig_orc()
    print("Done.")
