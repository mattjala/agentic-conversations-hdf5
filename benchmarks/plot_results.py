"""Render PNG figures from results/jsonl_vs_hdf5.json into results/figures/."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
DATA = ROOT / "results" / "jsonl_vs_hdf5.json"
OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

with open(DATA) as f:
    rows = json.load(f)

labels = [r["session"][:8] for r in rows]
x = np.arange(len(rows))
width = 0.27

C_JSONL = "#ff7f0e"
C_GZ = "#9467bd"
C_HDF5 = "#1f77b4"


def _save(fig, name):
    p = OUT / name
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


# Figure 1: file size (raw / gz / hdf5) per session, kB
fig, ax = plt.subplots(figsize=(8, 4.2))
jl = [r["jsonl_bytes"] / 1e3 for r in rows]
gz = [r["gzip_bytes"] / 1e3 for r in rows]
h5 = [r["hdf5_bytes"] / 1e3 for r in rows]
ax.bar(x - width, jl, width, label="JSONL (raw)", color=C_JSONL)
ax.bar(x,         gz, width, label="JSONL (gzip)", color=C_GZ)
ax.bar(x + width, h5, width, label="HDF5",         color=C_HDF5)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=0)
ax.set_ylabel("Size (kB)")
ax.set_title("On-disk size, 6 real Claude Code session logs")
ax.legend()
ax.grid(axis="y", linestyle=":", alpha=0.5)
_save(fig, "fig_file_size.png")

# Figure 2: tail-20 latency, ms
fig, ax = plt.subplots(figsize=(8, 4.2))
tj = [r["tail_jsonl_ms"] for r in rows]
th = [r["tail_hdf5_ms"] for r in rows]
ax.bar(x - width / 2, tj, width, label="JSONL", color=C_JSONL)
ax.bar(x + width / 2, th, width, label="HDF5",  color=C_HDF5)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Latency (ms, median of 3 cold reads)")
ax.set_title("Context tail-20 latency")
ax.legend()
ax.grid(axis="y", linestyle=":", alpha=0.5)
for i, (a, b) in enumerate(zip(tj, th)):
    ax.text(i - width / 2, a, f"{a:.1f}", ha="center", va="bottom", fontsize=8)
    ax.text(i + width / 2, b, f"{b:.2f}", ha="center", va="bottom", fontsize=8)
_save(fig, "fig_tail_latency.png")

# Figure 3: total-token-usage query latency, ms
fig, ax = plt.subplots(figsize=(8, 4.2))
uj = [r["usage_jsonl_ms"] for r in rows]
uh = [r["usage_hdf5_ms"] for r in rows]
ax.bar(x - width / 2, uj, width, label="JSONL", color=C_JSONL)
ax.bar(x + width / 2, uh, width, label="HDF5",  color=C_HDF5)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Latency (ms, median of 3 cold reads)")
ax.set_title('"Sum total token usage for this session" — analytical query')
ax.legend()
ax.grid(axis="y", linestyle=":", alpha=0.5)
for i, (a, b) in enumerate(zip(uj, uh)):
    ax.text(i - width / 2, a, f"{a:.1f}", ha="center", va="bottom", fontsize=8)
    ax.text(i + width / 2, b, f"{b:.2f}", ha="center", va="bottom", fontsize=8)
_save(fig, "fig_usage_latency.png")
