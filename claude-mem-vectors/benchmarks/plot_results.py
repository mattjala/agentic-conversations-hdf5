"""Render one graph per benchmark metric from results/*.json.

Outputs go to results/figures/. Each figure shows every backend / config
that was tested for that metric. Run after the bench_*.py scripts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BACKEND_STYLE = {
    "inmem":       {"label": "in-memory (numpy)", "color": "#2ca02c", "marker": "D"},
    "sqlite_blob": {"label": "SQLite + BLOB",     "color": "#ff7f0e", "marker": "^"},
    "hdf5":        {"label": "HDF5 (h5py)",       "color": "#1f77b4", "marker": "o"},
}

FILTER_HATCH = {"none": "", "project": "//", "project_and_type": "xx"}
FILTER_LABEL = {"none": "no filter", "project": "+ project=p0",
                "project_and_type": "+ project=p0 & doc_type=obs"}


def load(results_dir: Path, name: str) -> dict:
    p = results_dir / f"{name}.json"
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------

def plot_query_latency(data: dict, out: Path) -> None:
    rows = data["results"]
    backends = sorted({r["backend"] for r in rows}, key=lambda b: list(BACKEND_STYLE).index(b))
    sizes    = sorted({r["n"] for r in rows})
    filters  = ["none", "project", "project_and_type"]
    filters  = [f for f in filters if any(r["filter"] == f for r in rows)]

    fig, axes = plt.subplots(1, len(filters), figsize=(5 * len(filters), 4.5),
                             sharey=True, squeeze=False)

    for col, f in enumerate(filters):
        ax = axes[0, col]
        for backend in backends:
            xs, p50, p95, p99 = [], [], [], []
            for n in sizes:
                cell = next((r for r in rows if r["backend"] == backend
                             and r["n"] == n and r["filter"] == f), None)
                if cell is None:
                    continue
                xs.append(n)
                p50.append(cell["latency_ms"]["p50"])
                p95.append(cell["latency_ms"]["p95"])
                p99.append(cell["latency_ms"]["p99"])
            if not xs:
                continue
            style = BACKEND_STYLE[backend]
            ax.plot(xs, p50, marker=style["marker"], color=style["color"],
                    label=style["label"] + " (p50)", linewidth=2)
            ax.fill_between(xs, p50, p95, color=style["color"], alpha=0.15,
                            label=style["label"] + " p50→p95")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"filter: {FILTER_LABEL[f]}")
        ax.set_xlabel("corpus size N")
        if col == 0:
            ax.set_ylabel("query latency (ms)  log scale")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")

    fig.suptitle("Query latency vs. corpus size (top-k=20, dim=384)\n"
                 "shaded band = p50→p95",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_upsert_throughput(data: dict, out: Path) -> None:
    rows = data["results"]
    backends = sorted({r["backend"] for r in rows}, key=lambda b: list(BACKEND_STYLE).index(b))
    batches  = sorted({r["batch_size"] for r in rows})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: throughput.
    x = np.arange(len(batches))
    width = 0.8 / len(backends)
    for i, backend in enumerate(backends):
        style = BACKEND_STYLE[backend]
        vals = []
        for bs in batches:
            cell = next((r for r in rows if r["backend"] == backend
                         and r["batch_size"] == bs), None)
            vals.append(cell["throughput_docs_per_sec"] if cell else 0)
        bars = ax1.bar(x + i * width - 0.4 + width / 2, vals, width,
                       label=style["label"], color=style["color"],
                       edgecolor="white")
        for b, v in zip(bars, vals):
            ax1.text(b.get_x() + b.get_width() / 2, max(v, 1) * 1.05,
                     f"{v:,.0f}", ha="center", va="bottom", fontsize=7)
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(bs) for bs in batches])
    ax1.set_xlabel("batch size")
    ax1.set_ylabel("docs / second  log scale")
    ax1.set_title("Throughput at total=10,000")
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(True, which="both", axis="y", alpha=0.3)

    # Right: per-batch p99 latency (the hook-blocking cost).
    for i, backend in enumerate(backends):
        style = BACKEND_STYLE[backend]
        vals = []
        for bs in batches:
            cell = next((r for r in rows if r["backend"] == backend
                         and r["batch_size"] == bs), None)
            vals.append(cell["per_batch_ms"]["p99"] if cell else 0)
        ax2.plot(batches, vals, marker=style["marker"], color=style["color"],
                 label=style["label"], linewidth=2)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("batch size")
    ax2.set_ylabel("per-batch p99 latency (ms)  log scale")
    ax2.set_title("Per-batch latency p99")
    ax2.legend(fontsize=8)
    ax2.grid(True, which="both", alpha=0.3)

    fig.suptitle("Upsert throughput and per-batch latency (dim=384)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_file_size(data: dict, out: Path) -> None:
    rows = data["results"]
    sizes = sorted({r["n"] for r in rows})

    # Configs: SQLite (one), HDF5×{none,gzip1,gzip4,gzip9}.
    series: list[tuple[str, str, str]] = []  # (key_match, label, colour)
    series.append(("sqlite_blob:default", "SQLite + BLOB",            "#ff7f0e"))
    series.append(("hdf5:none",           "HDF5 (no compression)",    "#9ecae1"))
    series.append(("hdf5:gzip1",          "HDF5 (gzip-1 + shuffle)",  "#6baed6"))
    series.append(("hdf5:gzip4",          "HDF5 (gzip-4 + shuffle)",  "#3182bd"))
    series.append(("hdf5:gzip9",          "HDF5 (gzip-9 + shuffle)",  "#08519c"))

    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(sizes))
    width = 0.8 / len(series)

    for i, (key, label, color) in enumerate(series):
        backend, comp = key.split(":")
        vals = []
        for n in sizes:
            cell = next((r for r in rows if r["backend"] == backend
                         and r["n"] == n and r["compression"] == comp), None)
            vals.append((cell["bytes"] / 1024) if cell else 0)
        bars = ax.bar(x + i * width - 0.4 + width / 2, vals, width,
                      label=label, color=color, edgecolor="white")
        for b, v in zip(bars, vals):
            if v:
                ax.text(b.get_x() + b.get_width() / 2, v * 1.02,
                        f"{v:,.0f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([f"N={n:,}" for n in sizes])
    ax.set_ylabel("file size (kB)")
    ax.set_title("On-disk size by backend and compression (dim=384)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_memory(data: dict, out: Path) -> None:
    rows = data["results"]
    backends = sorted({r["backend"] for r in rows}, key=lambda b: list(BACKEND_STYLE).index(b))
    sizes    = sorted({r["n"] for r in rows})

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(sizes))
    width = 0.8 / len(backends)

    for i, backend in enumerate(backends):
        style = BACKEND_STYLE[backend]
        vals = []
        for n in sizes:
            cell = next((r for r in rows if r["backend"] == backend and r["n"] == n), None)
            vals.append(cell["peak_rss_bytes"] / 1e6 if cell else 0)
        bars = ax.bar(x + i * width - 0.4 + width / 2, vals, width,
                      label=style["label"], color=style["color"], edgecolor="white")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.01,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=7)

    # Reference line for Chroma RAM.
    ax.axhline(35_000, linestyle="--", color="#d62728", linewidth=1.5)
    ax.text(len(sizes) - 0.5, 35_000, "Chroma reported: ~35,000 MB",
            color="#d62728", ha="right", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"N={n:,}" for n in sizes])
    ax.set_ylabel("peak RSS (MB)  log scale")
    ax.set_yscale("log")
    ax.set_title("Peak resident memory of the entire process\n"
                 "(ingest + 32 queries with FakeHashEmbedder)")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cold_start(data: dict, out: Path) -> None:
    rows = data["results"]
    backends = [r["backend"] for r in rows]
    phases = ["open_ms", "first_query_ms", "second_query_ms"]
    phase_labels = {"open_ms": "open file",
                    "first_query_ms": "first query",
                    "second_query_ms": "second query (warm)"}

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(backends))
    width = 0.27

    for i, phase in enumerate(phases):
        vals = [r["phases"][phase]["p50"] for r in rows]
        errs_lo = [r["phases"][phase]["p50"] - r["phases"][phase]["min"] for r in rows]
        errs_hi = [r["phases"][phase]["max"] - r["phases"][phase]["p50"] for r in rows]
        bars = ax.bar(x + (i - 1) * width, vals, width,
                      yerr=[errs_lo, errs_hi],
                      capsize=3,
                      label=phase_labels[phase])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.02,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([BACKEND_STYLE[b]["label"] for b in backends])
    ax.set_ylabel("latency (ms, median)\nerror bars = min..max across trials")
    ax.set_title("Cold-start phases at N=10,000\n"
                 "(file already populated; opens an existing store)")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_hdf5_tuning(data: dict, out: Path) -> None:
    rows = data["results"]
    chunks = sorted({r["chunk_rows"] for r in rows})

    # Build the actual configurations that the harness produced — the tuning
    # script skips redundant cells (e.g. shuffle has no effect when there is
    # no compression). Plot every distinct (compression, shuffle) pair.
    seen = []
    for r in rows:
        key = (r["compression"], r["shuffle"])
        if key not in seen:
            seen.append(key)
    # Order: none first, then by compression name and shuffle on before off.
    def sort_key(k):
        comp, sh = k
        order = {"none": 0, "gzip1": 1, "gzip4": 2, "gzip9": 3}.get(comp, 99)
        return (order, 0 if sh else 1)
    configs = sorted(seen, key=sort_key)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    x = np.arange(len(chunks))
    width = 0.85 / max(1, len(configs))

    palette = ["#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c",
               "#cb181d", "#ef3b2c", "#fb6a4a"]

    for ci, (comp, shuffle) in enumerate(configs):
        sizes_kb, lat_p50 = [], []
        for ch in chunks:
            cell = next((r for r in rows if r["chunk_rows"] == ch
                         and r["compression"] == comp
                         and r["shuffle"] == shuffle), None)
            sizes_kb.append((cell["bytes"] / 1024) if cell else 0)
            lat_p50.append(cell["query_latency_ms"]["p50"] if cell else 0)

        color = palette[ci % len(palette)]
        hatch = "//" if shuffle else ""
        label = f"{comp}" + (" + shuffle" if shuffle else "")
        offset = ci * width - 0.425 + width / 2
        ax1.bar(x + offset, sizes_kb, width, color=color, hatch=hatch,
                edgecolor="white", label=label)
        ax2.bar(x + offset, lat_p50, width, color=color, hatch=hatch,
                edgecolor="white")

    for ax, ylabel, title in (
        (ax1, "file size (kB)", "On-disk size"),
        (ax2, "query latency p50 (ms)", "Query latency"),
    ):
        ax.set_xticks(x)
        ax.set_xticklabels([f"chunk={c}" for c in chunks])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)

    # Single legend, dedup.
    handles, labels = ax1.get_legend_handles_labels()
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        h2.append(h); l2.append(l)
    fig.legend(h2, l2, loc="upper center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("HDF5 tuning sweep at N=10,000 (dim=384, top-k=20)",
                 fontsize=11, y=1.07)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=Path(__file__).parent.parent / "results")
    args = p.parse_args()

    figdir = args.results_dir / "figures"
    figdir.mkdir(exist_ok=True)

    plots = [
        ("query_latency",     plot_query_latency,     "fig_query_latency.png"),
        ("upsert_throughput", plot_upsert_throughput, "fig_upsert_throughput.png"),
        ("file_size",         plot_file_size,         "fig_file_size.png"),
        ("memory",            plot_memory,            "fig_memory.png"),
        ("cold_start",        plot_cold_start,        "fig_cold_start.png"),
        ("hdf5_tuning",       plot_hdf5_tuning,       "fig_hdf5_tuning.png"),
    ]

    for name, fn, out in plots:
        try:
            data = load(args.results_dir, name)
        except FileNotFoundError:
            print(f"  skip {name}: no JSON yet")
            continue
        out_path = figdir / out
        fn(data, out_path)
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
