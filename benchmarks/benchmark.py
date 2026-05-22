"""Benchmark harness for HDF5 vs. SQLite vs. JSON+NumPy session backends.

Usage
-----
    python benchmark.py                     # default sizes, both scenarios
    python benchmark.py --quick             # 50 / 500 turns only
    python benchmark.py --sizes 20 100 500  # custom session lengths
    python benchmark.py --only hdf5 sqlite  # subset of backends
    python benchmark.py --outdir results/   # output directory

Metrics
-------
1. Write throughput    — turns/sec (add_turn + add_tool_call per turn)
2. File / dir size     — bytes on disk after writing N turns
3. Context latency     — get_recent_context(20) median latency (ms)
4. Artifact latency    — store_artifact + get_artifact round-trip (ms)

Scenarios
---------
- "text"  — turns with text only (no embeddings, no array tool results)
- "data"  — turns with embeddings (1536-d) + tool calls with 1000-element results

Graphs
------
- Line: file size vs. session length, by scenario
- Bar:  write throughput at max session length, by scenario
- Line: context latency vs. session length
- Bar:  artifact round-trip at max session length
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentic_conversations_hdf5 import (
    HDF5Session, JSONSession, SQLiteSession, ORCSession,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 1536
TOOL_RESULT_LEN = 1000
ARTIFACT_SHAPE = (500, 200)    # 100 k float32 elements
CONTEXT_N = 20                 # turns to reconstruct
REPEATS = 5                    # median over this many latency reads
RNG_SEED = 42

# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

_rng = np.random.default_rng(RNG_SEED)

# Pre-generate a pool of turn texts so string generation is not the bottleneck
_TURN_TEXTS = [
    f"Turn {i}: The assistant analysed the dataset and found {i * 3} anomalies "
    f"in the time series at index {i % 100}. Recommended action: re-run filter."
    for i in range(10_000)
]
_TURN_ROLES = ["user", "assistant"]

def turn_text(i: int) -> str:
    return _TURN_TEXTS[i % len(_TURN_TEXTS)]

def turn_role(i: int) -> str:
    return _TURN_ROLES[i % 2]

def rand_embedding() -> np.ndarray:
    v = _rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
    return v / np.linalg.norm(v)

def rand_tool_result() -> np.ndarray:
    return _rng.standard_normal(TOOL_RESULT_LEN).astype(np.float32)

def rand_artifact() -> np.ndarray:
    return _rng.standard_normal(ARTIFACT_SHAPE).astype(np.float32)

# ---------------------------------------------------------------------------
# Backend factory / cleanup
# ---------------------------------------------------------------------------

BACKENDS = {
    "hdf5": {
        "label": "HDF5 (flush=1)",
        "color": "#9467bd",
        "marker": "o",
    },
    "hdf5_lazy": {
        "label": "HDF5 (flush=100)",
        "color": "#c5b0d5",
        "marker": "X",
    },
    "hdf5_core": {
        "label": "HDF5 (Core VFD)",
        "color": "#7f3fbf",
        "marker": "*",
    },
    "hdf5_buf10": {
        "label": "HDF5 (batch=10)",
        "color": "#bcbddc",
        "marker": "P",
    },
    "hdf5_buf25": {
        "label": "HDF5 (batch=25)",
        "color": "#9e9ac8",
        "marker": "P",
    },
    "hdf5_buf100": {
        "label": "HDF5 (batch=100)",
        "color": "#756bb1",
        "marker": "P",
    },
    "sqlite": {
        "label": "SQLite",
        "color": "#ff7f0e",
        "marker": "^",
    },
    "json": {
        "label": "JSON + NumPy",
        "color": "#2ca02c",
        "marker": "D",
    },
    "orc_batch": {
        "label": "ORC (batch)",
        "color": "#8c564b",
        "marker": "v",
    },
    "orc_rewrite": {
        "label": "ORC (per-turn rewrite)",
        "color": "#e377c2",
        "marker": "<",
    },
}

# Backends that store a directory tree rather than a single file.
_DIR_BACKENDS = ("json", "orc_batch", "orc_rewrite")

# Backends that write a single .h5 file (excludes Core VFD, whose mid-session
# on-disk size is 0).
_H5_FILE_BACKENDS = ("hdf5", "hdf5_lazy", "hdf5_buf10", "hdf5_buf25", "hdf5_buf100")

SCENARIOS = ["text", "data"]
SCENARIO_LABELS = {"text": "text-only", "data": "array-heavy"}
SCENARIO_LINESTYLES = {"text": "-", "data": "--"}


def build_backend(backend: str, store_path: Path, session_id: str):
    if backend == "hdf5":
        return HDF5Session(store_path, session_id=session_id, mode="a", flush_every=1)
    elif backend == "hdf5_lazy":
        return HDF5Session(store_path, session_id=session_id, mode="a", flush_every=100)
    elif backend == "hdf5_core":
        return HDF5Session(store_path, session_id=session_id, mode="a",
                           flush_every=0, core_vfd=True)
    elif backend == "hdf5_buf10":
        return HDF5Session(store_path, session_id=session_id, mode="a",
                           flush_every=10, batch_size=10)
    elif backend == "hdf5_buf25":
        return HDF5Session(store_path, session_id=session_id, mode="a",
                           flush_every=25, batch_size=25)
    elif backend == "hdf5_buf100":
        return HDF5Session(store_path, session_id=session_id, mode="a",
                           flush_every=100, batch_size=100)
    elif backend == "sqlite":
        return SQLiteSession(store_path, session_id=session_id)
    elif backend == "json":
        return JSONSession(store_path, session_id=session_id)
    elif backend == "orc_batch":
        return ORCSession(store_path, session_id=session_id, mode="batch")
    elif backend == "orc_rewrite":
        return ORCSession(store_path, session_id=session_id, mode="rewrite")
    else:
        raise ValueError(f"Unknown backend: {backend}")


def clean_store(backend: str, store_path: Path) -> None:
    if backend in (*_H5_FILE_BACKENDS, "hdf5_core", "sqlite") \
            and store_path.exists():
        store_path.unlink()
    elif backend in _DIR_BACKENDS and store_path.exists():
        shutil.rmtree(store_path)


def storage_bytes(backend: str, store_path: Path) -> int:
    if backend in (*_H5_FILE_BACKENDS, "sqlite"):
        return store_path.stat().st_size if store_path.exists() else 0
    elif backend in _DIR_BACKENDS:
        if not store_path.exists():
            return 0
        total = 0
        for f in store_path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
        return total
    return 0


# ---------------------------------------------------------------------------
# Core benchmark loop
# ---------------------------------------------------------------------------

def run_scenario(
    backend: str,
    scenario: str,
    sizes: list[int],
    store_root: Path,
    verbose: bool = True,
) -> dict[int, dict]:
    """Run one (backend, scenario) combination across all session sizes."""
    results = {}

    for n in sorted(sizes):
        if verbose:
            print(f"    n={n:>6,} ... ", end="", flush=True)

        # Unique path per (backend, scenario, n) to avoid contamination
        if backend in (*_H5_FILE_BACKENDS, "hdf5_core"):
            sp = store_root / f"sess_{backend}_{scenario}_{n}.h5"
        elif backend == "sqlite":
            sp = store_root / f"sess_{backend}_{scenario}_{n}.db"
        else:  # json / orc (directory-based)
            sp = store_root / f"sess_{backend}_{scenario}_{n}"

        clean_store(backend, sp)

        with_embeddings = (scenario == "data")
        with_arrays = (scenario == "data")

        # ---- Write pass ----
        t_write_start = time.perf_counter()
        sess = build_backend(backend, sp, session_id="bench")
        for i in range(n):
            emb = rand_embedding() if with_embeddings else None
            sess.add_turn(turn_role(i), turn_text(i), embedding=emb)
            if with_arrays:
                sess.add_tool_call(
                    name="analyse",
                    args={"turn": i, "dataset": "timeseries"},
                    result_text=f"Found {i % 7} anomalies.",
                    result_data=rand_tool_result(),
                )
        sess.close()
        t_write = time.perf_counter() - t_write_start
        throughput = n / t_write

        # ---- File size ----
        size_bytes = storage_bytes(backend, sp)

        # ---- Context latency ----
        # Measure "cold open + get_recent_context" as one operation:
        # the real question is how long until a fresh consumer has the context.
        context_times = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            sess = build_backend(backend, sp, session_id="bench")
            _ = sess.get_recent_context(CONTEXT_N)
            context_times.append(time.perf_counter() - t0)
            sess.close()
        context_ms = np.median(context_times) * 1000

        # ---- Artifact latency ----
        art = rand_artifact()
        sess = build_backend(backend, sp, session_id="bench")
        art_write_times = []
        art_read_times = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            sess.store_artifact("sensor_data", art)
            art_write_times.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            _ = sess.get_artifact("sensor_data")
            art_read_times.append(time.perf_counter() - t0)
        sess.close()
        artifact_ms = (np.median(art_write_times) + np.median(art_read_times)) * 1000

        results[n] = {
            "n": n,
            "throughput_tps": throughput,
            "size_bytes": size_bytes,
            "context_ms": context_ms,
            "artifact_ms": artifact_ms,
        }

        if verbose:
            print(
                f"write={throughput:,.0f} t/s  "
                f"size={size_bytes/1e3:.0f} kB  "
                f"ctx={context_ms:.1f} ms  "
                f"art={artifact_ms:.1f} ms"
            )

    return results


def run_benchmark(
    sizes: list[int],
    backends: list[str],
    scenarios: list[str],
    store_root: Path,
    verbose: bool = True,
) -> dict:
    """Returns results[backend][scenario][n]."""
    results: dict = {b: {s: {} for s in scenarios} for b in backends}

    for b in backends:
        for s in scenarios:
            if verbose:
                label = BACKENDS[b]["label"]
                print(f"\n  [{label} / {SCENARIO_LABELS[s]}]")
            results[b][s] = run_scenario(b, s, sizes, store_root, verbose)

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results: dict, outdir: Path, sizes: list[int]) -> None:
    backends = list(results.keys())
    max_n = max(sizes)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(
        "Agent Session Backend Benchmarks\n"
        "HDF5 vs. SQLite vs. JSON+NumPy  |  solid=text-only  dashed=array-heavy",
        fontsize=12,
        y=1.01,
    )

    # 1. File size vs. session length — line chart, split by scenario
    ax = axes[0, 0]
    for b in backends:
        for s in SCENARIOS:
            data = results[b][s]
            xs = sorted(data.keys())
            ys = [data[x]["size_bytes"] / 1e3 for x in xs]
            ax.plot(
                xs, ys,
                label=f"{BACKENDS[b]['label']} ({SCENARIO_LABELS[s]})",
                color=BACKENDS[b]["color"],
                marker=BACKENDS[b]["marker"],
                linestyle=SCENARIO_LINESTYLES[s],
            )
    ax.set_title("Session file size vs. turn count")
    ax.set_xlabel("Turns (N)")
    ax.set_ylabel("Size (kB)")
    ax.legend(fontsize=7)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # 2. Write throughput — grouped bar chart at max_n, two scenario groups
    ax = axes[0, 1]
    x_pos = np.arange(len(backends))
    width = 0.35
    for si, s in enumerate(SCENARIOS):
        vals = [results[b][s][max_n]["throughput_tps"] for b in backends]
        offset = (si - 0.5) * width
        bars = ax.bar(
            x_pos + offset, vals, width,
            label=SCENARIO_LABELS[s],
            color=[BACKENDS[b]["color"] for b in backends],
            alpha=0.7 + 0.3 * si,
            edgecolor="white",
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                f"{val:,.0f}", ha="center", va="bottom", fontsize=7,
            )
    ax.set_title(f"Write throughput at N={max_n:,}")
    ax.set_ylabel("Turns / second")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([BACKENDS[b]["label"] for b in backends])
    ax.legend(fontsize=8)

    # 3. Context reconstruction latency vs. session length — line chart
    ax = axes[1, 0]
    for b in backends:
        for s in SCENARIOS:
            data = results[b][s]
            xs = sorted(data.keys())
            ys = [data[x]["context_ms"] for x in xs]
            ax.plot(
                xs, ys,
                label=f"{BACKENDS[b]['label']} ({SCENARIO_LABELS[s]})",
                color=BACKENDS[b]["color"],
                marker=BACKENDS[b]["marker"],
                linestyle=SCENARIO_LINESTYLES[s],
            )
    ax.set_title(f"get_recent_context(n={CONTEXT_N}) latency vs. turn count")
    ax.set_xlabel("Turns (N)")
    ax.set_ylabel("Latency (ms, median)")
    ax.legend(fontsize=7)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # 4. Artifact round-trip — grouped bar at max_n
    ax = axes[1, 1]
    for si, s in enumerate(SCENARIOS):
        vals = [results[b][s][max_n]["artifact_ms"] for b in backends]
        offset = (si - 0.5) * width
        bars = ax.bar(
            x_pos + offset, vals, width,
            label=SCENARIO_LABELS[s],
            color=[BACKENDS[b]["color"] for b in backends],
            alpha=0.7 + 0.3 * si,
            edgecolor="white",
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                f"{val:.1f}", ha="center", va="bottom", fontsize=7,
            )
    ax.set_title(f"Artifact store+get round-trip at N={max_n:,}")
    ax.set_ylabel("Latency (ms, median)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([BACKENDS[b]["label"] for b in backends])
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = outdir / "benchmark_results.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved: {out_path}")


def save_raw(results: dict, outdir: Path) -> None:
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    serialisable: dict = {}
    for b, scenarios in results.items():
        serialisable[b] = {}
        for s, ns in scenarios.items():
            serialisable[b][s] = {}
            for n, metrics in ns.items():
                serialisable[b][s][str(n)] = {
                    k: convert(v) for k, v in metrics.items()
                }

    out_path = outdir / "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"Raw data:   {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Agent session backend benchmark")
    p.add_argument(
        "--sizes", nargs="+", type=int,
        default=[50, 200, 1000, 5000],
        help="Session lengths (turns) to test",
    )
    p.add_argument(
        "--quick", action="store_true",
        help="Quick run: sizes 50 / 500 only",
    )
    p.add_argument(
        "--only", nargs="+",
        choices=list(BACKENDS.keys()),
        metavar="BACKEND",
        help=f"Run only these backends: {list(BACKENDS.keys())}",
    )
    p.add_argument(
        "--outdir", type=Path, default=Path("results"),
        help="Output directory for figures and JSON (default: results/)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    store_root = outdir / "stores"
    store_root.mkdir(exist_ok=True)

    sizes = [50, 500] if args.quick else args.sizes
    backends = args.only or list(BACKENDS.keys())

    print(f"Backends  : {', '.join(BACKENDS[b]['label'] for b in backends)}")
    print(f"Scenarios : {', '.join(SCENARIO_LABELS[s] for s in SCENARIOS)}")
    print(f"Sizes     : {', '.join(f'{n:,}' for n in sorted(sizes))}")
    print(f"Output    : {outdir}")

    results = run_benchmark(
        sizes=sorted(sizes),
        backends=backends,
        scenarios=SCENARIOS,
        store_root=store_root,
    )

    plot_results(results, outdir, sizes)
    save_raw(results, outdir)

    # Summary table
    max_n = max(sizes)
    print("\n" + "=" * 90)
    print(
        f"{'Backend':<18} {'Scenario':<13} {'N':>6}  "
        f"{'kB':>8}  {'t/s':>10}  {'ctx ms':>8}  {'art ms':>8}"
    )
    print("-" * 90)
    for b in backends:
        for s in SCENARIOS:
            for n in sorted(sizes):
                d = results[b][s][n]
                print(
                    f"{BACKENDS[b]['label']:<18} {SCENARIO_LABELS[s]:<13} {n:>6,}  "
                    f"{d['size_bytes']/1e3:>8.0f}  "
                    f"{d['throughput_tps']:>10,.0f}  "
                    f"{d['context_ms']:>8.2f}  "
                    f"{d['artifact_ms']:>8.2f}"
                )
    print("=" * 90)


if __name__ == "__main__":
    main()
