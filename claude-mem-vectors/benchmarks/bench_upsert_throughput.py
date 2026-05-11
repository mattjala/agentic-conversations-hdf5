"""Upsert-throughput benchmark.

Measures docs/sec for VectorStore.upsert() across batch sizes. claude-mem
writes via lifecycle hooks — observations and summaries arrive in small
bursts (single doc to ~10 docs at a time during a session, larger batches
during backfill). So the matrix sweeps both regimes.

Configurable axes:
    * backend         (inmem / sqlite_blob / hdf5)
    * batch_size      (1 / 10 / 100 / 1000)
    * total_docs      (corpus size to ingest)
    * dim             (embedding dimensionality)

Reports docs/sec and per-batch latency p50/p95/p99 (the latter matters for
hook-blocking time — a Stop hook that takes 500ms is user-visible).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from _common import BACKENDS, build_backend, clean_path, now_ms, save_json, summarise
from workload import WorkloadConfig, generate


DEFAULT_BATCHES = [1, 10, 100, 1000]
DEFAULT_TOTALS = [10_000]
DEFAULT_DIMS = [384]
DEFAULT_BACKENDS = ["inmem", "sqlite_blob", "hdf5"]


def run_cell(
    backend: str,
    batch_size: int,
    total: int,
    dim: int,
    store_root: Path,
) -> dict:
    label = f"bs{batch_size}_n{total}_d{dim}"
    clean_path(backend, store_root, label)
    store = build_backend(backend, store_root, label, dim)

    docs = generate(WorkloadConfig(n_docs=total))

    # Warm: embed once outside the timing loop so first-call lazy paths
    # (e.g. embedder model load) don't poison the first batch.
    if docs:
        store.upsert(docs[: min(batch_size, len(docs))])

    # Reset for clean measurement: drop & re-create.
    store.close()
    clean_path(backend, store_root, label)
    store = build_backend(backend, store_root, label, dim)

    per_batch_ms: list[float] = []
    t_total_start = time.perf_counter()
    for i in range(0, len(docs), batch_size):
        chunk = docs[i : i + batch_size]
        t0 = now_ms()
        store.upsert(chunk)
        per_batch_ms.append(now_ms() - t0)
    t_total = time.perf_counter() - t_total_start
    store.close()

    return {
        "backend": backend,
        "batch_size": batch_size,
        "total": total,
        "dim": dim,
        "elapsed_seconds": t_total,
        "throughput_docs_per_sec": total / t_total if t_total > 0 else 0,
        "per_batch_ms": summarise(per_batch_ms),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Vector-store upsert throughput benchmark")
    p.add_argument("--backends", nargs="+", default=DEFAULT_BACKENDS, choices=list(BACKENDS))
    p.add_argument("--batches", nargs="+", type=int, default=DEFAULT_BATCHES)
    p.add_argument("--totals", nargs="+", type=int, default=DEFAULT_TOTALS)
    p.add_argument("--dims", nargs="+", type=int, default=DEFAULT_DIMS)
    p.add_argument("--quick", action="store_true",
                   help="tiny matrix: batches=[1,100], totals=[1000], dims=[384]")
    p.add_argument("--outdir", type=Path, default=Path("results"))
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.batches = [1, 100]
        args.totals = [1_000]
        args.dims = [384]

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    store_root = outdir / "stores"
    store_root.mkdir(exist_ok=True)

    print(f"matrix: backends={args.backends} batches={args.batches} "
          f"totals={args.totals} dims={args.dims}")

    results: list[dict] = []
    for backend in args.backends:
        for total in args.totals:
            for dim in args.dims:
                for bs in args.batches:
                    print(f"  [{backend} bs={bs:>4} total={total} dim={dim}] ", end="", flush=True)
                    cell = run_cell(backend, bs, total, dim, store_root)
                    results.append(cell)
                    print(
                        f"{cell['throughput_docs_per_sec']:>9,.0f} d/s  "
                        f"batch p50={cell['per_batch_ms']['p50']:6.2f}ms  "
                        f"p99={cell['per_batch_ms']['p99']:6.2f}ms"
                    )

    save_json(outdir / "upsert_throughput.json", {"args": vars(args), "results": results})
    print(f"\nwrote: {outdir / 'upsert_throughput.json'}")


if __name__ == "__main__":
    main()
