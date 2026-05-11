"""Query-latency benchmark.

Measures p50/p95/p99 latency of VectorStore.query() across:

    * backend       (inmem / sqlite_blob / hdf5)
    * N             (corpus size: 1k / 10k / 100k by default)
    * dim           (384 / 768 / 1536)
    * filter        (none / project / project+doc_type — varies selectivity)
    * top_k         (5 / 20 / 100)

Each cell runs `--queries` queries from a deterministic pool and reports
p50/p95/p99 in ms. Results saved as JSON; CLI flags let you cut down the
matrix for fast iteration.

Examples
--------
    python bench_query_latency.py --quick
    python bench_query_latency.py --backends inmem hdf5 --sizes 10000
    python bench_query_latency.py --dims 384 --filters none project
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from _common import BACKENDS, build_backend, clean_path, now_ms, save_json, summarise
from workload import WorkloadConfig, generate, query_pool


DEFAULT_SIZES = [1_000, 10_000, 100_000]
DEFAULT_DIMS = [384]                  # add 768/1536 via --dims for sweeps
DEFAULT_FILTERS = ["none", "project", "project_and_type"]
DEFAULT_TOP_K = [20]
DEFAULT_BACKENDS = ["inmem", "sqlite_blob", "hdf5"]


def build_filter(name: str) -> dict | None:
    # Filters target a known project bucket so selectivity is roughly:
    #   none                 -> 100%
    #   project              -> 1/n_projects (~25% with 4 projects)
    #   project_and_type     -> ~25% × ~70% (observation share) = ~17%
    if name == "none":
        return None
    if name == "project":
        return {"project": "project_0"}
    if name == "project_and_type":
        return {"$and": [{"project": "project_0"}, {"doc_type": "observation"}]}
    raise ValueError(f"unknown filter: {name}")


def run_cell(
    backend: str,
    n: int,
    dim: int,
    filter_name: str,
    top_k: int,
    n_queries: int,
    store_root: Path,
) -> dict:
    label = f"n{n}_d{dim}"
    clean_path(backend, store_root, label)
    store = build_backend(backend, store_root, label, dim)

    # Ingest.
    t_ingest = time.perf_counter()
    docs = generate(WorkloadConfig(n_docs=n))
    # Batch the upsert at 256 to avoid pathological single-doc paths.
    batch = 256
    for i in range(0, len(docs), batch):
        store.upsert(docs[i : i + batch])
    t_ingest = time.perf_counter() - t_ingest

    # Warm: one query to load any first-call paths (lazy embedder, etc.).
    where = build_filter(filter_name)
    queries = query_pool(n=n_queries)
    store.query(queries[0], top_k, where)

    # Measure.
    samples_ms: list[float] = []
    n_results: list[int] = []
    for q in queries:
        t0 = now_ms()
        result = store.query(q, top_k, where)
        samples_ms.append(now_ms() - t0)
        n_results.append(len(result.ids))

    store.close()

    return {
        "backend": backend,
        "n": n,
        "dim": dim,
        "filter": filter_name,
        "top_k": top_k,
        "n_queries": n_queries,
        "ingest_seconds": t_ingest,
        "latency_ms": summarise(samples_ms),
        "result_count": summarise([float(c) for c in n_results]),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Vector-store query latency benchmark")
    p.add_argument("--backends", nargs="+", default=DEFAULT_BACKENDS, choices=list(BACKENDS))
    p.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES)
    p.add_argument("--dims", nargs="+", type=int, default=DEFAULT_DIMS)
    p.add_argument("--filters", nargs="+", default=DEFAULT_FILTERS,
                   choices=["none", "project", "project_and_type"])
    p.add_argument("--top-k", nargs="+", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--queries", type=int, default=64, help="queries per cell")
    p.add_argument("--quick", action="store_true",
                   help="tiny matrix: sizes=[1000], dims=[384], filters=[none,project], top_k=[20]")
    p.add_argument("--outdir", type=Path, default=Path("results"))
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.sizes = [1_000]
        args.dims = [384]
        args.filters = ["none", "project"]
        args.top_k = [20]
        args.queries = 16

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    store_root = outdir / "stores"
    store_root.mkdir(exist_ok=True)

    print(f"matrix: backends={args.backends} sizes={args.sizes} dims={args.dims} "
          f"filters={args.filters} top_k={args.top_k} queries={args.queries}")

    results: list[dict] = []
    for backend in args.backends:
        for n in args.sizes:
            for dim in args.dims:
                for f in args.filters:
                    for k in args.top_k:
                        print(f"  [{backend} n={n} d={dim} filter={f} k={k}] ", end="", flush=True)
                        cell = run_cell(backend, n, dim, f, k, args.queries, store_root)
                        results.append(cell)
                        lat = cell["latency_ms"]
                        print(f"p50={lat['p50']:6.2f}ms  p95={lat['p95']:6.2f}ms  "
                              f"p99={lat['p99']:6.2f}ms  ingest={cell['ingest_seconds']:.1f}s")

    save_json(outdir / "query_latency.json", {"args": vars(args), "results": results})
    print(f"\nwrote: {outdir / 'query_latency.json'}")


if __name__ == "__main__":
    main()
