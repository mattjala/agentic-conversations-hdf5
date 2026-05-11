"""HDF5 tuning sweep — chunk shape × compression × shuffle.

HDF5-only sweep over the storage knobs that matter for query latency and
file size. Used to pick a default config before the headline benchmarks
fix the HDF5 backend at one setting.

Configurable axes:
    * chunk_rows         (256 / 1024 / 4096) — chunk axis-0 of /embeddings
    * compression        (none / gzip-1 / gzip-4 / gzip-9)
    * shuffle            (on / off; only affects compressed cases)
    * N                  (corpus size)
    * dim                (embedding dim)

Reports for each cell:
    file size on disk
    upsert throughput (docs/sec)
    query latency p50/p95
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from _common import BACKENDS, build_backend, clean_path, now_ms, save_json, summarise
from workload import WorkloadConfig, generate, query_pool


DEFAULT_CHUNKS = [256, 1024, 4096]
DEFAULT_COMPRESSIONS = [
    {"name": "none",  "compression": None,   "compression_opts": None},
    {"name": "gzip1", "compression": "gzip", "compression_opts": 1},
    {"name": "gzip4", "compression": "gzip", "compression_opts": 4},
    {"name": "gzip9", "compression": "gzip", "compression_opts": 9},
]
DEFAULT_SHUFFLE = [True, False]
DEFAULT_N = 10_000
DEFAULT_DIM = 384
DEFAULT_QUERIES = 32


def file_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return path.stat().st_size


def run_cell(
    chunk_rows: int,
    compression: dict,
    shuffle: bool,
    n: int,
    dim: int,
    n_queries: int,
    store_root: Path,
) -> dict:
    label = f"c{chunk_rows}_{compression['name']}_{'sh' if shuffle else 'nosh'}"
    clean_path("hdf5", store_root, label)

    cfg = {
        "chunk_rows": chunk_rows,
        "compression": compression["compression"],
        "compression_opts": compression["compression_opts"],
        "shuffle": shuffle,
    }
    store = build_backend("hdf5", store_root, label, dim, cfg)

    docs = generate(WorkloadConfig(n_docs=n))
    t_ingest_start = time.perf_counter()
    for i in range(0, len(docs), 256):
        store.upsert(docs[i : i + 256])
    t_ingest = time.perf_counter() - t_ingest_start

    qs = query_pool(n=n_queries)
    store.query(qs[0], 20)  # warm
    samples_ms: list[float] = []
    for q in qs:
        t0 = now_ms()
        store.query(q, 20)
        samples_ms.append(now_ms() - t0)

    store.close()

    spec = BACKENDS["hdf5"]
    p = store_root / f"hdf5_{label}{spec['ext']}"

    return {
        "chunk_rows": chunk_rows,
        "compression": compression["name"],
        "shuffle": shuffle,
        "n": n,
        "dim": dim,
        "bytes": file_bytes(p),
        "ingest_seconds": t_ingest,
        "throughput_docs_per_sec": n / t_ingest if t_ingest else 0,
        "query_latency_ms": summarise(samples_ms),
    }


def parse_args():
    p = argparse.ArgumentParser(description="HDF5 storage-knob sweep")
    p.add_argument("--chunk-rows", nargs="+", type=int, default=DEFAULT_CHUNKS)
    p.add_argument("--compressions", nargs="+",
                   default=[c["name"] for c in DEFAULT_COMPRESSIONS],
                   choices=[c["name"] for c in DEFAULT_COMPRESSIONS])
    p.add_argument("--shuffle", nargs="+", type=int, default=[1, 0],
                   help="1 for on, 0 for off")
    p.add_argument("--n", type=int, default=DEFAULT_N)
    p.add_argument("--dim", type=int, default=DEFAULT_DIM)
    p.add_argument("--queries", type=int, default=DEFAULT_QUERIES)
    p.add_argument("--quick", action="store_true",
                   help="tiny: chunks=[1024], compressions=[none,gzip4], shuffle=[1], n=1000")
    p.add_argument("--outdir", type=Path, default=Path("results"))
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.chunk_rows = [1024]
        args.compressions = ["none", "gzip4"]
        args.shuffle = [1]
        args.n = 1_000

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    store_root = outdir / "stores"
    store_root.mkdir(exist_ok=True)

    comp_lookup = {c["name"]: c for c in DEFAULT_COMPRESSIONS}
    selected_comps = [comp_lookup[name] for name in args.compressions]
    shuffles = [bool(s) for s in args.shuffle]

    print(f"matrix: chunks={args.chunk_rows} comps={args.compressions} "
          f"shuffles={shuffles} n={args.n} dim={args.dim}")

    results: list[dict] = []
    for chunk in args.chunk_rows:
        for comp in selected_comps:
            for sh in shuffles:
                if comp["compression"] is None and sh:
                    # shuffle has no effect without compression — skip the duplicate.
                    continue
                print(f"  [c={chunk:>5} {comp['name']:<5} shuffle={sh}] ",
                      end="", flush=True)
                cell = run_cell(chunk, comp, sh, args.n, args.dim,
                                args.queries, store_root)
                results.append(cell)
                print(
                    f"{cell['bytes']/1e3:>9,.1f} kB  "
                    f"{cell['throughput_docs_per_sec']:>8,.0f} d/s  "
                    f"q p50={cell['query_latency_ms']['p50']:6.2f} ms  "
                    f"p95={cell['query_latency_ms']['p95']:6.2f} ms"
                )

    save_json(outdir / "hdf5_tuning.json", {"args": vars(args), "results": results})
    print(f"\nwrote: {outdir / 'hdf5_tuning.json'}")


if __name__ == "__main__":
    main()
