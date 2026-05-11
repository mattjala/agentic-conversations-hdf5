"""Child process for bench_memory.py.

Runs an ingest + query workload against a single (backend, n, dim) cell and
prints peak RSS on exit. Spawned in isolation so the parent's RSS doesn't
contaminate the measurement.
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
from pathlib import Path

from _common import build_backend, clean_path
from workload import WorkloadConfig, generate, query_pool


def maxrss_bytes() -> int:
    # On Linux, ru_maxrss is in kilobytes; on macOS it's in bytes.
    val = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return val
    return val * 1024


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--dim", type=int, required=True)
    p.add_argument("--queries", type=int, default=32)
    p.add_argument("--store-root", type=Path, required=True)
    args = p.parse_args()

    label = f"mem_n{args.n}_d{args.dim}"
    clean_path(args.backend, args.store_root, label)
    store = build_backend(args.backend, args.store_root, label, args.dim)

    docs = generate(WorkloadConfig(n_docs=args.n))
    for i in range(0, len(docs), 256):
        store.upsert(docs[i : i + 256])

    qs = query_pool(n=args.queries)
    for q in qs:
        store.query(q, limit=20)

    store.close()

    print(json.dumps({
        "backend": args.backend,
        "n": args.n,
        "dim": args.dim,
        "peak_rss_bytes": maxrss_bytes(),
    }))


if __name__ == "__main__":
    main()
