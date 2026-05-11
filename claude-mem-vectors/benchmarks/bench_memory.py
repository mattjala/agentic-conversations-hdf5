"""Peak-RSS benchmark.

The headline metric: this is the dimension on which Chroma loses (35GB+
on macOS per Issue #707). Measures peak resident memory of a subprocess
that performs a representative ingest + query workload.

Subprocess isolation matters because Python's GC keeps freed pages
resident; co-running multiple backends in one process would conflate
their footprints. Each (backend, n, dim) cell gets its own process.

Configurable axes:
    * backend         (inmem / sqlite_blob / hdf5)
    * N               (1k / 10k / 100k / 500k)
    * dim             (384 / 768 / 1536)

Output: peak RSS bytes per cell. A useful derived metric is bytes per doc;
for an on-disk store like HDF5 we expect this to be largely independent of N
(only the query embedding + working buffers should grow).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from _common import BACKENDS, save_json


DEFAULT_SIZES = [1_000, 10_000, 100_000]
DEFAULT_DIMS = [384]
DEFAULT_BACKENDS = ["inmem", "sqlite_blob", "hdf5"]


def run_cell(backend: str, n: int, dim: int, store_root: Path) -> dict:
    here = Path(__file__).parent
    proc = subprocess.run(
        [
            sys.executable,
            str(here / "_mem_child.py"),
            "--backend", backend,
            "--n", str(n),
            "--dim", str(dim),
            "--store-root", str(store_root.resolve()),
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=here,
    )
    # Child prints exactly one JSON line on stdout.
    line = proc.stdout.strip().splitlines()[-1]
    return json.loads(line)


def parse_args():
    p = argparse.ArgumentParser(description="Vector-store peak-RSS benchmark")
    p.add_argument("--backends", nargs="+", default=DEFAULT_BACKENDS, choices=list(BACKENDS))
    p.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES)
    p.add_argument("--dims", nargs="+", type=int, default=DEFAULT_DIMS)
    p.add_argument("--quick", action="store_true",
                   help="tiny matrix: sizes=[1000, 10000], dims=[384]")
    p.add_argument("--outdir", type=Path, default=Path("results"))
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.sizes = [1_000, 10_000]
        args.dims = [384]

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    store_root = outdir / "stores"
    store_root.mkdir(exist_ok=True)

    print(f"matrix: backends={args.backends} sizes={args.sizes} dims={args.dims}")

    results: list[dict] = []
    for backend in args.backends:
        for n in args.sizes:
            for dim in args.dims:
                print(f"  [{backend} n={n} d={dim}] ", end="", flush=True)
                cell = run_cell(backend, n, dim, store_root)
                results.append(cell)
                print(f"peak RSS = {cell['peak_rss_bytes']/1e6:8.1f} MB")

    save_json(outdir / "memory.json", {"args": vars(args), "results": results})
    print(f"\nwrote: {outdir / 'memory.json'}")


if __name__ == "__main__":
    main()
