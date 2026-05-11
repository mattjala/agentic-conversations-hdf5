"""File-size benchmark.

Measures bytes-on-disk after writing N docs, varying:
    * backend         (sqlite_blob / hdf5 — inmem skipped, no disk footprint)
    * N               (1k / 10k / 100k)
    * dim             (384 / 768 / 1536)
    * compression     (HDF5 only: none / gzip-1 / gzip-4 / gzip-9)

The compression knobs are HDF5-specific. SQLite has no per-row compression
that maps cleanly onto this comparison; reported as a single config.

Output: bytes per backend × N × dim × compression. Useful for deciding the
default chunk/compression config before benchmarking query latency at the
chosen settings.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from _common import BACKENDS, build_backend, clean_path, save_json
from workload import WorkloadConfig, generate


DEFAULT_SIZES = [1_000, 10_000, 100_000]
DEFAULT_DIMS = [384]
DEFAULT_BACKENDS = ["sqlite_blob", "hdf5"]
DEFAULT_HDF5_CONFIGS = [
    {"name": "none",   "compression": None,   "compression_opts": None, "shuffle": False},
    {"name": "gzip1",  "compression": "gzip", "compression_opts": 1,    "shuffle": True},
    {"name": "gzip4",  "compression": "gzip", "compression_opts": 4,    "shuffle": True},
    {"name": "gzip9",  "compression": "gzip", "compression_opts": 9,    "shuffle": True},
]


def file_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return path.stat().st_size


def run_cell(backend: str, n: int, dim: int, hdf5_cfg: dict | None, store_root: Path) -> dict:
    label = f"n{n}_d{dim}_{hdf5_cfg['name'] if hdf5_cfg else 'default'}"
    clean_path(backend, store_root, label)
    store = build_backend(backend, store_root, label, dim, hdf5_cfg or {})

    docs = generate(WorkloadConfig(n_docs=n))
    batch = 256
    for i in range(0, len(docs), batch):
        store.upsert(docs[i : i + batch])
    store.close()

    spec = BACKENDS[backend]
    p = store_root / f"{backend}_{label}{spec['ext']}"
    bytes_on_disk = file_bytes(p)

    return {
        "backend": backend,
        "n": n,
        "dim": dim,
        "compression": (hdf5_cfg or {}).get("name", "default"),
        "bytes": bytes_on_disk,
        "bytes_per_doc": bytes_on_disk / n if n else 0,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Vector-store on-disk size benchmark")
    p.add_argument("--backends", nargs="+", default=DEFAULT_BACKENDS,
                   choices=[b for b in BACKENDS if b != "inmem"])
    p.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES)
    p.add_argument("--dims", nargs="+", type=int, default=DEFAULT_DIMS)
    p.add_argument("--quick", action="store_true",
                   help="tiny matrix: sizes=[1000], dims=[384], gzip4 only")
    p.add_argument("--outdir", type=Path, default=Path("results"))
    return p.parse_args()


def main():
    args = parse_args()
    hdf5_configs = DEFAULT_HDF5_CONFIGS
    if args.quick:
        args.sizes = [1_000]
        args.dims = [384]
        hdf5_configs = [c for c in DEFAULT_HDF5_CONFIGS if c["name"] == "gzip4"]

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    store_root = outdir / "stores"
    store_root.mkdir(exist_ok=True)

    print(f"matrix: backends={args.backends} sizes={args.sizes} dims={args.dims}")

    results: list[dict] = []
    for backend in args.backends:
        for n in args.sizes:
            for dim in args.dims:
                if backend == "hdf5":
                    for cfg in hdf5_configs:
                        print(f"  [hdf5 n={n} d={dim} comp={cfg['name']}] ", end="", flush=True)
                        cell = run_cell("hdf5", n, dim, cfg, store_root)
                        results.append(cell)
                        print(f"{cell['bytes']/1e3:>9,.1f} kB  ({cell['bytes_per_doc']:.1f} B/doc)")
                else:
                    print(f"  [{backend} n={n} d={dim}] ", end="", flush=True)
                    cell = run_cell(backend, n, dim, None, store_root)
                    results.append(cell)
                    print(f"{cell['bytes']/1e3:>9,.1f} kB  ({cell['bytes_per_doc']:.1f} B/doc)")

    save_json(outdir / "file_size.json", {"args": vars(args), "results": results})
    print(f"\nwrote: {outdir / 'file_size.json'}")


if __name__ == "__main__":
    main()
