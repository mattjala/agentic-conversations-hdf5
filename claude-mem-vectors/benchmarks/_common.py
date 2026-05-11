"""Common scaffolding shared by all benchmark scripts.

Backend factory + paths + percentile helper. Each benchmark script imports
from here so the matrix dimensions stay consistent across files.
"""
from __future__ import annotations

import json
import shutil
import statistics
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

# Make `store` and `benchmarks` importable when scripts run as `python bench_*.py`.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from store import FakeHashEmbedder
from store.inmem_store import InMemoryVectorStore
from store.sqlite_blob_store import SQLiteBlobVectorStore
from store.vector_store import VectorStore
from store.hdf5_store import HDF5VectorStore
from store.hdf5_packed_store import HDF5PackedVectorStore
from store.hdf5_compound_store import HDF5CompoundVectorStore
from store.chroma_store import ChromaVectorStore


# ----- Backend registry -----

# Backends are constructed lazily (factory) so benchmarks can hand in
# different paths / dims / config without us pre-wiring stores at import.

BackendFactory = Callable[[Path, int, dict], VectorStore]


def _build_inmem(_path: Path, dim: int, _cfg: dict) -> VectorStore:
    return InMemoryVectorStore(FakeHashEmbedder(dim=dim))


def _build_sqlite_blob(path: Path, dim: int, _cfg: dict) -> VectorStore:
    if path.exists():
        path.unlink()
    return SQLiteBlobVectorStore(path, FakeHashEmbedder(dim=dim))


def _build_hdf5(path: Path, dim: int, cfg: dict) -> VectorStore:
    if path.exists():
        path.unlink()
    return HDF5VectorStore.open(
        path,
        FakeHashEmbedder(dim=dim),
        chunk_rows=cfg.get("chunk_rows", 1024),
        compression=cfg.get("compression", "gzip"),
        compression_opts=cfg.get("compression_opts", 4),
        shuffle=cfg.get("shuffle", True),
    )


def _build_hdf5_packed(path: Path, dim: int, cfg: dict) -> VectorStore:
    if path.exists():
        path.unlink()
    return HDF5PackedVectorStore.open(
        path,
        FakeHashEmbedder(dim=dim),
        chunk_rows=cfg.get("chunk_rows", 1024),
        compression=cfg.get("compression", "gzip"),
        compression_opts=cfg.get("compression_opts", 4),
        shuffle=cfg.get("shuffle", True),
    )


def _build_hdf5_compound(path: Path, dim: int, cfg: dict) -> VectorStore:
    if path.exists():
        path.unlink()
    return HDF5CompoundVectorStore.open(
        path,
        FakeHashEmbedder(dim=dim),
        chunk_rows=cfg.get("chunk_rows", 1024),
        compression=cfg.get("compression", "gzip"),
        compression_opts=cfg.get("compression_opts", 4),
        shuffle=cfg.get("shuffle", True),
    )


def _build_chroma(path: Path, dim: int, _cfg: dict) -> VectorStore:
    return ChromaVectorStore.open(path, FakeHashEmbedder(dim=dim))


BACKENDS: dict[str, dict] = {
    "inmem": {
        "label": "in-memory (numpy)",
        "color": "#2ca02c",
        "marker": "D",
        "factory": _build_inmem,
        "needs_path": False,
        "ext": "",
    },
    "sqlite_blob": {
        "label": "SQLite + BLOB",
        "color": "#ff7f0e",
        "marker": "^",
        "factory": _build_sqlite_blob,
        "needs_path": True,
        "ext": ".db",
    },
    "hdf5": {
        "label": "HDF5 VLEN",
        "color": "#1f77b4",
        "marker": "o",
        "factory": _build_hdf5,
        "needs_path": True,
        "ext": ".h5",
    },
    "hdf5_packed": {
        "label": "HDF5 packed",
        "color": "#9467bd",
        "marker": "P",
        "factory": _build_hdf5_packed,
        "needs_path": True,
        "ext": ".h5",
    },
    "hdf5_compound": {
        "label": "HDF5 compound",
        "color": "#2ca02c",
        "marker": "D",
        "factory": _build_hdf5_compound,
        "needs_path": True,
        "ext": ".h5",
    },
    "chroma": {
        "label": "Chroma (HNSW)",
        "color": "#d62728",
        "marker": "X",
        "factory": _build_chroma,
        "needs_path": True,
        "ext": "",   # directory, not a file
    },
}


def build_backend(name: str, store_root: Path, label: str, dim: int, cfg: dict | None = None) -> VectorStore:
    spec = BACKENDS[name]
    cfg = cfg or {}
    if spec["needs_path"]:
        path = store_root / f"{name}_{label}{spec['ext']}"
    else:
        path = Path("/dev/null")
    return spec["factory"](path, dim, cfg)


def clean_path(name: str, store_root: Path, label: str) -> None:
    spec = BACKENDS[name]
    if not spec["needs_path"]:
        return
    p = store_root / f"{name}_{label}{spec['ext']}"
    if p.exists():
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()


# ----- Stats helpers -----

def percentile(values: Iterable[float], p: float) -> float:
    arr = sorted(values)
    if not arr:
        return float("nan")
    k = (len(arr) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(arr) - 1)
    frac = k - lo
    return arr[lo] + (arr[hi] - arr[lo]) * frac


def summarise(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {"n": 0}
    return {
        "n": len(samples),
        "mean": statistics.mean(samples),
        "p50": percentile(samples, 50),
        "p95": percentile(samples, 95),
        "p99": percentile(samples, 99),
        "min": min(samples),
        "max": max(samples),
    }


# ----- IO helpers -----

def save_json(path: Path, data) -> None:
    def default(o):
        if is_dataclass(o):
            return asdict(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, Path):
            return str(o)
        raise TypeError(f"not serialisable: {type(o)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=default))


def now_ms() -> float:
    return time.perf_counter() * 1000.0
