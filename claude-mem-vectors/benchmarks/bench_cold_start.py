"""Cold-start latency benchmark.

Decomposes time-to-first-query into its components — the metric that
matters for hook UX, since claude-mem invokes search inside lifecycle
hooks where every blocking ms is user-visible.

Phases measured:
    1. embedder_init_ms     time to construct embedder (model not yet loaded)
    2. open_ms              time to open the store at a pre-existing file
    3. first_embed_ms       time to embed the first query (loads model on first call)
    4. first_query_ms       full first-call query latency (includes 3)
    5. second_query_ms      second query, hot

Run against a pre-populated store (so we measure the open-on-existing-file
path, not the create path).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from _common import (
    BACKENDS,
    BackendFactory,
    build_backend,
    clean_path,
    now_ms,
    save_json,
    summarise,
)
from store import FakeHashEmbedder
from workload import WorkloadConfig, generate, query_pool


DEFAULT_BACKENDS = ["sqlite_blob", "hdf5"]   # inmem has no "cold open" — always fresh
DEFAULT_SIZE = 10_000
DEFAULT_DIM = 384
DEFAULT_TRIALS = 5


def measure_one(backend: str, n: int, dim: int, store_root: Path) -> dict:
    label = f"cold_n{n}_d{dim}"

    # Pre-populate so we measure cold-open of an existing file.
    clean_path(backend, store_root, label)
    store = build_backend(backend, store_root, label, dim)
    docs = generate(WorkloadConfig(n_docs=n))
    for i in range(0, len(docs), 256):
        store.upsert(docs[i : i + 256])
    store.close()

    qs = query_pool(n=2)

    # Phase 1: embedder construction (no model load yet).
    t0 = now_ms()
    embedder = FakeHashEmbedder(dim=dim)
    embedder_init_ms = now_ms() - t0

    # Phase 2: store open (non-destructive — opens the populated file).
    spec = BACKENDS[backend]
    path = store_root / f"{backend}_{label}{spec['ext']}"
    t0 = now_ms()
    if backend == "sqlite_blob":
        from store.sqlite_blob_store import SQLiteBlobVectorStore
        store = SQLiteBlobVectorStore(path, embedder)
    elif backend == "hdf5":
        from store.hdf5_store import HDF5VectorStore
        store = HDF5VectorStore.open(path, embedder)
    elif backend == "hdf5_packed":
        from store.hdf5_packed_store import HDF5PackedVectorStore
        store = HDF5PackedVectorStore.open(path, embedder)
    elif backend == "hdf5_compound":
        from store.hdf5_compound_store import HDF5CompoundVectorStore
        store = HDF5CompoundVectorStore.open(path, embedder)
    elif backend == "chroma":
        # Chroma cold-start: open existing persistent dir without wiping it.
        from store.chroma_store import ChromaVectorStore
        import chromadb
        store = ChromaVectorStore.__new__(ChromaVectorStore)
        store.embedder = embedder
        # The populated data lives in a uuid-named subdir created by open().
        subdirs = [d for d in path.iterdir() if d.is_dir()]
        if not subdirs:
            raise RuntimeError(f"No Chroma data subdirectory found under {path}")
        store._data_dir = subdirs[0]
        store._client = chromadb.PersistentClient(path=str(store._data_dir))
        store._col = store._client.get_or_create_collection(
            name="store", metadata={"hnsw:space": "cosine"}
        )
    else:
        raise ValueError(f"unhandled cold-start backend: {backend}")
    open_ms = now_ms() - t0

    # Phase 3: first embed (in our case FakeHashEmbedder is already loaded;
    # this column will mostly be ~0ms. Kept for parity once MiniLM is wired.)
    t0 = now_ms()
    _ = embedder.embed([qs[0]])
    first_embed_ms = now_ms() - t0

    # Phase 4: first full query.
    t0 = now_ms()
    _ = store.query(qs[0], 20)
    first_query_ms = now_ms() - t0

    # Phase 5: second query (hot).
    t0 = now_ms()
    _ = store.query(qs[1], 20)
    second_query_ms = now_ms() - t0

    store.close()

    return {
        "backend": backend,
        "n": n,
        "dim": dim,
        "embedder_init_ms": embedder_init_ms,
        "open_ms": open_ms,
        "first_embed_ms": first_embed_ms,
        "first_query_ms": first_query_ms,
        "second_query_ms": second_query_ms,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Vector-store cold-start benchmark")
    p.add_argument("--backends", nargs="+", default=DEFAULT_BACKENDS,
                   choices=[b for b in BACKENDS if b != "inmem"])
    p.add_argument("--n", type=int, default=DEFAULT_SIZE)
    p.add_argument("--dim", type=int, default=DEFAULT_DIM)
    p.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    p.add_argument("--quick", action="store_true",
                   help="trials=2, n=1000")
    p.add_argument("--outdir", type=Path, default=Path("results"))
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.trials = 2
        args.n = 1_000

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    store_root = outdir / "stores"
    store_root.mkdir(exist_ok=True)

    print(f"backends={args.backends} n={args.n} dim={args.dim} trials={args.trials}")

    results: list[dict] = []
    for backend in args.backends:
        per_phase: dict[str, list[float]] = {
            "embedder_init_ms": [],
            "open_ms": [],
            "first_embed_ms": [],
            "first_query_ms": [],
            "second_query_ms": [],
        }
        for trial in range(args.trials):
            print(f"  [{backend} trial {trial+1}/{args.trials}] ", end="", flush=True)
            cell = measure_one(backend, args.n, args.dim, store_root)
            for k in per_phase:
                per_phase[k].append(cell[k])
            print(
                f"open={cell['open_ms']:5.1f}ms "
                f"q1={cell['first_query_ms']:6.1f}ms "
                f"q2={cell['second_query_ms']:6.1f}ms"
            )
        results.append({
            "backend": backend,
            "n": args.n,
            "dim": args.dim,
            "trials": args.trials,
            "phases": {k: summarise(v) for k, v in per_phase.items()},
        })

    save_json(outdir / "cold_start.json", {"args": vars(args), "results": results})
    print(f"\nwrote: {outdir / 'cold_start.json'}")


if __name__ == "__main__":
    main()
