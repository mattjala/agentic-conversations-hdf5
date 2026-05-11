"""HDF5 compound-dataset VectorStore.

Same external contract as HDF5VectorStore, but all per-document metadata
is stored in a single compound dataset rather than nine parallel 1-D datasets.

Layout
------
    /                         attrs: format_version, embedding_dim, n_used
    /embeddings  (N, D)      float32,  chunked + gzip + shuffle  (unchanged)
    /metadata    (N,)        compound  — one row per slot (live or tombstoned)

    compound fields:
        tombstoned          u1
        sqlite_id           i8
        created_at_epoch    i8
        doc_type            S24     fixed-length — inline in chunk, compressible
        project             S48     fixed-length
        field_type          S24     fixed-length
        id                  VLEN    unbounded doc-id string
        extras_json         VLEN    JSON blob for non-indexed metadata

Upsert write cost: 2 h5py fancy-index writes (metadata + embeddings) vs 9 in
the parallel layout — directly attacking the small-batch latency identified in
the benchmark as the main weakness of the original design.

Open/cache-rebuild cost: 1 compound slice vs 7 separate column reads.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import h5py
import numpy as np

from .vector_store import (
    Embedder,
    MetadataValue,
    QueryResult,
    VectorDocument,
    VectorStore,
    WhereFilter,
    matches,
    parse_where,
)

FORMAT_VERSION = 1
INDEXED_COLS = ("doc_type", "sqlite_id", "project", "field_type", "created_at_epoch")
INITIAL_CAPACITY = 256
GROWTH_FACTOR = 2.0

_META_DTYPE = np.dtype([
    ("tombstoned",       np.uint8),
    ("sqlite_id",        np.int64),
    ("created_at_epoch", np.int64),
    ("doc_type",         "S24"),
    ("project",          "S48"),
    ("field_type",       "S24"),
    ("id",               h5py.string_dtype()),
    ("extras_json",      h5py.string_dtype()),
])


def _enc(s: str, maxlen: int) -> bytes:
    """Encode str to fixed-length bytes, truncating if needed."""
    b = s.encode("utf-8") if s else b""
    return b[:maxlen]


def _dec(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (bytes, np.bytes_)):
        return v.rstrip(b"\x00").decode("utf-8", errors="replace")
    return str(v)


class HDF5CompoundVectorStore(VectorStore):
    """HDF5 vector store using a single compound dataset for all metadata."""

    def __init__(
        self,
        path: str | Path,
        embedder: Embedder,
        *,
        chunk_rows: int = 1024,
        compression: str | None = "gzip",
        compression_opts: int | None = 4,
        shuffle: bool = True,
    ):
        self.path = str(path)
        self.embedder = embedder
        self._dim = embedder.dim
        self._chunk_rows = chunk_rows
        self._compression = compression
        self._compression_opts = compression_opts
        self._shuffle = shuffle if compression else False

        new_file = not Path(self.path).exists()
        self._h5 = h5py.File(self.path, "a")

        if new_file:
            self._init_schema()
        self._validate_schema()

        self._n_used: int = int(self._h5.attrs.get("n_used", 0))
        self._id_to_row: dict[str, int] = {}
        self._free_rows: list[int] = []
        self._cache_tomb: np.ndarray = np.zeros(0, dtype=np.uint8)
        self._cache: dict[str, np.ndarray] = {}
        self._rebuild_indexes()

    @classmethod
    def open(cls, path: str | Path, embedder: Embedder, **kwargs):
        return cls(path, embedder, **kwargs)

    # ---- Required ----

    def upsert(self, docs: Sequence[VectorDocument]) -> None:
        if not docs:
            return
        embeddings = self.embedder.embed([d.text for d in docs])

        rows = np.empty(len(docs), dtype=np.int64)
        for i, doc in enumerate(docs):
            r = self._id_to_row.get(doc.id)
            if r is None:
                r = self._alloc_row()
                self._id_to_row[doc.id] = r
            rows[i] = r

        n = len(docs)
        meta_arr = np.empty(n, dtype=_META_DTYPE)
        for i, doc in enumerate(docs):
            m = doc.metadata
            sid = m.get("sqlite_id")
            extras = {k: v for k, v in m.items() if k not in INDEXED_COLS}
            meta_arr[i]["tombstoned"]       = 0
            meta_arr[i]["sqlite_id"]        = int(sid) if sid is not None else -1
            meta_arr[i]["created_at_epoch"] = int(m.get("created_at_epoch") or 0)
            meta_arr[i]["doc_type"]         = _enc(str(m.get("doc_type", "") or ""), 24)
            meta_arr[i]["project"]          = _enc(str(m.get("project", "") or ""), 48)
            meta_arr[i]["field_type"]       = _enc(str(m.get("field_type", "") or ""), 24)
            meta_arr[i]["id"]               = doc.id
            meta_arr[i]["extras_json"]      = json.dumps(extras) if extras else ""

        order = np.argsort(rows)
        sorted_rows = rows[order]

        self._h5["/embeddings"][sorted_rows, :] = embeddings[order].astype(np.float32, copy=False)
        self._h5["/metadata"][sorted_rows]      = meta_arr[order]

        # Sync in-memory cache.
        self._resize_cache_to(self._n_used)
        for i, r in enumerate(sorted_rows):
            row_i = order[np.searchsorted(order, i)] if False else i  # just use loop index
        # Bulk cache update from the written meta_arr (sorted order).
        s_meta = meta_arr[order]
        self._cache_tomb[sorted_rows]              = 0
        self._cache["doc_type"][sorted_rows]       = np.array([_dec(v) for v in s_meta["doc_type"]], dtype=object)
        self._cache["project"][sorted_rows]        = np.array([_dec(v) for v in s_meta["project"]], dtype=object)
        self._cache["field_type"][sorted_rows]     = np.array([_dec(v) for v in s_meta["field_type"]], dtype=object)
        self._cache["sqlite_id"][sorted_rows]      = s_meta["sqlite_id"]
        self._cache["created_at_epoch"][sorted_rows] = s_meta["created_at_epoch"]

        self._h5.attrs["n_used"] = self._n_used
        self._h5.flush()

    def delete(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        rows: list[int] = []
        for doc_id in ids:
            row = self._id_to_row.pop(doc_id, None)
            if row is None:
                continue
            rows.append(row)
            self._free_rows.append(row)
        if not rows:
            return
        rows_arr = np.array(sorted(rows), dtype=np.int64)
        # Write tombstone into the compound dataset field.
        for r in rows_arr:
            self._h5["/metadata"][r]["tombstoned"] = 1
        self._cache_tomb[rows_arr] = 1
        self._h5.flush()

    def query(
        self,
        query_text: str,
        limit: int,
        where: WhereFilter | None = None,
    ) -> QueryResult:
        if self._n_used == 0:
            return QueryResult([], [], [])

        q = self.embedder.embed([query_text])[0]
        preds = parse_where(where)
        mask = self._build_mask(preds)
        if not mask.any():
            return QueryResult([], [], [])

        cand_idx = np.flatnonzero(mask)
        if cand_idx.size <= self._chunk_rows or cand_idx.size < 0.1 * self._n_used:
            embs = self._h5["/embeddings"][cand_idx, :]
        else:
            embs = self._h5["/embeddings"][: self._n_used][cand_idx]

        sims = embs @ q.astype(np.float32)
        k = min(limit, sims.size)
        if k <= 0:
            return QueryResult([], [], [])
        top_local = np.argpartition(-sims, k - 1)[:k]
        top_local = top_local[np.argsort(-sims[top_local])]
        rows = cand_idx[top_local]

        order = np.argsort(rows)
        sorted_rows = rows[order]
        # One compound slice for all result rows.
        meta_rows = self._h5["/metadata"][sorted_rows]
        inv_order = np.argsort(order)

        ids_out = [_dec(meta_rows["id"][i]) for i in inv_order]
        metas_out: list[dict[str, MetadataValue]] = []
        for i in inv_order:
            row_m = meta_rows[i]
            m: dict[str, MetadataValue] = {}
            dt = _dec(row_m["doc_type"])
            if dt: m["doc_type"] = dt
            sid = int(row_m["sqlite_id"])
            if sid != -1: m["sqlite_id"] = sid
            pj = _dec(row_m["project"])
            if pj: m["project"] = pj
            ft = _dec(row_m["field_type"])
            if ft: m["field_type"] = ft
            ts = int(row_m["created_at_epoch"])
            if ts: m["created_at_epoch"] = ts
            raw = _dec(row_m["extras_json"])
            if raw:
                try: m.update(json.loads(raw))
                except json.JSONDecodeError: pass
            metas_out.append(m)

        return QueryResult(
            ids=ids_out,
            distances=[float(1.0 - sims[t]) for t in top_local],
            metadatas=metas_out,
        )

    def list_ids(self, where: WhereFilter | None = None) -> list[str]:
        preds = parse_where(where)
        mask = self._build_mask(preds)
        if not mask.any():
            return []
        rows = np.flatnonzero(mask)
        meta_ds = self._h5["/metadata"]
        return [_dec(meta_ds[r]["id"]) for r in rows]

    def update_metadata(
        self, ids: Sequence[str], patch: Mapping[str, MetadataValue]
    ) -> None:
        if not ids:
            return
        for doc_id in ids:
            row = self._id_to_row.get(doc_id)
            if row is None:
                continue
            row_data = self._h5["/metadata"][row]
            for col, val in patch.items():
                if col == "doc_type":
                    row_data["doc_type"] = _enc(str(val), 24)
                    self._cache["doc_type"][row] = str(val)
                elif col == "project":
                    row_data["project"] = _enc(str(val), 48)
                    self._cache["project"][row] = str(val)
                elif col == "field_type":
                    row_data["field_type"] = _enc(str(val), 24)
                    self._cache["field_type"][row] = str(val)
                elif col == "sqlite_id":
                    row_data["sqlite_id"] = int(val)
                    self._cache["sqlite_id"][row] = int(val)
                elif col == "created_at_epoch":
                    row_data["created_at_epoch"] = int(val)
                    self._cache["created_at_epoch"][row] = int(val)
                else:
                    cur = _dec(row_data["extras_json"])
                    extras = json.loads(cur) if cur else {}
                    extras[col] = val
                    row_data["extras_json"] = json.dumps(extras)
            self._h5["/metadata"][row] = row_data
        self._h5.flush()

    def close(self) -> None:
        if self._h5 is None:
            return
        self._h5.attrs["n_used"] = self._n_used
        self._h5.flush()
        self._h5.close()
        self._h5 = None  # type: ignore[assignment]

    # ---- Internals ----

    def _init_schema(self) -> None:
        h = self._h5
        h.attrs["format_version"] = FORMAT_VERSION
        h.attrs["embedding_dim"] = self._dim
        h.attrs["n_used"] = 0

        emb_kw: dict = dict(
            shape=(0, self._dim), maxshape=(None, self._dim), dtype="float32",
            chunks=(self._chunk_rows, self._dim),
        )
        if self._compression:
            emb_kw["compression"] = self._compression
            emb_kw["shuffle"] = self._shuffle
            if self._compression_opts is not None:
                emb_kw["compression_opts"] = self._compression_opts
        h.create_dataset("/embeddings", **emb_kw)

        meta_kw: dict = dict(
            shape=(0,), maxshape=(None,), dtype=_META_DTYPE,
            chunks=(self._chunk_rows,),
        )
        if self._compression:
            meta_kw["compression"] = self._compression
            if self._compression_opts is not None:
                meta_kw["compression_opts"] = self._compression_opts
        h.create_dataset("/metadata", **meta_kw)

    def _validate_schema(self) -> None:
        v = self._h5.attrs.get("format_version")
        if v is not None and int(v) != FORMAT_VERSION:
            raise RuntimeError(f"format_version mismatch: {v} != {FORMAT_VERSION}")
        d = self._h5.attrs.get("embedding_dim")
        if d is not None and int(d) != self._dim:
            raise RuntimeError(f"embedding_dim mismatch: {d} != {self._dim}")

    def _rebuild_indexes(self) -> None:
        n = self._n_used
        self._cache_tomb = np.zeros(0, dtype=np.uint8)
        for col, dtype in (
            ("doc_type", object), ("project", object), ("field_type", object),
            ("sqlite_id", np.int64), ("created_at_epoch", np.int64),
        ):
            self._cache[col] = np.empty(0, dtype=dtype)

        if n == 0:
            return

        # One compound slice — all fields for all live rows in one read.
        rows = self._h5["/metadata"][:n]
        self._cache_tomb = rows["tombstoned"].astype(np.uint8)
        self._cache["doc_type"]         = np.array([_dec(v) for v in rows["doc_type"]], dtype=object)
        self._cache["project"]          = np.array([_dec(v) for v in rows["project"]], dtype=object)
        self._cache["field_type"]       = np.array([_dec(v) for v in rows["field_type"]], dtype=object)
        self._cache["sqlite_id"]        = rows["sqlite_id"].astype(np.int64)
        self._cache["created_at_epoch"] = rows["created_at_epoch"].astype(np.int64)

        for row in range(n):
            if rows["tombstoned"][row]:
                self._free_rows.append(row)
            else:
                self._id_to_row[_dec(rows["id"][row])] = row

    def _alloc_row(self) -> int:
        if self._free_rows:
            return self._free_rows.pop()
        new_size = self._n_used + 1
        self._extend_datasets(new_size)
        row = self._n_used
        self._n_used = new_size
        return row

    def _extend_datasets(self, new_n: int) -> None:
        cur = self._h5["/metadata"].shape[0]
        if new_n <= cur:
            return
        target = max(new_n, max(INITIAL_CAPACITY, int(cur * GROWTH_FACTOR)))
        self._h5["/embeddings"].resize((target, self._dim))
        self._h5["/metadata"].resize((target,))

    def _resize_cache_to(self, n_used: int) -> None:
        if self._cache_tomb.size >= n_used:
            return
        new_cap = max(n_used, max(INITIAL_CAPACITY, int(self._cache_tomb.size * GROWTH_FACTOR)))
        new_tomb = np.zeros(new_cap, dtype=np.uint8)
        new_tomb[: self._cache_tomb.size] = self._cache_tomb
        self._cache_tomb = new_tomb
        for col, dtype in (
            ("doc_type", object), ("project", object), ("field_type", object),
            ("sqlite_id", np.int64), ("created_at_epoch", np.int64),
        ):
            old = self._cache.get(col)
            if dtype is object:
                new = np.empty(new_cap, dtype=object)
                new[:] = ""
            else:
                new = np.zeros(new_cap, dtype=dtype)
            if old is not None and old.size:
                new[: old.size] = old
            self._cache[col] = new

    def _build_mask(self, preds: list[tuple[str, MetadataValue]]) -> np.ndarray:
        n = self._n_used
        if n == 0:
            return np.zeros(0, dtype=bool)
        live = self._cache_tomb[:n] == 0
        if not preds:
            return live
        mask = live.copy()
        residual: list[tuple[str, MetadataValue]] = []
        for field_, value in preds:
            if field_ in INDEXED_COLS:
                col = self._cache[field_][:n]
                mask &= (col == value)
            else:
                residual.append((field_, value))
        if residual:
            survivors = np.flatnonzero(mask)
            if survivors.size:
                meta_rows = self._h5["/metadata"][survivors]
                for i, r in enumerate(survivors):
                    raw = _dec(meta_rows["extras_json"][i])
                    meta_obj = json.loads(raw) if raw else {}
                    if not matches(meta_obj, residual):
                        mask[r] = False
        return mask
