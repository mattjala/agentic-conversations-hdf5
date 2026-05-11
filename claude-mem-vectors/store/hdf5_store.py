"""HDF5-backed VectorStore.

On-disk layout (one file per project):

    /                          attrs: format_version, embedding_dim
    /embeddings  (N, D)  float32   chunks=(chunk_rows, D),  optional gzip + shuffle
    /ids                 vlen-utf8  chunks=(chunk_rows,)
    /tombstoned          uint8      chunks=(chunk_rows,)         # soft delete
    /meta/doc_type           vlen-utf8
    /meta/sqlite_id          int64
    /meta/project            vlen-utf8
    /meta/field_type         vlen-utf8
    /meta/created_at_epoch   int64
    /meta/extras_json        vlen-utf8                             # all overflow keys

Design choices (carried over from the TS sketch):
  * Soft delete via /tombstoned mask + free-list reuse on next upsert.
    HDF5 has no row-shrink; tombstones avoid rewrites on every delete.
  * Indexed columns for the fields claude-mem actually filters on
    (doc_type, sqlite_id, project, field_type, created_at_epoch).
    Everything else falls into extras_json — same partition strategy as
    SQLiteBlobVectorStore so the comparison is apples-to-apples.
  * Single-writer assumption (claude-mem worker is the only writer).
  * Brute-force cosine via numpy: embeddings @ q.T. At N < ~500k this is
    sub-50ms even with disk reads + gzip decompress, which the benchmarks
    will confirm or refute.
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
COMPACT_THRESHOLD = 0.20  # rewrite-on-close if dead-row ratio exceeds this


class HDF5VectorStore(VectorStore):

    def __init__(self, path: str | Path, embedder: Embedder, *,
                 chunk_rows: int = 1024,
                 compression: str | None = "gzip",
                 compression_opts: int | None = 4,
                 shuffle: bool = True):
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

        # In-memory indexes — rebuilt on open.
        self._id_to_row: dict[str, int] = {}
        self._free_rows: list[int] = []
        self._n_used: int = int(self._h5.attrs.get("n_used", 0))
        # Cached metadata columns (live + indexed). Kept in sync on writes.
        # Small footprint (a few MB even at 100k rows) and avoids re-reading
        # whole columns from HDF5 on every query.
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
        if embeddings.shape[1] != self._dim:
            raise ValueError(
                f"Embedder produced dim={embeddings.shape[1]}, store dim={self._dim}"
            )

        # Plan row assignments first (allocates as needed), then write each
        # column in one bulk operation. h5py per-scalar writes are O(syscall);
        # one slice/fancy-index write per column is O(1) in syscalls.
        rows = np.empty(len(docs), dtype=np.int64)
        for i, doc in enumerate(docs):
            r = self._id_to_row.get(doc.id)
            if r is None:
                r = self._alloc_row()
                self._id_to_row[doc.id] = r
            rows[i] = r

        n = len(docs)
        ids_arr        = np.empty(n, dtype=object)
        doc_type_arr   = np.empty(n, dtype=object)
        sqlite_id_arr  = np.empty(n, dtype=np.int64)
        project_arr    = np.empty(n, dtype=object)
        field_type_arr = np.empty(n, dtype=object)
        epoch_arr      = np.empty(n, dtype=np.int64)
        extras_arr     = np.empty(n, dtype=object)

        for i, doc in enumerate(docs):
            m = doc.metadata
            ids_arr[i]        = doc.id
            doc_type_arr[i]   = str(m.get("doc_type", "") or "")
            sid               = m.get("sqlite_id")
            sqlite_id_arr[i]  = int(sid) if sid is not None else -1
            project_arr[i]    = str(m.get("project", "") or "")
            field_type_arr[i] = str(m.get("field_type", "") or "")
            epoch_arr[i]      = int(m.get("created_at_epoch") or 0)
            extras = {k: v for k, v in m.items() if k not in INDEXED_COLS}
            extras_arr[i]     = json.dumps(extras) if extras else ""

        h = self._h5
        # h5py supports fancy indexing on writes; sort the row order so
        # selections are monotonic (fewer chunks touched per column).
        order = np.argsort(rows)
        sorted_rows = rows[order]

        h["/embeddings"][sorted_rows, :]    = embeddings[order].astype(np.float32, copy=False)
        h["/ids"][sorted_rows]              = ids_arr[order]
        h["/tombstoned"][sorted_rows]       = 0
        h["/meta/doc_type"][sorted_rows]    = doc_type_arr[order]
        h["/meta/sqlite_id"][sorted_rows]   = sqlite_id_arr[order]
        h["/meta/project"][sorted_rows]     = project_arr[order]
        h["/meta/field_type"][sorted_rows]  = field_type_arr[order]
        h["/meta/created_at_epoch"][sorted_rows] = epoch_arr[order]
        h["/meta/extras_json"][sorted_rows] = extras_arr[order]

        # Sync cache. Resize the cache arrays to match the dataset capacity.
        self._resize_cache_to(self._n_used)
        self._cache_tomb[sorted_rows] = 0
        self._cache["doc_type"][sorted_rows]         = doc_type_arr[order]
        self._cache["sqlite_id"][sorted_rows]        = sqlite_id_arr[order]
        self._cache["project"][sorted_rows]          = project_arr[order]
        self._cache["field_type"][sorted_rows]       = field_type_arr[order]
        self._cache["created_at_epoch"][sorted_rows] = epoch_arr[order]

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
        self._h5["/tombstoned"][rows_arr] = 1
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

        # Build candidate row-mask using indexed columns first, then residuals.
        mask = self._build_mask(preds)
        if not mask.any():
            return QueryResult([], [], [])

        # Read embeddings for candidate rows. For high-selectivity filters
        # this avoids reading the full dataset; for low-selectivity, h5py's
        # fancy-indexing path falls back to reading whole chunks anyway, so
        # we choose between two strategies.
        cand_idx = np.flatnonzero(mask)
        if cand_idx.size <= self._chunk_rows or cand_idx.size < 0.1 * self._n_used:
            # Sparse selection — fancy index is cheaper.
            embs = self._h5["/embeddings"][cand_idx, :]
        else:
            # Most rows in play — read the full block once and slice.
            embs = self._h5["/embeddings"][: self._n_used][cand_idx]

        sims = embs @ q.astype(np.float32)
        k = min(limit, sims.size)
        if k <= 0:
            return QueryResult([], [], [])
        top_local = np.argpartition(-sims, k - 1)[:k]
        top_local = top_local[np.argsort(-sims[top_local])]
        rows = cand_idx[top_local]

        # Batch reads for the result rows: one HDF5 call per column instead
        # of per row. Sort the indices to keep selections monotonic.
        order = np.argsort(rows)
        sorted_rows = rows[order]
        ids_arr = self._h5["/ids"][sorted_rows]
        extras_arr = self._h5["/meta/extras_json"][sorted_rows]
        # Indexed columns: read from in-memory cache, no HDF5 syscall.
        meta_cols = {col: self._cache[col][sorted_rows] for col in INDEXED_COLS}
        # Reorder back to similarity order.
        inv_order = np.argsort(order)
        ids_out = [self._decode(ids_arr[i]) for i in inv_order]
        metas_out: list[dict[str, MetadataValue]] = []
        for i in inv_order:
            m: dict[str, MetadataValue] = {}
            dt = meta_cols["doc_type"][i]
            if dt: m["doc_type"] = self._decode(dt)
            sid = int(meta_cols["sqlite_id"][i])
            if sid != -1: m["sqlite_id"] = sid
            pj = meta_cols["project"][i]
            if pj: m["project"] = self._decode(pj)
            ft = meta_cols["field_type"][i]
            if ft: m["field_type"] = self._decode(ft)
            ts = int(meta_cols["created_at_epoch"][i])
            if ts: m["created_at_epoch"] = ts
            raw = self._decode(extras_arr[i])
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
        ids_ds = self._h5["/ids"]
        return [self._decode(ids_ds[r]) for r in rows]

    def update_metadata(
        self, ids: Sequence[str], patch: Mapping[str, MetadataValue]
    ) -> None:
        if not ids:
            return
        indexed_patch = {k: v for k, v in patch.items() if k in INDEXED_COLS}
        extras_patch = {k: v for k, v in patch.items() if k not in INDEXED_COLS}

        for doc_id in ids:
            row = self._id_to_row.get(doc_id)
            if row is None:
                continue
            for col, val in indexed_patch.items():
                self._h5[f"/meta/{col}"][row] = val
            if extras_patch:
                ds = self._h5["/meta/extras_json"]
                cur = self._decode(ds[row])
                extras = json.loads(cur) if cur else {}
                extras.update(extras_patch)
                ds[row] = json.dumps(extras)
        self._h5.flush()

    def close(self) -> None:
        if self._h5 is None:
            return
        # Optional compaction: if dead rows exceed threshold, rewrite the file.
        # Disabled by default for now — measurement first, optimisation second.
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

        ds_kwargs = dict(
            chunks=(self._chunk_rows, self._dim),
            compression=self._compression,
            shuffle=self._shuffle,
        )
        if self._compression_opts is not None and self._compression == "gzip":
            ds_kwargs["compression_opts"] = self._compression_opts
        h.create_dataset(
            "/embeddings",
            shape=(0, self._dim),
            maxshape=(None, self._dim),
            dtype="float32",
            **ds_kwargs,
        )
        vlen = h5py.string_dtype(encoding="utf-8")
        h.create_dataset("/ids", shape=(0,), maxshape=(None,),
                         dtype=vlen, chunks=(self._chunk_rows,))
        h.create_dataset("/tombstoned", shape=(0,), maxshape=(None,),
                         dtype="uint8", chunks=(self._chunk_rows,))
        meta = h.create_group("/meta")
        meta.create_dataset("doc_type", shape=(0,), maxshape=(None,),
                            dtype=vlen, chunks=(self._chunk_rows,))
        meta.create_dataset("sqlite_id", shape=(0,), maxshape=(None,),
                            dtype="int64", chunks=(self._chunk_rows,),
                            fillvalue=-1)
        meta.create_dataset("project", shape=(0,), maxshape=(None,),
                            dtype=vlen, chunks=(self._chunk_rows,))
        meta.create_dataset("field_type", shape=(0,), maxshape=(None,),
                            dtype=vlen, chunks=(self._chunk_rows,))
        meta.create_dataset("created_at_epoch", shape=(0,), maxshape=(None,),
                            dtype="int64", chunks=(self._chunk_rows,),
                            fillvalue=0)
        meta.create_dataset("extras_json", shape=(0,), maxshape=(None,),
                            dtype=vlen, chunks=(self._chunk_rows,))

    def _validate_schema(self) -> None:
        v = self._h5.attrs.get("format_version")
        if v is not None and int(v) != FORMAT_VERSION:
            raise RuntimeError(
                f"HDF5 vector store at {self.path} has format_version={v}, "
                f"expected {FORMAT_VERSION}"
            )
        d = self._h5.attrs.get("embedding_dim")
        if d is not None and int(d) != self._dim:
            raise RuntimeError(
                f"Embedding dim mismatch: file={d}, embedder={self._dim}"
            )

    def _rebuild_indexes(self) -> None:
        n = self._n_used
        # Always cache the tombstone column, even when empty.
        self._cache_tomb = self._h5["/tombstoned"][:n].astype(np.uint8) if n else np.zeros(0, dtype=np.uint8)
        # Cache indexed metadata columns.
        for col in INDEXED_COLS:
            self._cache[col] = self._h5[f"/meta/{col}"][:n] if n else np.empty(0)
        if n == 0:
            return
        ids = self._h5["/ids"][:n]
        for row in range(n):
            if self._cache_tomb[row]:
                self._free_rows.append(row)
                continue
            self._id_to_row[self._decode(ids[row])] = row

    def _alloc_row(self) -> int:
        if self._free_rows:
            r = self._free_rows.pop()
            self._h5["/tombstoned"][r] = 0
            return r
        # Need to grow datasets by one row.
        new_size = self._n_used + 1
        self._extend_datasets(new_size)
        row = self._n_used
        self._n_used = new_size
        return row

    def _resize_cache_to(self, n_used: int) -> None:
        if self._cache_tomb.size < n_used:
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

    def _extend_datasets(self, new_n: int) -> None:
        # Grow in geometric chunks to avoid per-row resize cost.
        h = self._h5
        cur_capacity = h["/ids"].shape[0]
        if new_n <= cur_capacity:
            return
        target = max(new_n, max(INITIAL_CAPACITY, int(cur_capacity * GROWTH_FACTOR)))
        h["/embeddings"].resize((target, self._dim))
        h["/ids"].resize((target,))
        h["/tombstoned"].resize((target,))
        for col in ("doc_type", "sqlite_id", "project", "field_type",
                    "created_at_epoch", "extras_json"):
            h[f"/meta/{col}"].resize((target,))

    def _write_row(self, row: int, doc_id: str, emb: np.ndarray,
                   meta: Mapping[str, MetadataValue]) -> None:
        h = self._h5
        h["/embeddings"][row] = emb.astype(np.float32, copy=False)
        h["/ids"][row] = doc_id
        h["/tombstoned"][row] = 0

        h["/meta/doc_type"][row]   = str(meta.get("doc_type", "") or "")
        h["/meta/sqlite_id"][row]  = int(meta.get("sqlite_id", -1) if meta.get("sqlite_id") is not None else -1)
        h["/meta/project"][row]    = str(meta.get("project", "") or "")
        h["/meta/field_type"][row] = str(meta.get("field_type", "") or "")
        h["/meta/created_at_epoch"][row] = int(meta.get("created_at_epoch") or 0)

        extras = {k: v for k, v in meta.items() if k not in INDEXED_COLS}
        h["/meta/extras_json"][row] = json.dumps(extras) if extras else ""

    def _read_meta(self, row: int) -> dict[str, MetadataValue]:
        h = self._h5
        meta: dict[str, MetadataValue] = {}
        dt = self._decode(h["/meta/doc_type"][row])
        if dt:
            meta["doc_type"] = dt
        sid = int(h["/meta/sqlite_id"][row])
        if sid != -1:
            meta["sqlite_id"] = sid
        proj = self._decode(h["/meta/project"][row])
        if proj:
            meta["project"] = proj
        ft = self._decode(h["/meta/field_type"][row])
        if ft:
            meta["field_type"] = ft
        ts = int(h["/meta/created_at_epoch"][row])
        if ts:
            meta["created_at_epoch"] = ts
        extras_raw = self._decode(h["/meta/extras_json"][row])
        if extras_raw:
            try:
                meta.update(json.loads(extras_raw))
            except json.JSONDecodeError:
                pass
        return meta

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
                if isinstance(value, (int, np.integer)):
                    col_match = col == int(value)
                else:
                    if col.dtype == object:
                        # vlen strings: cached as object array of Python str
                        # (h5py decodes when reading). Direct equality works.
                        col_match = col == value
                    else:
                        col_match = col == value
                mask &= col_match
            else:
                residual.append((field_, value))

        if residual:
            # Residuals only need the extras_json column for surviving rows.
            survivors = np.flatnonzero(mask)
            if survivors.size:
                extras = self._h5["/meta/extras_json"][survivors]
                for i, r in enumerate(survivors):
                    raw = self._decode(extras[i])
                    meta_obj = json.loads(raw) if raw else {}
                    if not matches(meta_obj, residual):
                        mask[r] = False
        return mask

    @staticmethod
    def _decode(v) -> str:
        if v is None:
            return ""
        if isinstance(v, bytes):
            return v.decode("utf-8")
        if isinstance(v, np.ndarray) and v.dtype.kind in ("S", "O"):
            if v.size == 0:
                return ""
            item = v.item() if v.shape == () else v[0]
            return item.decode("utf-8") if isinstance(item, bytes) else str(item)
        return str(v)
