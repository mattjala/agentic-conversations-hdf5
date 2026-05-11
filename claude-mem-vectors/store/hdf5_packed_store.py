"""HDF5 packed-bytes VectorStore.

Same external contract as HDF5VectorStore, but unbounded string columns
(doc ids and extras_json) are stored as flat uint8 byte buffers rather
than VLEN string datasets, so gzip can compress them.

Bounded metadata strings (doc_type, project, field_type) are stored as
fixed-length byte-string datasets (S24 / S48 / S24), which are also
compressible since they live in fixed-width chunked storage.

Layout
------
    /                           attrs: format_version, embedding_dim, n_used
    /embeddings  (N, D)        float32   chunked + gzip + shuffle  (unchanged)
    /tombstoned  (N,)          uint8
    /sqlite_id   (N,)          int64
    /created_at_epoch (N,)     int64
    /doc_type    (N,)          S24   fixed-length, compressible
    /project     (N,)          S48   fixed-length, compressible
    /field_type  (N,)          S24   fixed-length, compressible
    /str_bytes   (B,)          uint8   flat buffer: id_bytes || extras_bytes
    /str_index   (N,)          compound(id_off u8, id_len u4, ex_off u8, ex_len u4)

String buffer notes:
  * On upsert of a NEW row: append id + extras bytes, record offsets.
  * On upsert of an EXISTING row (overwrite): append new bytes and update
    the index entry, leaving the old bytes as dead space (amortised waste
    is small for claude-mem's near-pure-insert workload).
  * The buffer grows monotonically; compaction can be added later if needed.
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
STR_CHUNK_BYTES = 64 * 1024

_STR_INDEX_DTYPE = np.dtype([
    ("id_off",  np.uint64),
    ("id_len",  np.uint32),
    ("ex_off",  np.uint64),
    ("ex_len",  np.uint32),
])


def _dec_fixed(v) -> str:
    if isinstance(v, (bytes, np.bytes_)):
        return v.rstrip(b"\x00").decode("utf-8", errors="replace")
    return str(v) if v is not None else ""


def _dec_vlen(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    return str(v)


class HDF5PackedVectorStore(VectorStore):
    """HDF5 vector store with byte-buffer encoding for variable-length strings."""

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
        doc_type_arr   = np.empty(n, dtype="S24")
        project_arr    = np.empty(n, dtype="S48")
        field_type_arr = np.empty(n, dtype="S24")
        sqlite_id_arr  = np.empty(n, dtype=np.int64)
        epoch_arr      = np.empty(n, dtype=np.int64)
        id_bufs:      list[bytes] = []
        extras_bufs:  list[bytes] = []

        for i, doc in enumerate(docs):
            m = doc.metadata
            sid = m.get("sqlite_id")
            extras = {k: v for k, v in m.items() if k not in INDEXED_COLS}
            doc_type_arr[i]   = str(m.get("doc_type", "") or "")[:24].encode()
            project_arr[i]    = str(m.get("project", "") or "")[:48].encode()
            field_type_arr[i] = str(m.get("field_type", "") or "")[:24].encode()
            sqlite_id_arr[i]  = int(sid) if sid is not None else -1
            epoch_arr[i]      = int(m.get("created_at_epoch") or 0)
            id_bufs.append(doc.id.encode("utf-8"))
            extras_bufs.append(json.dumps(extras).encode("utf-8") if extras else b"")

        # Append id and extras bytes to the flat buffer.
        str_ds = self._h5["/str_bytes"]
        idx_ds = self._h5["/str_index"]
        base = int(str_ds.shape[0])
        total = sum(len(b) for b in id_bufs) + sum(len(b) for b in extras_bufs)
        str_ds.resize((base + total,))

        new_idx = np.zeros(n, dtype=_STR_INDEX_DTYPE)
        pos = base
        for i in range(n):
            id_b  = id_bufs[i]
            ex_b  = extras_bufs[i]
            if id_b:
                str_ds[pos:pos + len(id_b)] = np.frombuffer(id_b, dtype=np.uint8)
            new_idx[i]["id_off"] = pos
            new_idx[i]["id_len"] = len(id_b)
            pos += len(id_b)
            if ex_b:
                str_ds[pos:pos + len(ex_b)] = np.frombuffer(ex_b, dtype=np.uint8)
            new_idx[i]["ex_off"] = pos
            new_idx[i]["ex_len"] = len(ex_b)
            pos += len(ex_b)

        order = np.argsort(rows)
        sorted_rows = rows[order]

        self._h5["/embeddings"][sorted_rows, :]    = embeddings[order].astype(np.float32, copy=False)
        self._h5["/tombstoned"][sorted_rows]        = 0
        self._h5["/doc_type"][sorted_rows]          = doc_type_arr[order]
        self._h5["/project"][sorted_rows]           = project_arr[order]
        self._h5["/field_type"][sorted_rows]        = field_type_arr[order]
        self._h5["/sqlite_id"][sorted_rows]         = sqlite_id_arr[order]
        self._h5["/created_at_epoch"][sorted_rows]  = epoch_arr[order]
        self._h5["/str_index"][sorted_rows]         = new_idx[order]

        self._resize_cache_to(self._n_used)
        s = order  # sorted order alias
        self._cache_tomb[sorted_rows]              = 0
        self._cache["doc_type"][sorted_rows]       = np.array([_dec_fixed(v) for v in doc_type_arr[s]], dtype=object)
        self._cache["project"][sorted_rows]        = np.array([_dec_fixed(v) for v in project_arr[s]], dtype=object)
        self._cache["field_type"][sorted_rows]     = np.array([_dec_fixed(v) for v in field_type_arr[s]], dtype=object)
        self._cache["sqlite_id"][sorted_rows]      = sqlite_id_arr[order]
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
        idx_rows = self._h5["/str_index"][sorted_rows]
        inv_order = np.argsort(order)

        # Read id strings from byte buffer.
        ids_out: list[str] = []
        metas_out: list[dict[str, MetadataValue]] = []
        str_ds = self._h5["/str_bytes"]
        for i in inv_order:
            entry = idx_rows[i]
            id_off, id_len = int(entry["id_off"]), int(entry["id_len"])
            ex_off, ex_len = int(entry["ex_off"]), int(entry["ex_len"])
            doc_id = bytes(str_ds[id_off: id_off + id_len]).decode("utf-8") if id_len else ""
            ids_out.append(doc_id)

            r = sorted_rows[i]
            m: dict[str, MetadataValue] = {}
            dt = _dec_fixed(self._h5["/doc_type"][r])
            if dt: m["doc_type"] = dt
            sid = int(self._h5["/sqlite_id"][r])
            if sid != -1: m["sqlite_id"] = sid
            pj = _dec_fixed(self._h5["/project"][r])
            if pj: m["project"] = pj
            ft = _dec_fixed(self._h5["/field_type"][r])
            if ft: m["field_type"] = ft
            ts = int(self._h5["/created_at_epoch"][r])
            if ts: m["created_at_epoch"] = ts
            if ex_len:
                raw = bytes(str_ds[ex_off: ex_off + ex_len]).decode("utf-8")
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
        str_ds = self._h5["/str_bytes"]
        idx_ds = self._h5["/str_index"]
        out = []
        for r in rows:
            entry = idx_ds[r]
            off, ln = int(entry["id_off"]), int(entry["id_len"])
            out.append(bytes(str_ds[off: off + ln]).decode("utf-8") if ln else "")
        return out

    def update_metadata(
        self, ids: Sequence[str], patch: Mapping[str, MetadataValue]
    ) -> None:
        if not ids:
            return
        str_ds = self._h5["/str_bytes"]
        idx_ds = self._h5["/str_index"]
        indexed_patch = {k: v for k, v in patch.items() if k in INDEXED_COLS}
        extras_patch  = {k: v for k, v in patch.items() if k not in INDEXED_COLS}

        for doc_id in ids:
            row = self._id_to_row.get(doc_id)
            if row is None:
                continue
            if "doc_type" in indexed_patch:
                v = str(indexed_patch["doc_type"])[:24].encode()
                self._h5["/doc_type"][row] = v
                self._cache["doc_type"][row] = _dec_fixed(v)
            if "project" in indexed_patch:
                v = str(indexed_patch["project"])[:48].encode()
                self._h5["/project"][row] = v
                self._cache["project"][row] = _dec_fixed(v)
            if "field_type" in indexed_patch:
                v = str(indexed_patch["field_type"])[:24].encode()
                self._h5["/field_type"][row] = v
                self._cache["field_type"][row] = _dec_fixed(v)
            if "sqlite_id" in indexed_patch:
                self._h5["/sqlite_id"][row] = int(indexed_patch["sqlite_id"])
                self._cache["sqlite_id"][row] = int(indexed_patch["sqlite_id"])
            if "created_at_epoch" in indexed_patch:
                self._h5["/created_at_epoch"][row] = int(indexed_patch["created_at_epoch"])
                self._cache["created_at_epoch"][row] = int(indexed_patch["created_at_epoch"])
            if extras_patch:
                entry = idx_ds[row]
                ex_off, ex_len = int(entry["ex_off"]), int(entry["ex_len"])
                cur_raw = bytes(str_ds[ex_off: ex_off + ex_len]).decode("utf-8") if ex_len else ""
                extras = json.loads(cur_raw) if cur_raw else {}
                extras.update(extras_patch)
                new_ex_b = json.dumps(extras).encode("utf-8")
                base = int(str_ds.shape[0])
                str_ds.resize((base + len(new_ex_b),))
                str_ds[base: base + len(new_ex_b)] = np.frombuffer(new_ex_b, dtype=np.uint8)
                new_entry = np.zeros(1, dtype=_STR_INDEX_DTYPE)
                new_entry[0]["id_off"] = int(entry["id_off"])
                new_entry[0]["id_len"] = int(entry["id_len"])
                new_entry[0]["ex_off"] = base
                new_entry[0]["ex_len"] = len(new_ex_b)
                idx_ds[row] = new_entry[0]
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

        def _ds(name, shape, maxshape, dtype, **extra):
            kw: dict = dict(shape=shape, maxshape=maxshape, dtype=dtype,
                            chunks=(self._chunk_rows,) if len(shape) == 1 else (self._chunk_rows, self._dim))
            if self._compression:
                kw["compression"] = self._compression
                kw["shuffle"] = self._shuffle
                if self._compression_opts is not None:
                    kw["compression_opts"] = self._compression_opts
            kw.update(extra)
            return h.create_dataset(name, **kw)

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

        h.create_dataset("/tombstoned", shape=(0,), maxshape=(None,), dtype=np.uint8,
                         chunks=(self._chunk_rows,))
        h.create_dataset("/sqlite_id", shape=(0,), maxshape=(None,), dtype=np.int64,
                         chunks=(self._chunk_rows,), fillvalue=-1)
        h.create_dataset("/created_at_epoch", shape=(0,), maxshape=(None,), dtype=np.int64,
                         chunks=(self._chunk_rows,), fillvalue=0)

        for name, slen in (("/doc_type", "S24"), ("/project", "S48"), ("/field_type", "S24")):
            kw: dict = dict(shape=(0,), maxshape=(None,), dtype=slen, chunks=(self._chunk_rows,))
            if self._compression:
                kw["compression"] = self._compression
                if self._compression_opts is not None:
                    kw["compression_opts"] = self._compression_opts
            h.create_dataset(name, **kw)

        # Byte buffer for ids + extras_json.
        buf_kw: dict = dict(shape=(0,), maxshape=(None,), dtype=np.uint8,
                            chunks=(STR_CHUNK_BYTES,))
        if self._compression:
            buf_kw["compression"] = self._compression
            buf_kw["shuffle"] = True
            if self._compression_opts is not None:
                buf_kw["compression_opts"] = self._compression_opts
        h.create_dataset("/str_bytes", **buf_kw)

        idx_kw: dict = dict(shape=(0,), maxshape=(None,), dtype=_STR_INDEX_DTYPE,
                            chunks=(self._chunk_rows,))
        if self._compression:
            idx_kw["compression"] = self._compression
            idx_kw["shuffle"] = True
            if self._compression_opts is not None:
                idx_kw["compression_opts"] = self._compression_opts
        h.create_dataset("/str_index", **idx_kw)

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

        self._cache_tomb      = self._h5["/tombstoned"][:n].astype(np.uint8)
        dt_raw                = self._h5["/doc_type"][:n]
        pj_raw                = self._h5["/project"][:n]
        ft_raw                = self._h5["/field_type"][:n]
        self._cache["doc_type"]         = np.array([_dec_fixed(v) for v in dt_raw], dtype=object)
        self._cache["project"]          = np.array([_dec_fixed(v) for v in pj_raw], dtype=object)
        self._cache["field_type"]       = np.array([_dec_fixed(v) for v in ft_raw], dtype=object)
        self._cache["sqlite_id"]        = self._h5["/sqlite_id"][:n].astype(np.int64)
        self._cache["created_at_epoch"] = self._h5["/created_at_epoch"][:n].astype(np.int64)

        idx = self._h5["/str_index"][:n]
        str_ds = self._h5["/str_bytes"]
        for row in range(n):
            if self._cache_tomb[row]:
                self._free_rows.append(row)
                continue
            id_off = int(idx[row]["id_off"])
            id_len = int(idx[row]["id_len"])
            doc_id = bytes(str_ds[id_off: id_off + id_len]).decode("utf-8") if id_len else ""
            self._id_to_row[doc_id] = row

    def _alloc_row(self) -> int:
        if self._free_rows:
            return self._free_rows.pop()
        new_size = self._n_used + 1
        self._extend_datasets(new_size)
        row = self._n_used
        self._n_used = new_size
        return row

    def _extend_datasets(self, new_n: int) -> None:
        cur = self._h5["/tombstoned"].shape[0]
        if new_n <= cur:
            return
        target = max(new_n, max(INITIAL_CAPACITY, int(cur * GROWTH_FACTOR)))
        self._h5["/embeddings"].resize((target, self._dim))
        for name in ("/tombstoned", "/sqlite_id", "/created_at_epoch",
                     "/doc_type", "/project", "/field_type", "/str_index"):
            self._h5[name].resize((target,))
        # /str_bytes grows separately by appending.

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
                str_ds = self._h5["/str_bytes"]
                idx_rows = self._h5["/str_index"][survivors]
                for i, r in enumerate(survivors):
                    ex_off = int(idx_rows[i]["ex_off"])
                    ex_len = int(idx_rows[i]["ex_len"])
                    raw = bytes(str_ds[ex_off: ex_off + ex_len]).decode("utf-8") if ex_len else ""
                    meta_obj = json.loads(raw) if raw else {}
                    if not matches(meta_obj, residual):
                        mask[r] = False
        return mask
