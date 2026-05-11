"""In-memory baseline VectorStore.

dict[id, row] + numpy (N, D) float32 array. Upper bound on speed for a
brute-force scan-and-rank store; serves as the reference impl that
verifies the benchmark harness works end-to-end before HDF5 lands.

Soft-delete via a tombstone mask + free-list (matches the strategy the
HDF5 store will use, so latency comparisons are apples-to-apples).
"""
from __future__ import annotations

from typing import Mapping, Sequence

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


class InMemoryVectorStore(VectorStore):
    INITIAL_CAPACITY = 1024
    GROWTH_FACTOR = 2.0

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self._dim = embedder.dim
        self._capacity = self.INITIAL_CAPACITY
        self._n_used = 0
        self._embeddings = np.zeros((self._capacity, self._dim), dtype=np.float32)
        self._ids: list[str | None] = [None] * self._capacity
        self._metas: list[dict | None] = [None] * self._capacity
        self._tombstoned = np.zeros(self._capacity, dtype=bool)
        self._id_to_row: dict[str, int] = {}
        self._free_rows: list[int] = []

    @classmethod
    def open(cls, embedder: Embedder, **_):
        return cls(embedder)

    # ---- Required ----

    def upsert(self, docs: Sequence[VectorDocument]) -> None:
        if not docs:
            return
        embeddings = self.embedder.embed([d.text for d in docs])
        for doc, emb in zip(docs, embeddings):
            row = self._id_to_row.get(doc.id)
            if row is None:
                row = self._alloc_row()
                self._id_to_row[doc.id] = row
            self._embeddings[row] = emb
            self._ids[row] = doc.id
            self._metas[row] = dict(doc.metadata)
            self._tombstoned[row] = False

    def delete(self, ids: Sequence[str]) -> None:
        for doc_id in ids:
            row = self._id_to_row.pop(doc_id, None)
            if row is None:
                continue
            self._tombstoned[row] = True
            self._ids[row] = None
            self._metas[row] = None
            self._free_rows.append(row)

    def query(
        self,
        query_text: str,
        limit: int,
        where: WhereFilter | None = None,
    ) -> QueryResult:
        if self._n_used == 0:
            return QueryResult([], [], [])

        q = self.embedder.embed([query_text])[0]  # already normalised, (D,)
        preds = parse_where(where)

        # Build candidate mask: live rows that pass the where-filter.
        live = ~self._tombstoned[: self._n_used]
        if preds:
            keep = np.zeros(self._n_used, dtype=bool)
            for r in range(self._n_used):
                if not live[r]:
                    continue
                if matches(self._metas[r], preds):
                    keep[r] = True
            mask = keep
        else:
            mask = live

        if not mask.any():
            return QueryResult([], [], [])

        # Cosine similarity = dot product (rows + q both unit-norm).
        # Score only candidate rows.
        cand_idx = np.flatnonzero(mask)
        sims = self._embeddings[cand_idx] @ q  # (M,)
        # Top-k by similarity (largest first); convert to distance = 1 - sim.
        k = min(limit, sims.size)
        if k <= 0:
            return QueryResult([], [], [])
        top_local = np.argpartition(-sims, k - 1)[:k]
        top_local = top_local[np.argsort(-sims[top_local])]
        rows = cand_idx[top_local]

        return QueryResult(
            ids=[self._ids[r] for r in rows],  # type: ignore[misc]
            distances=[float(1.0 - sims[t]) for t in top_local],
            metadatas=[dict(self._metas[r]) for r in rows],  # type: ignore[arg-type]
        )

    def list_ids(self, where: WhereFilter | None = None) -> list[str]:
        preds = parse_where(where)
        if not preds:
            return [i for i in self._ids[: self._n_used] if i is not None]
        out = []
        for r in range(self._n_used):
            if self._tombstoned[r] or self._ids[r] is None:
                continue
            if matches(self._metas[r], preds):  # type: ignore[arg-type]
                out.append(self._ids[r])  # type: ignore[arg-type]
        return out

    def update_metadata(
        self, ids: Sequence[str], patch: Mapping[str, MetadataValue]
    ) -> None:
        for doc_id in ids:
            row = self._id_to_row.get(doc_id)
            if row is None:
                continue
            self._metas[row].update(patch)  # type: ignore[union-attr]

    def close(self) -> None:
        # In-memory: nothing to flush. Provided for interface symmetry.
        pass

    # ---- Internals ----

    def _alloc_row(self) -> int:
        if self._free_rows:
            return self._free_rows.pop()
        if self._n_used >= self._capacity:
            self._grow()
        row = self._n_used
        self._n_used += 1
        return row

    def _grow(self) -> None:
        new_cap = max(self._capacity + 1, int(self._capacity * self.GROWTH_FACTOR))
        new_embeddings = np.zeros((new_cap, self._dim), dtype=np.float32)
        new_embeddings[: self._capacity] = self._embeddings
        self._embeddings = new_embeddings
        self._ids.extend([None] * (new_cap - self._capacity))
        self._metas.extend([None] * (new_cap - self._capacity))
        new_tomb = np.zeros(new_cap, dtype=bool)
        new_tomb[: self._capacity] = self._tombstoned
        self._tombstoned = new_tomb
        self._capacity = new_cap
