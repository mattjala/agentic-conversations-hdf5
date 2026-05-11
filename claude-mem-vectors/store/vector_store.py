"""VectorStore interface — Python mirror of the TS sketch for claude-mem.

The contract is intentionally narrow: only the primitives that ChromaSync
in claude-mem actually exercises against its vector backend. Orchestration
(formatting docs, watermark tracking, backfill loops) lives above this layer
unchanged.

Document IDs follow claude-mem's existing convention:
    obs_{sqlite_id}_{narrative|fact_N|...}
    summary_{sqlite_id}_{request|investigated|...}
    prompt_{sqlite_id}
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

DocType = str  # 'observation' | 'session_summary' | 'user_prompt'
MetadataValue = str | int | float


@dataclass(slots=True)
class VectorDocument:
    id: str
    text: str
    metadata: dict[str, MetadataValue]


@dataclass(slots=True)
class QueryResult:
    ids: list[str]
    distances: list[float]              # cosine distance [0, 2]
    metadatas: list[dict[str, MetadataValue]]


# WhereFilter: subset of Chroma's `where` language that claude-mem actually uses.
#   {field: value}                            -> equality
#   {"$and": [{f1: v1}, {f2: v2}, ...]}       -> conjunction
# Anything richer is rejected by parse_where().
WhereFilter = Mapping[str, object]


# ---------------------------------------------------------------------------
# Where-filter evaluator (shared by all backends)
# ---------------------------------------------------------------------------

def parse_where(where: WhereFilter | None) -> list[tuple[str, MetadataValue]]:
    """Flatten a WhereFilter into a list of (field, value) equality predicates.

    Raises ValueError for anything outside the supported subset.
    """
    if where is None:
        return []
    preds: list[tuple[str, MetadataValue]] = []
    for k, v in where.items():
        if k == "$and":
            if not isinstance(v, list):
                raise ValueError("$and must be a list of clauses")
            for clause in v:
                preds.extend(parse_where(clause))
        elif isinstance(v, (str, int, float)):
            preds.append((k, v))
        else:
            raise ValueError(
                f"Unsupported where clause for field {k!r}: {v!r}. "
                "Only equality and one-level $and are supported."
            )
    return preds


def matches(meta: Mapping[str, MetadataValue], preds: Sequence[tuple[str, MetadataValue]]) -> bool:
    return all(meta.get(field_) == value for field_, value in preds)


# ---------------------------------------------------------------------------
# Embedder protocol (duck-typed — see embedders.py for concrete impls)
# ---------------------------------------------------------------------------

class Embedder:
    """Anything with an `embed(texts) -> ndarray (N, D) float32` method.

    Implementations must guarantee unit-norm rows so cosine similarity reduces
    to a dot product. Stored as float32; queries cast to float32.
    """

    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------------------
# VectorStore ABC
# ---------------------------------------------------------------------------

class VectorStore(ABC):
    """Abstract base for vector-backed document stores.

    Lifecycle:
        store = ConcreteStore.open(path, embedder=...)
        store.upsert(docs)
        result = store.query("text", limit=10, where={"project": "foo"})
        store.close()

    Contract notes:
        * upsert() is idempotent on doc.id — repeated calls overwrite.
        * delete() is a no-op for unknown ids.
        * query() returns at most `limit` results, ranked by ascending cosine
          distance, after applying `where` filtering.
        * IDs returned by listIds() exclude tombstoned/deleted rows.
    """

    embedder: Embedder

    # --- Required ---

    @abstractmethod
    def upsert(self, docs: Sequence[VectorDocument]) -> None: ...

    @abstractmethod
    def delete(self, ids: Sequence[str]) -> None: ...

    @abstractmethod
    def query(
        self,
        query_text: str,
        limit: int,
        where: WhereFilter | None = None,
    ) -> QueryResult: ...

    @abstractmethod
    def list_ids(self, where: WhereFilter | None = None) -> list[str]: ...

    @abstractmethod
    def update_metadata(
        self, ids: Sequence[str], patch: Mapping[str, MetadataValue]
    ) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    # --- Convenience ---

    def __len__(self) -> int:
        return len(self.list_ids())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
