"""SQLite-blob baseline VectorStore.

Embeddings stored as float32 BLOBs; brute-force scan-and-rank at query time.
This is the honest comparison for "drop Chroma, keep semantic search inside
SQLite" — claude-mem Issue #707's adjacent option.

Schema (single table):
    docs(id TEXT PK, embedding BLOB, dim INT,
         doc_type TEXT, sqlite_id INT, project TEXT, field_type TEXT,
         created_at_epoch INT, extras_json TEXT)

Where-filter is pushed down for the indexed columns (doc_type, project) and
applied in Python for `extras`. The scan still loads all matching embeddings
into RAM each query — this is the inherent cost of brute-force-without-index
in SQLite, and the honest baseline against which HDF5's chunked reads compete.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
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


# Columns that live in their own SQLite columns (indexable). Everything else
# in metadata goes to extras_json.
INDEXED_COLS = ("doc_type", "sqlite_id", "project", "field_type", "created_at_epoch")


class SQLiteBlobVectorStore(VectorStore):

    def __init__(self, path: str | Path, embedder: Embedder):
        self.path = str(path)
        self.embedder = embedder
        self._dim = embedder.dim
        self._conn = sqlite3.connect(self.path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

    @classmethod
    def open(cls, path: str | Path, embedder: Embedder, **_):
        return cls(path, embedder)

    def _init_schema(self) -> None:
        c = self._conn
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS docs (
                id                 TEXT PRIMARY KEY,
                embedding          BLOB NOT NULL,
                dim                INTEGER NOT NULL,
                doc_type           TEXT,
                sqlite_id          INTEGER,
                project            TEXT,
                field_type         TEXT,
                created_at_epoch   INTEGER,
                extras_json        TEXT
            )
            """
        )
        c.execute("CREATE INDEX IF NOT EXISTS idx_docs_doc_type ON docs(doc_type)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_docs_project  ON docs(project)")
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_docs_proj_type "
            "ON docs(project, doc_type)"
        )
        c.commit()

    # ---- Required ----

    def upsert(self, docs: Sequence[VectorDocument]) -> None:
        if not docs:
            return
        embeddings = self.embedder.embed([d.text for d in docs])
        rows = []
        for doc, emb in zip(docs, embeddings):
            extras = {k: v for k, v in doc.metadata.items() if k not in INDEXED_COLS}
            rows.append(
                (
                    doc.id,
                    emb.astype(np.float32, copy=False).tobytes(),
                    self._dim,
                    doc.metadata.get("doc_type"),
                    doc.metadata.get("sqlite_id"),
                    doc.metadata.get("project"),
                    doc.metadata.get("field_type"),
                    doc.metadata.get("created_at_epoch"),
                    json.dumps(extras) if extras else None,
                )
            )
        self._conn.executemany(
            """
            INSERT INTO docs(id, embedding, dim, doc_type, sqlite_id, project,
                             field_type, created_at_epoch, extras_json)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                embedding=excluded.embedding,
                dim=excluded.dim,
                doc_type=excluded.doc_type,
                sqlite_id=excluded.sqlite_id,
                project=excluded.project,
                field_type=excluded.field_type,
                created_at_epoch=excluded.created_at_epoch,
                extras_json=excluded.extras_json
            """,
            rows,
        )
        self._conn.commit()

    def delete(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        self._conn.executemany("DELETE FROM docs WHERE id = ?", [(i,) for i in ids])
        self._conn.commit()

    def query(
        self,
        query_text: str,
        limit: int,
        where: WhereFilter | None = None,
    ) -> QueryResult:
        q = self.embedder.embed([query_text])[0]  # (D,) float32, unit-norm
        preds = parse_where(where)
        sql, params, residual_preds = self._build_select(preds)

        cur = self._conn.execute(sql, params)
        ids: list[str] = []
        embs: list[np.ndarray] = []
        metas: list[dict] = []

        for row in cur:
            doc_id, blob, dim, doc_type, sqlite_id, project, field_type, ts, extras_json = row
            meta: dict[str, MetadataValue] = {
                "doc_type": doc_type,
                "sqlite_id": sqlite_id,
                "project": project,
                "field_type": field_type,
                "created_at_epoch": ts,
            }
            meta = {k: v for k, v in meta.items() if v is not None}
            if extras_json:
                meta.update(json.loads(extras_json))

            if residual_preds and not matches(meta, residual_preds):
                continue

            ids.append(doc_id)
            embs.append(np.frombuffer(blob, dtype=np.float32))
            metas.append(meta)

        if not embs:
            return QueryResult([], [], [])

        E = np.vstack(embs)                       # (M, D)
        sims = E @ q                              # (M,)
        k = min(limit, sims.size)
        top_local = np.argpartition(-sims, k - 1)[:k]
        top_local = top_local[np.argsort(-sims[top_local])]
        return QueryResult(
            ids=[ids[i] for i in top_local],
            distances=[float(1.0 - sims[i]) for i in top_local],
            metadatas=[metas[i] for i in top_local],
        )

    def list_ids(self, where: WhereFilter | None = None) -> list[str]:
        preds = parse_where(where)
        sql, params, residual_preds = self._build_select(preds, columns="id, extras_json, doc_type, sqlite_id, project, field_type, created_at_epoch")
        cur = self._conn.execute(sql, params)
        if not residual_preds:
            return [r[0] for r in cur]
        out = []
        for r in cur:
            doc_id, extras_json, *vals = r
            cols = dict(zip(("doc_type", "sqlite_id", "project", "field_type", "created_at_epoch"), vals))
            cols = {k: v for k, v in cols.items() if v is not None}
            if extras_json:
                cols.update(json.loads(extras_json))
            if matches(cols, residual_preds):
                out.append(doc_id)
        return out

    def update_metadata(
        self, ids: Sequence[str], patch: Mapping[str, MetadataValue]
    ) -> None:
        if not ids:
            return
        # Split patch into indexed vs. extras.
        indexed_patch = {k: v for k, v in patch.items() if k in INDEXED_COLS}
        extras_patch = {k: v for k, v in patch.items() if k not in INDEXED_COLS}

        for doc_id in ids:
            sets = []
            params: list = []
            if indexed_patch:
                for col, val in indexed_patch.items():
                    sets.append(f"{col} = ?")
                    params.append(val)
            if extras_patch:
                row = self._conn.execute(
                    "SELECT extras_json FROM docs WHERE id = ?", (doc_id,)
                ).fetchone()
                if row is None:
                    continue
                extras = json.loads(row[0]) if row[0] else {}
                extras.update(extras_patch)
                sets.append("extras_json = ?")
                params.append(json.dumps(extras))
            if not sets:
                continue
            params.append(doc_id)
            self._conn.execute(f"UPDATE docs SET {', '.join(sets)} WHERE id = ?", params)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ---- Internals ----

    def _build_select(
        self,
        preds: list[tuple[str, MetadataValue]],
        columns: str = "id, embedding, dim, doc_type, sqlite_id, project, field_type, created_at_epoch, extras_json",
    ) -> tuple[str, list, list[tuple[str, MetadataValue]]]:
        """Push indexed-column equality predicates into SQL; return residuals.

        Residuals (predicates over fields living only in extras_json) get
        applied in Python after the SQL fetch.
        """
        wheres = []
        params: list = []
        residual: list[tuple[str, MetadataValue]] = []
        for field_, value in preds:
            if field_ in INDEXED_COLS:
                wheres.append(f"{field_} = ?")
                params.append(value)
            else:
                residual.append((field_, value))
        sql = f"SELECT {columns} FROM docs"
        if wheres:
            sql += " WHERE " + " AND ".join(wheres)
        return sql, params, residual
