"""SQLite agent session backend.

Schema
------
turns (id TEXT, session_id TEXT, role TEXT, content TEXT, timestamp REAL,
       embedding BLOB)
tool_calls (id TEXT, session_id TEXT, name TEXT, args TEXT, result_text TEXT,
            result_data BLOB, timestamp REAL)
artifacts  (session_id TEXT, name TEXT, data BLOB, dtype TEXT, shape TEXT)
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .base import SessionBackend


class SQLiteSession(SessionBackend):
    def __init__(self, path: str | Path, session_id: str = ""):
        self.path = Path(path)
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self._conn = sqlite3.connect(str(self.path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS turns (
                id         TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role       TEXT NOT NULL,
                content    TEXT NOT NULL,
                timestamp  REAL NOT NULL,
                embedding  BLOB
            );
            CREATE TABLE IF NOT EXISTS tool_calls (
                id          TEXT PRIMARY KEY,
                session_id  TEXT NOT NULL,
                name        TEXT NOT NULL,
                args        TEXT NOT NULL,
                result_text TEXT,
                result_data BLOB,
                dtype       TEXT,
                shape       TEXT,
                timestamp   REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS artifacts (
                session_id TEXT NOT NULL,
                name       TEXT NOT NULL,
                data       BLOB NOT NULL,
                dtype      TEXT NOT NULL,
                shape      TEXT NOT NULL,
                PRIMARY KEY (session_id, name)
            );
        """)
        self._conn.commit()

    # ------------------------------------------------------------------

    def add_turn(
        self,
        role: str,
        content: str,
        embedding: Optional[np.ndarray] = None,
    ) -> str:
        turn_id = str(uuid.uuid4())[:12]
        emb_blob = embedding.astype(np.float32).tobytes() if embedding is not None else None
        self._conn.execute(
            "INSERT INTO turns (id, session_id, role, content, timestamp, embedding) "
            "VALUES (?,?,?,?,?,?)",
            (turn_id, self.session_id, role, content, time.time(), emb_blob),
        )
        self._conn.commit()
        return turn_id

    def add_tool_call(
        self,
        name: str,
        args: dict[str, Any],
        result_text: Optional[str] = None,
        result_data: Optional[np.ndarray] = None,
    ) -> str:
        call_id = str(uuid.uuid4())[:12]
        rd_blob = dtype = shape = None
        if result_data is not None:
            rd_blob = result_data.tobytes()
            dtype = str(result_data.dtype)
            shape = json.dumps(list(result_data.shape))
        self._conn.execute(
            "INSERT INTO tool_calls "
            "(id, session_id, name, args, result_text, result_data, dtype, shape, timestamp) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (call_id, self.session_id, name, json.dumps(args),
             result_text, rd_blob, dtype, shape, time.time()),
        )
        self._conn.commit()
        return call_id

    def get_recent_context(self, n: int = 20) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT id, role, content, timestamp, embedding FROM turns "
            "WHERE session_id=? ORDER BY timestamp DESC LIMIT ?",
            (self.session_id, n),
        )
        rows = cur.fetchall()
        # rows are DESC; reverse to chronological order
        rows = rows[::-1]

        results = []
        for row in rows:
            tid, role, content, ts, emb_blob = row
            d: dict[str, Any] = {
                "turn_id": tid,
                "role": role,
                "content": content,
                "timestamp": ts,
            }
            if emb_blob is not None:
                d["embedding"] = np.frombuffer(emb_blob, dtype=np.float32)
            results.append(d)
        return results

    def store_artifact(self, name: str, data: np.ndarray) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO artifacts (session_id, name, data, dtype, shape) "
            "VALUES (?,?,?,?,?)",
            (self.session_id, name, data.tobytes(),
             str(data.dtype), json.dumps(list(data.shape))),
        )
        self._conn.commit()

    def get_artifact(self, name: str) -> Optional[np.ndarray]:
        cur = self._conn.execute(
            "SELECT data, dtype, shape FROM artifacts WHERE session_id=? AND name=?",
            (self.session_id, name),
        )
        row = cur.fetchone()
        if row is None:
            return None
        data_blob, dtype_str, shape_str = row
        shape = tuple(json.loads(shape_str))
        return np.frombuffer(data_blob, dtype=np.dtype(dtype_str)).reshape(shape)

    def turn_count(self) -> int:
        cur = self._conn.execute(
            "SELECT COUNT(*) FROM turns WHERE session_id=?", (self.session_id,)
        )
        return cur.fetchone()[0]

    def close(self) -> None:
        self._conn.close()
