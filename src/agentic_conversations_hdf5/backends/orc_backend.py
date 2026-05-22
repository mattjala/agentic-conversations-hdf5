"""Apache ORC agent session backend (via pyarrow).

ORC is write-once: a stripe is sealed when written, with no in-place row append.
Two write modes expose that constraint:

* mode="batch"   — buffer all turns in memory, write the ORC file(s) once at
                   close().
* mode="rewrite" — rewrite the whole file on every add_turn/add_tool_call; live
                   append at O(N^2) total cost.

Layout (a directory, like the JSON backend)
-------------------------------------------
<base>/<session_id>/
    messages.orc      columnar table (uuid, role, type, model, timestamp,
                      content_text, content_json, token counts, has_embedding)
    tool_calls.orc    columnar table (when tool calls are present)
    embeddings.npy    (N, dim) float32 sidecar, row-aligned with messages
    result_data.npz   per-call array results (call_id -> array)
    artifacts.npz     named artifacts (name -> array)

Dense vectors go in sidecars rather than table columns so the ORC table holds
only the scalar/text log.
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pyarrow as pa
import pyarrow.orc as orc

from .base import SessionBackend
from ..schema import USAGE_FIELDS

_MSG_SCHEMA = pa.schema([
    ("uuid", pa.string()),
    ("parent_uuid", pa.string()),
    ("type", pa.string()),
    ("role", pa.string()),
    ("model", pa.string()),
    ("timestamp", pa.float64()),
    ("content_text", pa.string()),
    ("content_json", pa.string()),
    ("input_tokens", pa.int64()),
    ("output_tokens", pa.int64()),
    ("cache_creation_input_tokens", pa.int64()),
    ("cache_read_input_tokens", pa.int64()),
    ("has_embedding", pa.bool_()),
])

_TOOL_SCHEMA = pa.schema([
    ("tool_use_id", pa.string()),
    ("message_uuid", pa.string()),
    ("name", pa.string()),
    ("result_uuid", pa.string()),
    ("timestamp", pa.float64()),
    ("is_error", pa.bool_()),
    ("args_json", pa.string()),
    ("result_text", pa.string()),
])


class ORCSession(SessionBackend):
    def __init__(
        self,
        base_dir: str | Path,
        session_id: str = "",
        mode: str = "batch",
    ):
        if mode not in ("batch", "rewrite"):
            raise ValueError(f"mode must be 'batch' or 'rewrite', got {mode!r}")
        self.base_dir = Path(base_dir)
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self._mode = mode
        self._dir = self.base_dir / self.session_id
        self._dir.mkdir(parents=True, exist_ok=True)

        self._msg_path = self._dir / "messages.orc"
        self._tool_path = self._dir / "tool_calls.orc"
        self._emb_path = self._dir / "embeddings.npy"

        # In-memory write buffers (column-oriented).
        self._msg: dict[str, list] = {f.name: [] for f in _MSG_SCHEMA}
        self._emb: list[Optional[np.ndarray]] = []
        self._tool: dict[str, list] = {f.name: [] for f in _TOOL_SCHEMA}
        self._result_data: dict[str, np.ndarray] = {}
        self._artifacts: dict[str, np.ndarray] = {}

        # Count rows already on disk (for a cold-opened, read-only session).
        self._disk_rows = orc.ORCFile(str(self._msg_path)).nrows \
            if self._msg_path.exists() else 0

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def add_turn(self, role, content, embedding=None) -> str:
        turn_id = str(uuid.uuid4())[:12]
        m = self._msg
        m["uuid"].append(turn_id)
        m["parent_uuid"].append("")
        m["type"].append(role)
        m["role"].append(role)
        m["model"].append("")
        m["timestamp"].append(time.time())
        m["content_text"].append(content)
        m["content_json"].append("")
        for f in USAGE_FIELDS:
            m[f].append(0)
        m["has_embedding"].append(embedding is not None)
        self._emb.append(None if embedding is None
                         else np.asarray(embedding, dtype=np.float32).ravel())
        if self._mode == "rewrite":
            self._write_messages()
        return turn_id

    def add_tool_call(self, name, args, result_text=None, result_data=None) -> str:
        import json
        call_id = str(uuid.uuid4())[:12]
        t = self._tool
        t["tool_use_id"].append(call_id)
        t["message_uuid"].append("")
        t["name"].append(name)
        t["result_uuid"].append("")
        t["timestamp"].append(time.time())
        t["is_error"].append(False)
        t["args_json"].append(json.dumps(args))
        t["result_text"].append(result_text or "")
        if result_data is not None:
            self._result_data[call_id] = np.asarray(result_data)
        if self._mode == "rewrite":
            self._write_tools()
        return call_id

    def _write_messages(self) -> None:
        table = pa.table(
            {k: pa.array(v, type=_MSG_SCHEMA.field(k).type) for k, v in self._msg.items()},
            schema=_MSG_SCHEMA,
        )
        orc.write_table(table, str(self._msg_path))
        self._write_embeddings()

    def _write_embeddings(self) -> None:
        if not any(e is not None for e in self._emb):
            return
        dim = next(e.shape[0] for e in self._emb if e is not None)
        arr = np.zeros((len(self._emb), dim), dtype=np.float32)
        for i, e in enumerate(self._emb):
            if e is not None:
                arr[i] = e
        np.save(str(self._emb_path), arr)

    def _write_tools(self) -> None:
        if not self._tool["tool_use_id"]:
            return
        table = pa.table(
            {k: pa.array(v, type=_TOOL_SCHEMA.field(k).type) for k, v in self._tool.items()},
            schema=_TOOL_SCHEMA,
        )
        orc.write_table(table, str(self._tool_path))
        if self._result_data:
            np.savez(str(self._dir / "result_data.npz"), **self._result_data)

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def get_recent_context(self, n: int = 20) -> list[dict[str, Any]]:
        # Serve from the in-memory buffer if this session is still writing
        # (batch mode before close); otherwise read the sealed ORC file.
        if self._msg["uuid"]:
            total = len(self._msg["uuid"])
            start = max(0, total - n)
            cols = {k: self._msg[k][start:total] for k in
                    ("uuid", "role", "content_text", "timestamp", "has_embedding")}
            emb_src = self._emb[start:total]
            return self._rows_from(cols, emb_src)

        if not self._msg_path.exists():
            return []
        table = orc.read_table(str(self._msg_path),
                               columns=["uuid", "role", "content_text",
                                        "timestamp", "has_embedding"])
        total = table.num_rows
        start = max(0, total - n)
        sub = table.slice(start, total - start)
        cols = {k: sub.column(k).to_pylist() for k in
                ("uuid", "role", "content_text", "timestamp", "has_embedding")}
        emb_src = None
        if any(cols["has_embedding"]) and self._emb_path.exists():
            emb_all = np.load(str(self._emb_path), mmap_mode="r")
            emb_src = emb_all[start:total]
        return self._rows_from(cols, emb_src)

    @staticmethod
    def _rows_from(cols, emb_src) -> list[dict[str, Any]]:
        out = []
        for i in range(len(cols["uuid"])):
            d: dict[str, Any] = {
                "turn_id": cols["uuid"][i],
                "role": cols["role"][i],
                "content": cols["content_text"][i],
                "timestamp": float(cols["timestamp"][i]),
            }
            if emb_src is not None and cols["has_embedding"][i]:
                d["embedding"] = np.asarray(emb_src[i])
            out.append(d)
        return out

    def store_artifact(self, name: str, data: np.ndarray) -> None:
        self._artifacts[name] = np.asarray(data)
        np.savez(str(self._dir / "artifacts.npz"), **self._artifacts)

    def get_artifact(self, name: str) -> Optional[np.ndarray]:
        p = self._dir / "artifacts.npz"
        if name in self._artifacts:
            return self._artifacts[name]
        if not p.exists():
            return None
        with np.load(str(p)) as z:
            return z[name] if name in z.files else None

    def turn_count(self) -> int:
        return self._disk_rows + len(self._msg["uuid"])

    def total_usage(self) -> dict[str, int]:
        if self._msg["uuid"]:
            return {f: int(sum(self._msg[f])) for f in USAGE_FIELDS}
        if not self._msg_path.exists():
            return {f: 0 for f in USAGE_FIELDS}
        table = orc.read_table(str(self._msg_path), columns=list(USAGE_FIELDS))
        return {f: int(table.column(f).to_pandas().sum()) for f in USAGE_FIELDS}

    def close(self) -> None:
        # batch mode: write everything once now. rewrite mode: files already
        # current, but flush the final state to be safe.
        if self._msg["uuid"]:
            self._write_messages()
        if self._tool["tool_use_id"]:
            self._write_tools()
