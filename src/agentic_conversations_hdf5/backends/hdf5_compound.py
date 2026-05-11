"""HDF5 compound-dataset session backend.

Each message group is a single compound dataset (one row per message) rather
than nine parallel 1-D arrays.  Tool calls are similarly one compound dataset.

Layout
------
    /sessions/{session_id}/
        messages        compound (N,)  dtype=MSG_DTYPE
        tool_calls      compound (M,)  dtype=TOOL_DTYPE
        embeddings/     group — per-uuid float32 datasets (unchanged)
        result_data/    group — per-tool_use_id array datasets (unchanged)
        artifacts/      group — named array datasets (unchanged)

Fixed-length string fields (uuid, type, role, model, …) are stored inline in
the chunk, so gzip compresses them.  content_text and content_json are VLEN —
they go to the global heap and are not compressed (same limitation as the
original parallel-array layout).

Access pattern trade-offs
--------------------------
* get_recent_context(N): one compound slice vs. 7 separate dataset reads in
  the parallel layouts — compound wins.
* total_usage(): field selection on a compound reads all chunk bytes then
  discards non-selected fields, vs. one tight read of the numeric-only
  usage dataset in the parallel layouts — parallel wins.
"""
from __future__ import annotations

import json
import time
import uuid as _uuid_mod
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np

from .base import SessionBackend
from ..schema import (
    DEFAULT_COMPRESSION,
    DEFAULT_COMPRESSION_OPTS,
    SCHEMA_VERSION,
)
from ..schema_compound import (
    CHUNK_ROWS,
    MSG_DTYPE,
    TOOL_DTYPE,
    USAGE_FIELDS,
)


def _decode(v: Any) -> str:
    """Decode a field value returned by h5py from a compound dataset.

    Fixed-length string fields come back as numpy bytes_ (null-padded);
    VLEN string fields come back as Python str or bytes.
    """
    if isinstance(v, (bytes, np.bytes_)):
        return v.rstrip(b"\x00").decode("utf-8", errors="replace")
    if v is None:
        return ""
    return str(v)


class HDF5CompoundSession(SessionBackend):
    """HDF5 backend that stores messages and tool-calls as compound datasets."""

    def __init__(
        self,
        path: str | Path,
        session_id: str = "",
        model: str = "unknown",
        mode: str = "a",
        flush_every: int = 1,
        compression: Optional[str] = DEFAULT_COMPRESSION,
        compression_opts: Optional[int] = DEFAULT_COMPRESSION_OPTS,
        source: str = "agentic-conversations-hdf5-compound",
    ):
        self.path = Path(path)
        self.session_id = session_id or str(_uuid_mod.uuid4())[:8]
        self._flush_every = flush_every
        self._write_count = 0
        self._compression = compression
        self._compression_opts = compression_opts
        self._f = h5py.File(self.path, mode)
        self._ensure_root(source)
        self._ensure_session(model)

    # ------------------------------------------------------------------
    # Internal setup
    # ------------------------------------------------------------------

    def _ensure_root(self, source: str) -> None:
        if "schema_version" not in self._f.attrs:
            self._f.attrs["schema_version"] = SCHEMA_VERSION
            self._f.attrs["source"] = source
            self._f.attrs["layout"] = "compound"

    def _make_compound_ds(
        self, group: h5py.Group, name: str, dtype: np.dtype
    ) -> h5py.Dataset:
        kw: dict[str, Any] = dict(
            shape=(0,),
            maxshape=(None,),
            dtype=dtype,
            chunks=(CHUNK_ROWS,),
        )
        if self._compression:
            kw["compression"] = self._compression
            kw["compression_opts"] = self._compression_opts
        return group.create_dataset(name, **kw)

    def _ensure_session(self, model: str) -> None:
        root = f"sessions/{self.session_id}"
        if root not in self._f:
            sg = self._f.create_group(root)
            sg.attrs["model"] = model
            sg.attrs["created_at"] = time.time()
            sg.attrs["summary"] = ""
            sg.attrs["cwd"] = ""
            sg.attrs["git_branch"] = ""
            sg.attrs["agent_version"] = ""

        sg = self._f[root]

        if "messages" not in sg:
            self._make_compound_ds(sg, "messages", MSG_DTYPE)
            sg.create_group("embeddings")

        if "tool_calls" not in sg:
            self._make_compound_ds(sg, "tool_calls", TOOL_DTYPE)
            sg.create_group("result_data")

        if "artifacts" not in sg:
            sg.create_group("artifacts")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def _root(self) -> str:
        return f"sessions/{self.session_id}"

    @property
    def _msg_ds(self) -> h5py.Dataset:
        return self._f[f"{self._root}/messages"]

    @property
    def _tool_ds(self) -> h5py.Dataset:
        return self._f[f"{self._root}/tool_calls"]

    @property
    def _emb_grp(self) -> h5py.Group:
        return self._f[f"{self._root}/embeddings"]

    def _maybe_flush(self) -> None:
        self._write_count += 1
        if self._flush_every == 0:
            return
        if self._write_count % self._flush_every == 0:
            self._f.flush()

    # ------------------------------------------------------------------
    # Session metadata
    # ------------------------------------------------------------------

    def set_session_attrs(self, **attrs: Any) -> None:
        sg = self._f[self._root]
        for k, v in attrs.items():
            if v is None:
                continue
            sg.attrs[k] = v

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def append_message(
        self,
        *,
        uuid: str,
        parent_uuid: str = "",
        type: str = "user",
        role: str = "",
        timestamp: float,
        content_text: str = "",
        content_json: str = "",
        model: str = "",
        usage: Optional[dict[str, int]] = None,
        embedding: Optional[np.ndarray] = None,
    ) -> None:
        ds = self._msg_ds
        n = ds.shape[0]
        ds.resize((n + 1,))

        row = np.empty(1, dtype=MSG_DTYPE)
        row["uuid"]         = uuid
        row["parent_uuid"]  = parent_uuid
        row["type"]         = type
        row["role"]         = role
        row["model"]        = model
        row["timestamp"]    = timestamp
        row["content_text"] = content_text
        row["content_json"] = content_json

        u = usage or {}
        row["input_tokens"]                = int(u.get("input_tokens", 0) or 0)
        row["output_tokens"]               = int(u.get("output_tokens", 0) or 0)
        row["cache_creation_input_tokens"] = int(u.get("cache_creation_input_tokens", 0) or 0)
        row["cache_read_input_tokens"]     = int(u.get("cache_read_input_tokens", 0) or 0)

        ds[n] = row[0]

        if embedding is not None:
            self._emb_grp.create_dataset(
                uuid, data=np.asarray(embedding, dtype=np.float32),
            )

        self._maybe_flush()

    def append_tool_call(
        self,
        *,
        tool_use_id: str,
        message_uuid: str = "",
        name: str = "",
        args_json: str = "",
        result_text: str = "",
        result_uuid: str = "",
        timestamp: float = 0.0,
        is_error: bool = False,
        result_data: Optional[np.ndarray] = None,
    ) -> None:
        ds = self._tool_ds
        m = ds.shape[0]
        ds.resize((m + 1,))

        row = np.empty(1, dtype=TOOL_DTYPE)
        row["tool_use_id"]  = tool_use_id
        row["message_uuid"] = message_uuid
        row["name"]         = name
        row["result_uuid"]  = result_uuid
        row["timestamp"]    = timestamp
        row["is_error"]     = 1 if is_error else 0
        row["args_json"]    = args_json
        row["result_text"]  = result_text

        ds[m] = row[0]

        if result_data is not None and tool_use_id:
            self._f[f"{self._root}/result_data"].create_dataset(
                tool_use_id, data=result_data,
            )

        self._maybe_flush()

    def add_turn(
        self,
        role: str,
        content: str,
        embedding: Optional[np.ndarray] = None,
    ) -> str:
        turn_id = str(_uuid_mod.uuid4())[:12]
        self.append_message(
            uuid=turn_id,
            parent_uuid="",
            type=role,
            role=role,
            timestamp=time.time(),
            content_text=content,
            embedding=embedding,
        )
        return turn_id

    def add_tool_call(
        self,
        name: str,
        args: dict[str, Any],
        result_text: Optional[str] = None,
        result_data: Optional[np.ndarray] = None,
    ) -> str:
        call_id = str(_uuid_mod.uuid4())[:12]
        self.append_tool_call(
            tool_use_id=call_id,
            name=name,
            args_json=json.dumps(args),
            result_text=result_text or "",
            timestamp=time.time(),
            result_data=result_data,
        )
        return call_id

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def get_recent_context(self, n: int = 20) -> list[dict[str, Any]]:
        ds = self._msg_ds
        total = ds.shape[0]
        if total == 0:
            return []

        start = max(0, total - n)
        # One compound slice: all fields for the relevant rows in one read.
        rows = ds[start:total]

        results = []
        emb_grp = self._emb_grp
        for row in rows:
            tid = _decode(row["uuid"])
            d: dict[str, Any] = {
                "turn_id":   tid,
                "role":      _decode(row["role"]),
                "content":   _decode(row["content_text"]),
                "timestamp": float(row["timestamp"]),
            }
            if tid in emb_grp:
                d["embedding"] = emb_grp[tid][:]
            results.append(d)
        return results

    def store_artifact(self, name: str, data: np.ndarray) -> None:
        ag = self._f[f"{self._root}/artifacts"]
        if name in ag:
            del ag[name]
        ag.create_dataset(name, data=data)
        self._maybe_flush()

    def get_artifact(self, name: str) -> Optional[np.ndarray]:
        ag = self._f[f"{self._root}/artifacts"]
        if name not in ag:
            return None
        return ag[name][:]

    def turn_count(self) -> int:
        return int(self._msg_ds.shape[0])

    def total_usage(self) -> dict[str, int]:
        """Sum token usage across all messages.

        Uses h5py field selection to read only the four integer fields from
        the compound dataset, avoiding deserialization of the VLEN content
        fields.  HDF5 still reads full chunks then discards non-selected
        fields, so this is less efficient than the parallel-array layout's
        standalone usage dataset — but correct.
        """
        ds = self._msg_ds
        n = ds.shape[0]
        if n == 0:
            return {f: 0 for f in USAGE_FIELDS}
        # ds.fields(name)[:] reads only that field from the compound,
        # skipping deserialization of the VLEN content fields.
        return {f: int(ds.fields(f)[:].sum()) for f in USAGE_FIELDS}

    def close(self) -> None:
        if self._f.id.valid:
            self._f.flush()
            self._f.close()
