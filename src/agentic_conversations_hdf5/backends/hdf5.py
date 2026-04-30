"""HDF5 agent session backend (schema v1).

See docs/schema.md for the full schema. Quick reference:

    /                                attrs: schema_version, source
    /sessions/{session_id}/          attrs: model, created_at, summary,
                                            cwd, git_branch, agent_version
        /messages/                   parallel-array dataset group
            uuid          VLEN str   (N,)
            parent_uuid   VLEN str   (N,)
            type          VLEN str   (N,)   "user"|"assistant"|...
            role          VLEN str   (N,)
            timestamp     float64    (N,)
            content_text  VLEN str   (N,)   plain text view (best-effort)
            content_json  VLEN str   (N,)   full content blocks JSON
            model         VLEN str   (N,)
            usage         compound   (N,)   per-message token counts
            /embeddings/{uuid}       optional float32 (D,)

        /tool_calls/                 parallel-array dataset group
            tool_use_id   VLEN str   (M,)
            message_uuid  VLEN str   (M,)   parent assistant message
            name          VLEN str   (M,)
            args_json     VLEN str   (M,)
            result_text   VLEN str   (M,)
            result_uuid   VLEN str   (M,)   tool_result message UUID
            timestamp     float64   (M,)
            is_error      uint8     (M,)
            /result_data/{tool_use_id}    optional ndarray

        /artifacts/{name}            arbitrary HDF5-storable arrays
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np

from .base import SessionBackend
from ..schema import (
    CHUNK_ROWS,
    DEFAULT_COMPRESSION,
    DEFAULT_COMPRESSION_OPTS,
    SCHEMA_VERSION,
    USAGE_DTYPE,
    USAGE_FIELDS,
    VLEN_STR,
)

_STR_COLS = ("uuid", "parent_uuid", "type", "role",
             "content_text", "content_json", "model")
_TOOL_STR_COLS = ("tool_use_id", "message_uuid", "name",
                  "args_json", "result_text", "result_uuid")


def _decode(v):
    return v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else v


class HDF5Session(SessionBackend):
    def __init__(
        self,
        path: str | Path,
        session_id: str = "",
        model: str = "unknown",
        mode: str = "a",
        flush_every: int = 1,
        compression: Optional[str] = DEFAULT_COMPRESSION,
        compression_opts: Optional[int] = DEFAULT_COMPRESSION_OPTS,
        source: str = "agentic-conversations-hdf5",
    ):
        self.path = Path(path)
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self._flush_every = flush_every
        self._write_count = 0
        self._compression = compression
        self._compression_opts = compression_opts
        self._f = h5py.File(self.path, mode)
        self._ensure_root(source)
        self._ensure_session(model)

    def _ensure_root(self, source: str) -> None:
        if "schema_version" not in self._f.attrs:
            self._f.attrs["schema_version"] = SCHEMA_VERSION
            self._f.attrs["source"] = source

    def _create_str_ds(self, group: h5py.Group, name: str) -> None:
        kwargs: dict[str, Any] = dict(
            shape=(0,), maxshape=(None,), dtype=VLEN_STR,
            chunks=(CHUNK_ROWS,),
        )
        if self._compression:
            kwargs["compression"] = self._compression
            kwargs["compression_opts"] = self._compression_opts
        group.create_dataset(name, **kwargs)

    def _create_num_ds(self, group: h5py.Group, name: str, dtype) -> None:
        kwargs: dict[str, Any] = dict(
            shape=(0,), maxshape=(None,), dtype=dtype,
            chunks=(CHUNK_ROWS,),
        )
        if self._compression:
            kwargs["compression"] = self._compression
            kwargs["compression_opts"] = self._compression_opts
            kwargs["shuffle"] = True
        group.create_dataset(name, **kwargs)

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

        if f"{root}/messages" not in self._f:
            mg = self._f.create_group(f"{root}/messages")
            for c in _STR_COLS:
                self._create_str_ds(mg, c)
            self._create_num_ds(mg, "timestamp", np.float64)
            self._create_num_ds(mg, "usage", USAGE_DTYPE)
            mg.create_group("embeddings")

        if f"{root}/tool_calls" not in self._f:
            tcg = self._f.create_group(f"{root}/tool_calls")
            for c in _TOOL_STR_COLS:
                self._create_str_ds(tcg, c)
            self._create_num_ds(tcg, "timestamp", np.float64)
            self._create_num_ds(tcg, "is_error", np.uint8)
            tcg.create_group("result_data")

        if f"{root}/artifacts" not in self._f:
            self._f.create_group(f"{root}/artifacts")

    @property
    def _root(self) -> str:
        return f"sessions/{self.session_id}"

    @property
    def _mg(self) -> h5py.Group:
        return self._f[f"{self._root}/messages"]

    @property
    def _tcg(self) -> h5py.Group:
        return self._f[f"{self._root}/tool_calls"]

    def _maybe_flush(self) -> None:
        self._write_count += 1
        if self._flush_every == 0:
            return
        if self._write_count % self._flush_every == 0:
            self._f.flush()

    def set_session_attrs(self, **attrs: Any) -> None:
        sg = self._f[self._root]
        for k, v in attrs.items():
            if v is None:
                continue
            sg.attrs[k] = v

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
        mg = self._mg
        n = mg["uuid"].shape[0]
        new_n = n + 1

        for c in _STR_COLS:
            mg[c].resize(new_n, axis=0)
        mg["timestamp"].resize(new_n, axis=0)
        mg["usage"].resize(new_n, axis=0)

        mg["uuid"][n] = uuid
        mg["parent_uuid"][n] = parent_uuid
        mg["type"][n] = type
        mg["role"][n] = role
        mg["content_text"][n] = content_text
        mg["content_json"][n] = content_json
        mg["model"][n] = model
        mg["timestamp"][n] = timestamp

        usage_row = np.zeros(1, dtype=USAGE_DTYPE)
        if usage:
            for k in USAGE_FIELDS:
                if k in usage and usage[k] is not None:
                    usage_row[0][k] = int(usage[k])
        mg["usage"][n] = usage_row[0]

        if embedding is not None:
            mg["embeddings"].create_dataset(
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
        tcg = self._tcg
        m = tcg["tool_use_id"].shape[0]
        new_m = m + 1

        for c in _TOOL_STR_COLS:
            tcg[c].resize(new_m, axis=0)
        tcg["timestamp"].resize(new_m, axis=0)
        tcg["is_error"].resize(new_m, axis=0)

        tcg["tool_use_id"][m] = tool_use_id
        tcg["message_uuid"][m] = message_uuid
        tcg["name"][m] = name
        tcg["args_json"][m] = args_json
        tcg["result_text"][m] = result_text
        tcg["result_uuid"][m] = result_uuid
        tcg["timestamp"][m] = timestamp
        tcg["is_error"][m] = 1 if is_error else 0

        if result_data is not None and tool_use_id:
            tcg["result_data"].create_dataset(tool_use_id, data=result_data)

        self._maybe_flush()

    def add_turn(
        self,
        role: str,
        content: str,
        embedding: Optional[np.ndarray] = None,
    ) -> str:
        turn_id = str(uuid.uuid4())[:12]
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
        call_id = str(uuid.uuid4())[:12]
        self.append_tool_call(
            tool_use_id=call_id,
            name=name,
            args_json=json.dumps(args),
            result_text=result_text or "",
            timestamp=time.time(),
            result_data=result_data,
        )
        return call_id

    def get_recent_context(self, n: int = 20) -> list[dict[str, Any]]:
        mg = self._mg
        total = mg["uuid"].shape[0]
        if total == 0:
            return []

        start = max(0, total - n)
        uuids = mg["uuid"][start:total]
        roles = mg["role"][start:total]
        contents = mg["content_text"][start:total]
        timestamps = mg["timestamp"][start:total]

        results = []
        emb_grp = mg["embeddings"]
        for i in range(len(uuids)):
            tid = _decode(uuids[i])
            d: dict[str, Any] = {
                "turn_id": tid,
                "role": _decode(roles[i]),
                "content": _decode(contents[i]),
                "timestamp": float(timestamps[i]),
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
        return int(self._mg["uuid"].shape[0])

    def total_usage(self) -> dict[str, int]:
        """Sum token usage across all messages — single hyperslab read."""
        usage = self._mg["usage"][:]
        return {f: int(usage[f].sum()) for f in USAGE_FIELDS}

    def close(self) -> None:
        if self._f.id.valid:
            self._f.flush()
            self._f.close()
