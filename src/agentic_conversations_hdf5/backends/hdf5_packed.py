"""HDF5 packed-bytes session backend.

Identical external API to HDF5Session but stores variable-length text
(content_text, content_json, args_json, result_text) as flat uint8 byte
buffers rather than VLEN string datasets.

VLEN strings live in the HDF5 global heap — the gzip filter never touches
them. uint8 datasets are chunked fixed-width storage that gzip compresses
in full.  The trade: one extra indirection (offset-length index) per message.

Read optimisation: get_recent_context fetches the entire needed byte range
from content_bytes in a SINGLE h5py read, then carves out individual strings
in Python.  VLEN access requires one global-heap dereference per string.

See schema_packed.py for the dtype definitions and layout docs.
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
from ..schema_packed import (
    CALL_INDEX_DTYPE,
    CONTENT_CHUNK_BYTES,
    CONTENT_INDEX_DTYPE,
)

# VLEN string columns kept as-is (short, bounded, no compression payoff).
_MSG_STR_COLS = ("uuid", "parent_uuid", "type", "role", "model")
_TOOL_STR_COLS = ("tool_use_id", "message_uuid", "name", "result_uuid")


def _decode(v: Any) -> str:
    return v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v)


def _to_bytes(s: str) -> bytes:
    return s.encode("utf-8") if s else b""


class HDF5PackedSession(SessionBackend):
    """HDF5 backend that packs variable-length text into compressible uint8 buffers."""

    def __init__(
        self,
        path: str | Path,
        session_id: str = "",
        model: str = "unknown",
        mode: str = "a",
        flush_every: int = 1,
        compression: Optional[str] = DEFAULT_COMPRESSION,
        compression_opts: Optional[int] = DEFAULT_COMPRESSION_OPTS,
        source: str = "agentic-conversations-hdf5-packed",
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

    # ------------------------------------------------------------------
    # Internal setup
    # ------------------------------------------------------------------

    def _ensure_root(self, source: str) -> None:
        if "schema_version" not in self._f.attrs:
            self._f.attrs["schema_version"] = SCHEMA_VERSION
            self._f.attrs["source"] = source
            self._f.attrs["layout"] = "packed"

    def _create_str_ds(self, group: h5py.Group, name: str) -> None:
        kw: dict[str, Any] = dict(shape=(0,), maxshape=(None,), dtype=VLEN_STR,
                                  chunks=(CHUNK_ROWS,))
        if self._compression:
            kw["compression"] = self._compression
            kw["compression_opts"] = self._compression_opts
        group.create_dataset(name, **kw)

    def _create_num_ds(self, group: h5py.Group, name: str, dtype) -> None:
        kw: dict[str, Any] = dict(shape=(0,), maxshape=(None,), dtype=dtype,
                                  chunks=(CHUNK_ROWS,))
        if self._compression:
            kw["compression"] = self._compression
            kw["compression_opts"] = self._compression_opts
            kw["shuffle"] = True
        group.create_dataset(name, **kw)

    def _create_bytes_ds(self, group: h5py.Group, name: str) -> None:
        """Create the flat uint8 byte-buffer dataset."""
        kw: dict[str, Any] = dict(
            shape=(0,), maxshape=(None,), dtype=np.uint8,
            chunks=(CONTENT_CHUNK_BYTES,),
        )
        if self._compression:
            kw["compression"] = self._compression
            kw["compression_opts"] = self._compression_opts
            kw["shuffle"] = True
        group.create_dataset(name, **kw)

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
            for c in _MSG_STR_COLS:
                self._create_str_ds(mg, c)
            self._create_num_ds(mg, "timestamp", np.float64)
            self._create_num_ds(mg, "usage", USAGE_DTYPE)
            self._create_num_ds(mg, "content_index", CONTENT_INDEX_DTYPE)
            self._create_bytes_ds(mg, "content_bytes")
            mg.create_group("embeddings")

        if f"{root}/tool_calls" not in self._f:
            tcg = self._f.create_group(f"{root}/tool_calls")
            for c in _TOOL_STR_COLS:
                self._create_str_ds(tcg, c)
            self._create_num_ds(tcg, "timestamp", np.float64)
            self._create_num_ds(tcg, "is_error", np.uint8)
            self._create_num_ds(tcg, "call_index", CALL_INDEX_DTYPE)
            self._create_bytes_ds(tcg, "call_bytes")
            tcg.create_group("result_data")

        if f"{root}/artifacts" not in self._f:
            self._f.create_group(f"{root}/artifacts")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

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

    def _append_bytes(self, buf_ds: h5py.Dataset, *parts: bytes) -> list[tuple[int, int]]:
        """Append byte strings to a uint8 buffer dataset.

        Returns a list of (offset, length) tuples, one per part.
        """
        base = buf_ds.shape[0]
        total = sum(len(p) for p in parts)
        buf_ds.resize((base + total,))
        out: list[tuple[int, int]] = []
        pos = base
        for part in parts:
            n = len(part)
            if n:
                buf_ds[pos:pos + n] = np.frombuffer(part, dtype=np.uint8)
            out.append((pos, n))
            pos += n
        return out

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

        text_b = _to_bytes(content_text)
        json_b = _to_bytes(content_json)
        (text_off, text_len), (json_off, json_len) = self._append_bytes(
            mg["content_bytes"], text_b, json_b,
        )

        for c in _MSG_STR_COLS:
            mg[c].resize(new_n, axis=0)
        mg["timestamp"].resize(new_n, axis=0)
        mg["usage"].resize(new_n, axis=0)
        mg["content_index"].resize(new_n, axis=0)

        mg["uuid"][n] = uuid
        mg["parent_uuid"][n] = parent_uuid
        mg["type"][n] = type
        mg["role"][n] = role
        mg["model"][n] = model
        mg["timestamp"][n] = timestamp

        idx_row = np.zeros(1, dtype=CONTENT_INDEX_DTYPE)
        idx_row[0]["text_off"] = text_off
        idx_row[0]["text_len"] = text_len
        idx_row[0]["json_off"] = json_off
        idx_row[0]["json_len"] = json_len
        mg["content_index"][n] = idx_row[0]

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

        args_b = _to_bytes(args_json)
        result_b = _to_bytes(result_text)
        (args_off, args_len), (result_off, result_len) = self._append_bytes(
            tcg["call_bytes"], args_b, result_b,
        )

        for c in _TOOL_STR_COLS:
            tcg[c].resize(new_m, axis=0)
        tcg["timestamp"].resize(new_m, axis=0)
        tcg["is_error"].resize(new_m, axis=0)
        tcg["call_index"].resize(new_m, axis=0)

        tcg["tool_use_id"][m] = tool_use_id
        tcg["message_uuid"][m] = message_uuid
        tcg["name"][m] = name
        tcg["result_uuid"][m] = result_uuid
        tcg["timestamp"][m] = timestamp
        tcg["is_error"][m] = 1 if is_error else 0

        idx_row = np.zeros(1, dtype=CALL_INDEX_DTYPE)
        idx_row[0]["args_off"] = args_off
        idx_row[0]["args_len"] = args_len
        idx_row[0]["result_off"] = result_off
        idx_row[0]["result_len"] = result_len
        tcg["call_index"][m] = idx_row[0]

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

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def get_recent_context(self, n: int = 20) -> list[dict[str, Any]]:
        mg = self._mg
        total = mg["uuid"].shape[0]
        if total == 0:
            return []

        start = max(0, total - n)
        uuids = mg["uuid"][start:total]
        roles = mg["role"][start:total]
        timestamps = mg["timestamp"][start:total]
        idx = mg["content_index"][start:total]

        # One contiguous read covering all the text content we need.
        # Messages are appended sequentially so their byte ranges are
        # contiguous in content_bytes.
        byte_start = int(idx["text_off"][0])
        byte_end = int(idx["json_off"][-1]) + int(idx["json_len"][-1])
        chunk = bytes(mg["content_bytes"][byte_start:byte_end])

        results = []
        emb_grp = mg["embeddings"]
        for i in range(len(uuids)):
            tid = _decode(uuids[i])
            row = idx[i]
            t_off = int(row["text_off"]) - byte_start
            t_end = t_off + int(row["text_len"])
            content = chunk[t_off:t_end].decode("utf-8", errors="replace")
            d: dict[str, Any] = {
                "turn_id": tid,
                "role": _decode(roles[i]),
                "content": content,
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
        """Sum token usage — single hyperslab read of the numeric usage dataset."""
        usage = self._mg["usage"][:]
        return {f: int(usage[f].sum()) for f in USAGE_FIELDS}

    def close(self) -> None:
        if self._f.id.valid:
            self._f.flush()
            self._f.close()
