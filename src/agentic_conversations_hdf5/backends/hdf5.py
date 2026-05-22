"""HDF5 session backend.

Variable-length text (message content, tool args/results) is stored in flat
uint8 byte buffers plus a compact offset/length index per row, so gzip
compresses the text in full. Fixed-width metadata (uuid, role, timestamps,
token usage) lives in parallel typed datasets.

get_recent_context reads the whole needed byte range from the content buffer
in a single read, then slices out individual strings in Python.

See schema.py for the dtype definitions and layout.
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
    CALL_INDEX_DTYPE,
    CHUNK_ROWS,
    CONTENT_CHUNK_BYTES,
    CONTENT_INDEX_DTYPE,
    DEFAULT_COMPRESSION,
    DEFAULT_COMPRESSION_OPTS,
    SCHEMA_VERSION,
    USAGE_DTYPE,
    USAGE_FIELDS,
    VLEN_STR,
)

_INITIAL_CAPACITY = 256
_GROWTH_FACTOR = 2

# VLEN string columns kept as-is (short, bounded, no compression payoff).
_MSG_STR_COLS = ("uuid", "parent_uuid", "type", "role", "model")
_TOOL_STR_COLS = ("tool_use_id", "message_uuid", "name", "result_uuid")


def _decode(v: Any) -> str:
    return v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v)


def _to_bytes(s: str) -> bytes:
    return s.encode("utf-8") if s else b""


class HDF5Session(SessionBackend):
    """HDF5 conversation-log backend."""

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
        core_vfd: bool = False,
        chunk_rows: Optional[int] = None,
        content_chunk_bytes: Optional[int] = None,
        batch_size: int = 1,
    ):
        self.path = Path(path)
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self._flush_every = flush_every
        self._write_count = 0
        # Accumulate up to batch_size messages, then write the block with one
        # assign per dataset. batch_size=1 flushes every message.
        self._batch_size = max(1, batch_size)
        self._msg_buf: list[tuple] = []
        self._compression = compression
        self._compression_opts = compression_opts
        self._core_vfd = core_vfd
        self._chunk_rows = chunk_rows if chunk_rows is not None else CHUNK_ROWS
        self._content_chunk_bytes = (
            content_chunk_bytes if content_chunk_bytes is not None
            else CONTENT_CHUNK_BYTES
        )
        open_kwargs: dict[str, Any] = dict(libver="latest")
        if core_vfd:
            # All I/O in RAM; backing_store writes on close().
            # flush() with Core VFD rewrites the entire image, so callers
            # should set flush_every=0 (flush only on close) when using this.
            open_kwargs["driver"] = "core"
            open_kwargs["backing_store"] = True
        self._f = h5py.File(self.path, mode, **open_kwargs)
        self._ensure_root(source)
        self._ensure_session(model)
        # Logical row counts (separate from dataset capacity).
        mg = self._f[f"sessions/{self.session_id}/messages"]
        tcg = self._f[f"sessions/{self.session_id}/tool_calls"]
        self._msg_n: int = int(mg.attrs.get("n_used", mg["uuid"].shape[0]))
        self._tool_n: int = int(tcg.attrs.get("n_used", tcg["tool_use_id"].shape[0]))

    # ------------------------------------------------------------------
    # Internal setup
    # ------------------------------------------------------------------

    def _ensure_root(self, source: str) -> None:
        if "schema_version" not in self._f.attrs:
            self._f.attrs["schema_version"] = SCHEMA_VERSION
            self._f.attrs["source"] = source

    def _create_str_ds(self, group: h5py.Group, name: str, capacity: int = 0) -> None:
        kw: dict[str, Any] = dict(shape=(capacity,), maxshape=(None,), dtype=VLEN_STR,
                                  chunks=(self._chunk_rows,))
        if self._compression:
            kw["compression"] = self._compression
            kw["compression_opts"] = self._compression_opts
        group.create_dataset(name, **kw)

    def _create_num_ds(self, group: h5py.Group, name: str, dtype, capacity: int = 0) -> None:
        kw: dict[str, Any] = dict(shape=(capacity,), maxshape=(None,), dtype=dtype,
                                  chunks=(self._chunk_rows,))
        if self._compression:
            kw["compression"] = self._compression
            kw["compression_opts"] = self._compression_opts
            kw["shuffle"] = True
        group.create_dataset(name, **kw)

    def _create_bytes_ds(self, group: h5py.Group, name: str) -> None:
        """Create the flat uint8 byte-buffer dataset."""
        kw: dict[str, Any] = dict(
            shape=(0,), maxshape=(None,), dtype=np.uint8,
            chunks=(self._content_chunk_bytes,),
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
                self._create_str_ds(mg, c, capacity=_INITIAL_CAPACITY)
            self._create_num_ds(mg, "timestamp", np.float64, capacity=_INITIAL_CAPACITY)
            self._create_num_ds(mg, "usage", USAGE_DTYPE, capacity=_INITIAL_CAPACITY)
            self._create_num_ds(mg, "content_index", CONTENT_INDEX_DTYPE, capacity=_INITIAL_CAPACITY)
            self._create_num_ds(mg, "has_embedding", np.uint8, capacity=_INITIAL_CAPACITY)
            self._create_bytes_ds(mg, "content_bytes")
            mg.attrs["n_used"] = 0
            # Embeddings live in a single (N, dim) dataset, created lazily on the
            # first embedding (dim is unknown until then); has_embedding marks
            # which rows carry one.

        if f"{root}/tool_calls" not in self._f:
            tcg = self._f.create_group(f"{root}/tool_calls")
            for c in _TOOL_STR_COLS:
                self._create_str_ds(tcg, c, capacity=_INITIAL_CAPACITY)
            self._create_num_ds(tcg, "timestamp", np.float64, capacity=_INITIAL_CAPACITY)
            self._create_num_ds(tcg, "is_error", np.uint8, capacity=_INITIAL_CAPACITY)
            self._create_num_ds(tcg, "call_index", CALL_INDEX_DTYPE, capacity=_INITIAL_CAPACITY)
            self._create_bytes_ds(tcg, "call_bytes")
            tcg.attrs["n_used"] = 0
            tcg.create_group("result_data")

        if f"{root}/artifacts" not in self._f:
            self._f.create_group(f"{root}/artifacts")

    def _grow_group(self, group: h5py.Group, row_datasets: tuple[str, ...]) -> None:
        """Double the capacity of all row datasets in a group."""
        cur = group[row_datasets[0]].shape[0]
        new_cap = max(cur * _GROWTH_FACTOR, _INITIAL_CAPACITY)
        for name in row_datasets:
            group[name].resize((int(new_cap),))

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

    _MSG_ROW_DS = (*_MSG_STR_COLS, "timestamp", "usage", "content_index", "has_embedding")
    _TOOL_ROW_DS = (*_TOOL_STR_COLS, "timestamp", "is_error", "call_index")

    def _ensure_emb_ds(self, dim: int) -> h5py.Dataset:
        """Lazily create the (N, dim) embeddings dataset, zero-filling existing
        rows so row indices stay aligned with the message rows.

        Uncompressed: dense float32 vectors compress poorly, so gzip+shuffle
        would cost more per write than it saves.
        """
        mg = self._mg
        if "embeddings" in mg:
            return mg["embeddings"]
        # Pre-allocate to the message-row capacity so embeddings grow by doubling
        # in lockstep with the other columns; rows past msg_n are zero fill.
        cap = max(mg["uuid"].shape[0], self._msg_n)
        return mg.create_dataset(
            "embeddings",
            shape=(cap, dim), maxshape=(None, dim), dtype=np.float32,
            chunks=(self._chunk_rows, dim),
        )

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
        self._msg_buf.append((
            uuid, parent_uuid, type, role, model, timestamp,
            _to_bytes(content_text), _to_bytes(content_json), usage, embedding,
        ))
        if len(self._msg_buf) >= self._batch_size:
            self._flush_msg_buffer()
        self._maybe_flush()

    def _flush_msg_buffer(self) -> None:
        """Write all buffered messages with one block assign per dataset."""
        buf = self._msg_buf
        if not buf:
            return
        mg = self._mg
        b = len(buf)
        n = self._msg_n

        # Grow capacity once to fit the whole block.
        while n + b > mg["uuid"].shape[0]:
            self._grow_group(mg, self._MSG_ROW_DS)

        uuids: list[str] = []
        parents: list[str] = []
        types: list[str] = []
        roles: list[str] = []
        models: list[str] = []
        ts_arr = np.zeros(b, dtype=np.float64)
        idx_arr = np.zeros(b, dtype=CONTENT_INDEX_DTYPE)
        usage_arr = np.zeros(b, dtype=USAGE_DTYPE)

        base = mg["content_bytes"].shape[0]
        blob_parts: list[bytes] = []
        pos = base
        for i, (u, pu, ty, ro, mo, ts, tb, jb, us, _emb) in enumerate(buf):
            uuids.append(u)
            parents.append(pu)
            types.append(ty)
            roles.append(ro)
            models.append(mo)
            ts_arr[i] = ts
            tl, jl = len(tb), len(jb)
            idx_arr[i]["text_off"] = pos
            idx_arr[i]["text_len"] = tl
            idx_arr[i]["json_off"] = pos + tl
            idx_arr[i]["json_len"] = jl
            if tb:
                blob_parts.append(tb)
            if jb:
                blob_parts.append(jb)
            pos += tl + jl
            if us:
                for k in USAGE_FIELDS:
                    if k in us and us[k] is not None:
                        usage_arr[i][k] = int(us[k])

        total = pos - base
        if total:
            mg["content_bytes"].resize((base + total,))
            mg["content_bytes"][base:base + total] = np.frombuffer(
                b"".join(blob_parts), dtype=np.uint8,
            )

        # Embeddings: collect present rows, write the whole batch as one block.
        has_emb = np.zeros(b, dtype=np.uint8)
        emb_rows: list[tuple[int, np.ndarray]] = []
        for i, (_u, _pu, _ty, _ro, _mo, _ts, _tb, _jb, _us, emb) in enumerate(buf):
            if emb is not None:
                has_emb[i] = 1
                emb_rows.append((i, np.asarray(emb, dtype=np.float32).ravel()))

        mg["uuid"][n:n + b] = np.array(uuids, dtype=object)
        mg["parent_uuid"][n:n + b] = np.array(parents, dtype=object)
        mg["type"][n:n + b] = np.array(types, dtype=object)
        mg["role"][n:n + b] = np.array(roles, dtype=object)
        mg["model"][n:n + b] = np.array(models, dtype=object)
        mg["timestamp"][n:n + b] = ts_arr
        mg["content_index"][n:n + b] = idx_arr
        mg["usage"][n:n + b] = usage_arr
        mg["has_embedding"][n:n + b] = has_emb

        if emb_rows or "embeddings" in mg:
            dim = emb_rows[0][1].shape[0] if emb_rows else mg["embeddings"].shape[1]
            emb_ds = self._ensure_emb_ds(dim)
            cap = mg["uuid"].shape[0]  # already grown above to fit n + b
            if emb_ds.shape[0] < cap:
                emb_ds.resize((cap, dim))
            block = np.zeros((b, dim), dtype=np.float32)
            for li, vec in emb_rows:
                block[li] = vec
            emb_ds[n:n + b] = block

        self._msg_n += b
        self._msg_buf = []

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
        m = self._tool_n

        # Grow if pre-allocated capacity is exhausted.
        if m >= tcg["tool_use_id"].shape[0]:
            self._grow_group(tcg, self._TOOL_ROW_DS)

        args_b = _to_bytes(args_json)
        result_b = _to_bytes(result_text)
        (args_off, args_len), (result_off, result_len) = self._append_bytes(
            tcg["call_bytes"], args_b, result_b,
        )

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

        self._tool_n += 1
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
        self._flush_msg_buffer()
        mg = self._mg
        total = self._msg_n
        if total == 0:
            return []

        start = max(0, total - n)
        uuids = mg["uuid"][start:total]
        roles = mg["role"][start:total]
        timestamps = mg["timestamp"][start:total]
        idx = mg["content_index"][start:total]
        has_emb = mg["has_embedding"][start:total]

        # One contiguous read covering all the text content we need.
        # Messages are appended sequentially so their byte ranges are
        # contiguous in content_bytes.
        byte_start = int(idx["text_off"][0])
        byte_end = int(idx["json_off"][-1]) + int(idx["json_len"][-1])
        chunk = bytes(mg["content_bytes"][byte_start:byte_end])

        # One slice read covers all embeddings in range.
        emb_block = None
        if has_emb.any() and "embeddings" in mg:
            emb_block = mg["embeddings"][start:total]

        results = []
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
            if emb_block is not None and has_emb[i]:
                d["embedding"] = emb_block[i]
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
        return self._msg_n + len(self._msg_buf)

    def total_usage(self) -> dict[str, int]:
        """Sum token usage — single hyperslab read of the numeric usage dataset."""
        self._flush_msg_buffer()
        usage = self._mg["usage"][:self._msg_n]
        return {f: int(usage[f].sum()) for f in USAGE_FIELDS}

    def close(self) -> None:
        if not self._f.id.valid:
            return
        if self._f.mode == "r":  # read-only: nothing to flush or truncate
            self._f.close()
            return
        self._flush_msg_buffer()
        # Truncate pre-allocated capacity down to actual used rows.
        mg = self._mg
        tcg = self._tcg
        for name in self._MSG_ROW_DS:
            mg[name].resize((self._msg_n,))
        if "embeddings" in mg and mg["embeddings"].shape[0] != self._msg_n:
            mg["embeddings"].resize((self._msg_n, mg["embeddings"].shape[1]))
        for name in self._TOOL_ROW_DS:
            tcg[name].resize((self._tool_n,))
        mg.attrs["n_used"] = self._msg_n
        tcg.attrs["n_used"] = self._tool_n
        self._f.flush()
        self._f.close()
