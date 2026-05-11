"""Convert Claude Code JSONL session logs to/from the packed HDF5 format.

The packed format stores variable-length text in flat uint8 byte buffers
rather than VLEN string datasets, making them compressible by gzip.

Functions
---------
convert_jsonl_packed   JSONL → HDF5 (packed layout)
unpack_to_jsonl        HDF5 (packed) → approximate JSONL for inspection/round-trip
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional

from .backends.hdf5_packed import HDF5PackedSession
from .convert import (  # reuse all parsing helpers from the original converter
    _extract_tool_calls,
    _extract_tool_results,
    _flatten_content,
    _iter_jsonl,
    _parse_ts,
)


def convert_jsonl_packed(
    jsonl_path: str | Path,
    h5_path: str | Path,
    session_id: Optional[str] = None,
    overwrite: bool = False,
    flush_every: int = 0,
    compression: Optional[str] = "gzip",
) -> dict[str, int]:
    """Convert a Claude Code JSONL log to a packed HDF5 session file.

    Identical to convert_jsonl() from convert.py but writes using
    HDF5PackedSession (byte-buffer layout) instead of HDF5Session.

    Returns a dict of counts: messages, tool_uses, tool_results, skipped.
    """
    jsonl_path = Path(jsonl_path)
    h5_path = Path(h5_path)
    if h5_path.exists() and overwrite:
        h5_path.unlink()

    sid = session_id or jsonl_path.stem
    counts = {"messages": 0, "tool_uses": 0, "tool_results": 0, "skipped": 0}

    # First pass: pre-collect tool_results (they appear after their tool_use).
    tool_results: dict[str, dict] = {}
    for rec in _iter_jsonl(jsonl_path):
        if rec.get("type") != "user":
            continue
        for tr in _extract_tool_results(rec):
            tool_results[tr["tool_use_id"]] = {
                "result_text": tr["result_text"],
                "result_uuid": rec.get("uuid", ""),
                "is_error": tr["is_error"],
            }
            counts["tool_results"] += 1

    session_attrs: dict[str, Any] = {}

    sess = HDF5PackedSession(
        h5_path,
        session_id=sid,
        mode="w",
        flush_every=flush_every,
        compression=compression,
        source="claude-code",
    )

    try:
        for rec in _iter_jsonl(jsonl_path):
            rtype = rec.get("type")

            if rtype not in {"user", "assistant", "summary"}:
                if rtype == "permission-mode":
                    session_attrs.setdefault(
                        "permission_mode", rec.get("permissionMode", ""))
                counts["skipped"] += 1
                continue

            for k_src, k_dst in (
                ("cwd", "cwd"),
                ("gitBranch", "git_branch"),
                ("version", "agent_version"),
                ("sessionId", "session_id_original"),
            ):
                if k_src in rec and k_dst not in session_attrs:
                    session_attrs[k_dst] = rec[k_src]

            msg = rec.get("message") or {}
            role = msg.get("role", "")
            model = msg.get("model", "")
            content = msg.get("content")
            usage = msg.get("usage") or {}

            content_text = _flatten_content(content)
            content_json = json.dumps(content) if content is not None else ""

            uid = rec.get("uuid", "")
            parent = rec.get("parentUuid") or ""
            ts = _parse_ts(rec.get("timestamp"))

            sess.append_message(
                uuid=uid,
                parent_uuid=parent,
                type=rtype,
                role=role,
                timestamp=ts,
                content_text=content_text,
                content_json=content_json,
                model=model,
                usage={
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "cache_creation_input_tokens": usage.get(
                        "cache_creation_input_tokens", 0),
                    "cache_read_input_tokens": usage.get(
                        "cache_read_input_tokens", 0),
                },
            )
            counts["messages"] += 1

            if rtype == "assistant":
                for tu in _extract_tool_calls(rec):
                    tu_id = tu["tool_use_id"]
                    res = tool_results.pop(tu_id, None)
                    sess.append_tool_call(
                        tool_use_id=tu_id,
                        message_uuid=uid,
                        name=tu["name"],
                        args_json=json.dumps(tu["input"]),
                        result_text=res["result_text"] if res else "",
                        result_uuid=res["result_uuid"] if res else "",
                        timestamp=ts,
                        is_error=bool(res["is_error"]) if res else False,
                    )
                    counts["tool_uses"] += 1

        for tu_id, res in tool_results.items():
            sess.append_tool_call(
                tool_use_id=tu_id,
                message_uuid="",
                name="",
                args_json="",
                result_text=res["result_text"],
                result_uuid=res["result_uuid"],
                is_error=res["is_error"],
            )
            counts["tool_uses"] += 1

        sess.set_session_attrs(**session_attrs)
    finally:
        sess.close()

    return counts


def unpack_to_jsonl(
    h5_path: str | Path,
    out_path: str | Path,
    session_id: Optional[str] = None,
    overwrite: bool = False,
) -> int:
    """Reconstruct approximate JSONL from a packed HDF5 session file.

    This is a best-effort round-trip for inspection and verification.
    Fields that were not stored (e.g. original sessionId, gitBranch) are
    omitted; the content blocks JSON is reconstructed from content_json.

    Returns the number of message lines written.
    """
    import h5py
    import numpy as np

    h5_path = Path(h5_path)
    out_path = Path(out_path)
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} already exists; pass overwrite=True")

    with h5py.File(h5_path, "r") as f:
        sessions = list(f.get("sessions", {}).keys())
        if not sessions:
            raise ValueError(f"No sessions found in {h5_path}")
        sid = session_id or sessions[0]
        if sid not in f["sessions"]:
            raise ValueError(f"Session {sid!r} not found; available: {sessions}")

        mg = f[f"sessions/{sid}/messages"]
        tcg = f[f"sessions/{sid}/tool_calls"]

        n_msg = mg["uuid"].shape[0]
        uuids    = mg["uuid"][:]
        parents  = mg["parent_uuid"][:]
        types    = mg["type"][:]
        roles    = mg["role"][:]
        models   = mg["model"][:]
        timestamps = mg["timestamp"][:]
        usages   = mg["usage"][:]
        idx      = mg["content_index"][:]

        # One read for all content bytes
        total_bytes = mg["content_bytes"].shape[0]
        all_content = bytes(mg["content_bytes"][:]) if total_bytes > 0 else b""

        n_tc = tcg["tool_use_id"].shape[0]
        tc_ids      = tcg["tool_use_id"][:]
        tc_msg_uuids = tcg["message_uuid"][:]
        tc_names    = tcg["name"][:]
        tc_idx      = tcg["call_index"][:]
        tc_is_error = tcg["is_error"][:]

        total_call_bytes = tcg["call_bytes"].shape[0]
        all_call = bytes(tcg["call_bytes"][:]) if total_call_bytes > 0 else b""

    def _s(v: Any) -> str:
        return v.decode("utf-8", errors="replace") if isinstance(v, (bytes, bytearray)) else str(v)

    def _slice(buf: bytes, off: int, length: int) -> str:
        return buf[off:off + length].decode("utf-8", errors="replace")

    # Build a lookup: message_uuid → list of tool call rows
    from collections import defaultdict
    tc_by_msg: dict[str, list[dict]] = defaultdict(list)
    for i in range(n_tc):
        tc_by_msg[_s(tc_msg_uuids[i])].append({
            "tool_use_id": _s(tc_ids[i]),
            "name": _s(tc_names[i]),
            "args_json": _slice(all_call, int(tc_idx[i]["args_off"]), int(tc_idx[i]["args_len"])),
            "result_text": _slice(all_call, int(tc_idx[i]["result_off"]), int(tc_idx[i]["result_len"])),
            "is_error": bool(tc_is_error[i]),
        })

    lines = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for i in range(n_msg):
            row_idx = idx[i]
            content_json_str = _slice(all_content, int(row_idx["json_off"]), int(row_idx["json_len"]))
            try:
                content = json.loads(content_json_str) if content_json_str else None
            except json.JSONDecodeError:
                content = content_json_str or None

            uid = _s(uuids[i])
            usage_row = usages[i]
            record: dict[str, Any] = {
                "type": _s(types[i]),
                "uuid": uid,
                "parentUuid": _s(parents[i]) or None,
                "timestamp": float(timestamps[i]),
                "message": {
                    "role": _s(roles[i]),
                    "model": _s(models[i]),
                    "content": content,
                    "usage": {
                        "input_tokens": int(usage_row["input_tokens"]),
                        "output_tokens": int(usage_row["output_tokens"]),
                        "cache_creation_input_tokens": int(
                            usage_row["cache_creation_input_tokens"]),
                        "cache_read_input_tokens": int(
                            usage_row["cache_read_input_tokens"]),
                    },
                },
            }
            out.write(json.dumps(record) + "\n")
            lines += 1

    return lines


def convert_many_packed(
    jsonl_paths: Iterable[str | Path],
    h5_path: str | Path,
    overwrite: bool = False,
) -> dict[str, dict[str, int]]:
    """Convert several JSONL files into one multi-session packed HDF5 file."""
    h5_path = Path(h5_path)
    if h5_path.exists() and overwrite:
        h5_path.unlink()
    out: dict[str, dict[str, int]] = {}
    for p in jsonl_paths:
        p = Path(p)
        out[p.name] = convert_jsonl_packed(p, h5_path, session_id=p.stem)
    return out
