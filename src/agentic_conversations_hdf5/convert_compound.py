"""Convert Claude Code JSONL session logs to/from the compound HDF5 format."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional
import json

from .backends.hdf5_compound import HDF5CompoundSession
from .convert import (
    _extract_tool_calls,
    _extract_tool_results,
    _flatten_content,
    _iter_jsonl,
    _parse_ts,
)


def convert_jsonl_compound(
    jsonl_path: str | Path,
    h5_path: str | Path,
    session_id: Optional[str] = None,
    overwrite: bool = False,
    flush_every: int = 0,
    compression: Optional[str] = "gzip",
) -> dict[str, int]:
    """Convert a Claude Code JSONL log to a compound HDF5 session file."""
    jsonl_path = Path(jsonl_path)
    h5_path = Path(h5_path)
    if h5_path.exists() and overwrite:
        h5_path.unlink()

    sid = session_id or jsonl_path.stem
    counts = {"messages": 0, "tool_uses": 0, "tool_results": 0, "skipped": 0}

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

    sess = HDF5CompoundSession(
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


def convert_many_compound(
    jsonl_paths: Iterable[str | Path],
    h5_path: str | Path,
    overwrite: bool = False,
) -> dict[str, dict[str, int]]:
    h5_path = Path(h5_path)
    if h5_path.exists() and overwrite:
        h5_path.unlink()
    out: dict[str, dict[str, int]] = {}
    for p in jsonl_paths:
        p = Path(p)
        out[p.name] = convert_jsonl_compound(p, h5_path, session_id=p.stem)
    return out
