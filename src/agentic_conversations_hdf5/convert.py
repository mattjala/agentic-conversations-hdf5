"""Convert Claude Code JSONL session logs to HDF5.

Claude Code writes one JSON object per line to
``~/.claude/projects/<encoded-cwd>/<session-id>.jsonl``. Each line is one of:

- conversation message  (``type`` in {"user", "assistant", "summary"})
- session-state entry   (e.g. ``permission-mode``, ``file-history-snapshot``)

Conversation messages have ``uuid`` / ``parentUuid`` fields forming a DAG
(branches / forks share parents), a ``message`` block matching the Anthropic
API shape, and rich session metadata (cwd, gitBranch, version, sessionId).

This module preserves all of that with zero loss: the raw JSON of each
message's ``content`` is stored in ``content_json``, while ``content_text``
holds a best-effort flattened plain-text view for fast preview/scan.

Tool uses and tool results are also extracted into a parallel
``/tool_calls/`` table so analytical queries ("which tools failed", "what
were the args of every Read call") don't need JSON parsing.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from .backends.hdf5 import HDF5Session


def _parse_ts(ts: str | float | None) -> float:
    if ts is None:
        return 0.0
    if isinstance(ts, (int, float)):
        return float(ts)
    # ISO-8601 with trailing Z
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts).timestamp()
    except ValueError:
        return 0.0


def _flatten_content(content: Any) -> str:
    """Return a plain-text view of an Anthropic-style content field.

    Strings pass through. Lists of blocks are concatenated by joining the
    text of any text/thinking/tool_result blocks (skipping tool_use args and
    binary blocks). Anything unrecognised is dropped from the text view —
    the full JSON is preserved separately in ``content_json``.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            parts.append(block.get("text", ""))
        elif btype == "thinking":
            # Drop the signature blob, keep the prose.
            parts.append(block.get("thinking", ""))
        elif btype == "tool_result":
            inner = block.get("content")
            if isinstance(inner, str):
                parts.append(inner)
            elif isinstance(inner, list):
                parts.append(_flatten_content(inner))
    return "\n".join(p for p in parts if p)


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _extract_tool_calls(record: dict) -> Iterator[dict]:
    """Yield tool_use blocks from an assistant message."""
    msg = record.get("message") or {}
    content = msg.get("content")
    if not isinstance(content, list):
        return
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            yield {
                "tool_use_id": block.get("id", ""),
                "name": block.get("name", ""),
                "input": block.get("input", {}),
            }


def _extract_tool_results(record: dict) -> Iterator[dict]:
    """Yield tool_result blocks from a user message."""
    msg = record.get("message") or {}
    content = msg.get("content")
    if not isinstance(content, list):
        return
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_result":
            inner = block.get("content")
            if isinstance(inner, list):
                text = _flatten_content(inner)
            else:
                text = inner if isinstance(inner, str) else ""
            yield {
                "tool_use_id": block.get("tool_use_id", ""),
                "result_text": text,
                "is_error": bool(block.get("is_error", False)),
            }


def convert_jsonl(
    jsonl_path: str | Path,
    h5_path: str | Path,
    session_id: Optional[str] = None,
    overwrite: bool = False,
    flush_every: int = 0,
    compression: Optional[str] = "gzip",
) -> dict[str, int]:
    """Convert a Claude Code JSONL log to an HDF5 session file.

    Returns a dict of counts: messages, tool_uses, tool_results, skipped.
    """
    jsonl_path = Path(jsonl_path)
    h5_path = Path(h5_path)
    if h5_path.exists() and overwrite:
        h5_path.unlink()

    sid = session_id or jsonl_path.stem

    counts = {"messages": 0, "tool_uses": 0, "tool_results": 0, "skipped": 0}

    # First pass: collect tool_result blocks keyed by tool_use_id. In real
    # Claude Code JSONL the assistant's tool_use comes *before* the user's
    # tool_result, so to write a single joined tool_calls row we need the
    # result up front.
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

    sess = HDF5Session(
        h5_path,
        session_id=sid,
        mode="a",
        flush_every=flush_every,
        compression=compression,
        source="claude-code",
    )

    try:
        for rec in _iter_jsonl(jsonl_path):
            rtype = rec.get("type")

            # State entries: capture session metadata, otherwise skip.
            if rtype not in {"user", "assistant", "summary"}:
                if rtype == "permission-mode":
                    session_attrs.setdefault(
                        "permission_mode", rec.get("permissionMode", ""))
                counts["skipped"] += 1
                continue

            # Capture session-level metadata once.
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

        # Orphan tool_results (no matching assistant tool_use found)
        # still get a row so nothing is silently dropped.
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


def convert_many(
    jsonl_paths: Iterable[str | Path],
    h5_path: str | Path,
    overwrite: bool = False,
) -> dict[str, dict[str, int]]:
    """Convert several JSONL files into one multi-session HDF5 file."""
    h5_path = Path(h5_path)
    if h5_path.exists() and overwrite:
        h5_path.unlink()
    out: dict[str, dict[str, int]] = {}
    for p in jsonl_paths:
        p = Path(p)
        out[p.name] = convert_jsonl(p, h5_path, session_id=p.stem)
    return out


def unpack_to_jsonl(
    h5_path: str | Path,
    out_path: str | Path,
    session_id: Optional[str] = None,
    overwrite: bool = False,
) -> int:
    """Reconstruct approximate JSONL from an HDF5 session file.

    Best-effort round-trip for inspection/verification. Fields that were not
    stored (e.g. original sessionId, gitBranch) are omitted; content blocks are
    rebuilt from content_json. Returns the number of message lines written.
    """
    import h5py

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
        n_msg = mg["uuid"].shape[0]
        uuids = mg["uuid"][:]
        parents = mg["parent_uuid"][:]
        types = mg["type"][:]
        roles = mg["role"][:]
        models = mg["model"][:]
        timestamps = mg["timestamp"][:]
        usages = mg["usage"][:]
        idx = mg["content_index"][:]
        all_content = bytes(mg["content_bytes"][:]) if mg["content_bytes"].shape[0] else b""

    def _s(v: Any) -> str:
        return v.decode("utf-8", errors="replace") if isinstance(v, (bytes, bytearray)) else str(v)

    def _slice(buf: bytes, off: int, length: int) -> str:
        return buf[off:off + length].decode("utf-8", errors="replace")

    lines = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for i in range(n_msg):
            row = idx[i]
            content_json_str = _slice(all_content, int(row["json_off"]), int(row["json_len"]))
            try:
                content = json.loads(content_json_str) if content_json_str else None
            except json.JSONDecodeError:
                content = content_json_str or None
            u = usages[i]
            record: dict[str, Any] = {
                "type": _s(types[i]),
                "uuid": _s(uuids[i]),
                "parentUuid": _s(parents[i]) or None,
                "timestamp": float(timestamps[i]),
                "message": {
                    "role": _s(roles[i]),
                    "model": _s(models[i]),
                    "content": content,
                    "usage": {k: int(u[k]) for k in u.dtype.names},
                },
            }
            out.write(json.dumps(record) + "\n")
            lines += 1
    return lines
