"""Live Claude Code session hook — incremental JSONL → HDF5.

Called by Claude Code on UserPromptSubmit and Stop events. On each call it
reads any JSONL lines written since the last call, converts them to HDF5
rows using the packed backend, then updates the cursor stored in the HDF5
file itself.

Usage (the setup-hook CLI command does this automatically):

    # In ~/.claude/settings.json hooks section:
    "UserPromptSubmit": [{"hooks": [{"type": "command",
        "command": "python3 -m agentic_conversations_hdf5.hooks.live_session"}]}],
    "Stop": [{"hooks": [{"type": "command",
        "command": "python3 -m agentic_conversations_hdf5.hooks.live_session"}]}]

Environment:
    AGENTIC_HDF5_DIR   directory where per-session .h5 files are written
                       (default: ~/.claude/hdf5-sessions)
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Incremental conversion helpers
# ---------------------------------------------------------------------------

def _find_jsonl(session_id: str) -> Optional[Path]:
    """Locate the JSONL file for a session by scanning ~/.claude/projects/."""
    pattern = os.path.expanduser(f"~/.claude/projects/*/{session_id}.jsonl")
    matches = glob.glob(pattern)
    return Path(matches[0]) if matches else None


def _parse_ts(ts: Any) -> float:
    if ts is None:
        return 0.0
    if isinstance(ts, (int, float)):
        return float(ts)
    try:
        from datetime import datetime
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts).timestamp()
    except Exception:
        return 0.0


def _flatten_content(content: Any) -> str:
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
            parts.append(block.get("thinking", ""))
        elif btype == "tool_result":
            inner = block.get("content")
            if isinstance(inner, str):
                parts.append(inner)
            elif isinstance(inner, list):
                parts.append(_flatten_content(inner))
    return "\n".join(p for p in parts if p)


def _sync_jsonl_to_hdf5(jsonl_path: Path, h5_path: Path, session_id: str) -> None:
    """Incrementally sync new JSONL lines to the HDF5 file."""
    from ..backends.hdf5 import HDF5Session

    sess = HDF5Session(
        h5_path,
        session_id=session_id,
        mode="a",
        flush_every=0,   # flush only on close — this is a single short-lived call
        source="claude-code-live",
    )

    sg = sess._f[sess._root]

    # Cursor = number of JSONL lines already processed.
    cursor: int = int(sg.attrs.get("jsonl_line_cursor", 0))

    # Pending tool_uses that have not yet been matched to a tool_result.
    pending_raw = sg.attrs.get("pending_tool_uses", "{}")
    try:
        pending: dict[str, dict] = json.loads(pending_raw)
    except Exception:
        pending = {}

    # Session metadata written once.
    session_attrs: dict[str, str] = {}
    new_lines = 0

    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for lineno, raw in enumerate(f):
                if lineno < cursor:
                    continue
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                rtype = rec.get("type")
                new_lines += 1

                # Non-message entries: capture metadata, skip otherwise.
                if rtype not in {"user", "assistant", "summary"}:
                    if rtype == "permission-mode":
                        session_attrs.setdefault(
                            "permission_mode", rec.get("permissionMode", ""))
                    continue

                # Capture session metadata once.
                for k_src, k_dst in (
                    ("cwd", "cwd"),
                    ("gitBranch", "git_branch"),
                    ("version", "agent_version"),
                ):
                    if k_src in rec and k_dst not in session_attrs:
                        session_attrs[k_dst] = rec[k_src]

                msg = rec.get("message") or {}
                role = msg.get("role", "")
                model = msg.get("model", "")
                content = msg.get("content")
                usage = msg.get("usage") or {}
                uid = rec.get("uuid", "")
                parent = rec.get("parentUuid") or ""
                ts = _parse_ts(rec.get("timestamp"))

                content_text = _flatten_content(content)
                content_json = json.dumps(content) if content is not None else ""

                # Write message row.
                sess.append_message(
                    uuid=uid,
                    parent_uuid=parent,
                    type=rtype,
                    role=role,
                    timestamp=ts or time.time(),
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

                # Tool uses: park in pending dict until matching result arrives.
                if rtype == "assistant" and isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            pending[block.get("id", "")] = {
                                "tool_use_id": block.get("id", ""),
                                "message_uuid": uid,
                                "name": block.get("name", ""),
                                "args_json": json.dumps(block.get("input", {})),
                                "timestamp": ts or time.time(),
                            }

                # Tool results: match with pending tool_use and write row.
                if rtype == "user" and isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        if block.get("type") != "tool_result":
                            continue
                        tid = block.get("tool_use_id", "")
                        inner = block.get("content")
                        if isinstance(inner, list):
                            result_text = _flatten_content(inner)
                        else:
                            result_text = inner if isinstance(inner, str) else ""
                        is_error = bool(block.get("is_error", False))
                        pu = pending.pop(tid, None)
                        sess.append_tool_call(
                            tool_use_id=tid,
                            message_uuid=pu["message_uuid"] if pu else "",
                            name=pu["name"] if pu else "",
                            args_json=pu["args_json"] if pu else "",
                            result_text=result_text,
                            result_uuid=uid,
                            timestamp=pu["timestamp"] if pu else (ts or time.time()),
                            is_error=is_error,
                        )

    finally:
        # Persist cursor and pending state, then close (flushes + truncates).
        if new_lines > 0:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                total_lines = sum(1 for _ in f)
            sg.attrs["jsonl_line_cursor"] = total_lines
            sg.attrs["pending_tool_uses"] = json.dumps(pending)
            if session_attrs:
                sess.set_session_attrs(**session_attrs)
        sess.close()


# ---------------------------------------------------------------------------
# Hook entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Read Claude Code hook JSON from stdin, sync session to HDF5."""
    try:
        raw = sys.stdin.read()
        data = json.loads(raw) if raw.strip() else {}
    except Exception:
        data = {}

    session_id: str = data.get("session_id", "")
    if not session_id:
        sys.exit(0)

    output_dir = Path(
        os.environ.get("AGENTIC_HDF5_DIR", "~/.claude/hdf5-sessions")
    ).expanduser()

    jsonl_path = _find_jsonl(session_id)
    if jsonl_path is None or not jsonl_path.exists():
        sys.exit(0)

    output_dir.mkdir(parents=True, exist_ok=True)
    h5_path = output_dir / f"{session_id}.h5"

    try:
        _sync_jsonl_to_hdf5(jsonl_path, h5_path, session_id)
    except Exception:
        # Never block Claude Code — silently swallow all errors.
        pass

    sys.exit(0)


if __name__ == "__main__":
    main()
