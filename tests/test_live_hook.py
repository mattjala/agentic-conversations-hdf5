"""Tests for the live Claude Code session hook (incremental JSONL -> HDF5)."""
from __future__ import annotations

import json

import h5py
import pytest

from agentic_conversations_hdf5 import HDF5Session
from agentic_conversations_hdf5.hooks.live_session import _sync_jsonl_to_hdf5


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


USER = {
    "type": "user", "uuid": "u1", "timestamp": "2026-05-22T10:00:00Z",
    "cwd": "/repo", "gitBranch": "main", "version": "1.2.3",
    "message": {"role": "user", "content": "fix the bug"},
}
ASSISTANT = {
    "type": "assistant", "uuid": "a1", "parentUuid": "u1",
    "timestamp": "2026-05-22T10:00:01Z",
    "message": {"role": "assistant", "model": "claude",
                "content": [{"type": "text", "text": "reading the file"},
                            {"type": "tool_use", "id": "tu1", "name": "Read",
                             "input": {"path": "auth.py"}}],
                "usage": {"input_tokens": 100, "output_tokens": 20,
                          "cache_read_input_tokens": 5}},
}
TOOL_RESULT = {
    "type": "user", "uuid": "u2", "parentUuid": "a1",
    "timestamp": "2026-05-22T10:00:02Z",
    "message": {"role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "tu1",
                             "content": "def authenticate(): ..."}]},
}


def test_full_sync(tmp_path):
    jsonl = tmp_path / "s.jsonl"
    h5 = tmp_path / "s.h5"
    _write_jsonl(jsonl, [USER, ASSISTANT, TOOL_RESULT])
    _sync_jsonl_to_hdf5(jsonl, h5, "sess")

    s = HDF5Session(h5, session_id="sess")
    assert s.turn_count() == 3
    ctx = s.get_recent_context(3)
    assert ctx[0]["content"] == "fix the bug"
    assert "reading the file" in ctx[1]["content"]
    usage = s.total_usage()
    assert usage["input_tokens"] == 100
    assert usage["output_tokens"] == 20
    assert usage["cache_read_input_tokens"] == 5
    s.close()

    # session metadata + the matched tool call
    with h5py.File(h5, "r") as f:
        sg = f["sessions/sess"]
        assert sg.attrs["cwd"] == "/repo"
        assert sg.attrs["git_branch"] == "main"
        tc = f["sessions/sess/tool_calls"]
        assert tc["tool_use_id"].shape[0] == 1
        assert tc["name"][0].decode() == "Read"


def test_incremental_sync_advances_cursor(tmp_path):
    """Two syncs across an appended file: cursor advances, tool_use in the first
    batch is matched to its tool_result in the second."""
    jsonl = tmp_path / "s.jsonl"
    h5 = tmp_path / "s.h5"

    _write_jsonl(jsonl, [USER, ASSISTANT])      # tool_use, no result yet
    _sync_jsonl_to_hdf5(jsonl, h5, "sess")
    with h5py.File(h5, "r") as f:
        assert int(f["sessions/sess"].attrs["jsonl_line_cursor"]) == 2

    _write_jsonl(jsonl, [USER, ASSISTANT, TOOL_RESULT])  # append the result line
    _sync_jsonl_to_hdf5(jsonl, h5, "sess")

    s = HDF5Session(h5, session_id="sess")
    assert s.turn_count() == 3   # not 5 — the first two lines aren't reprocessed
    s.close()
    with h5py.File(h5, "r") as f:
        assert int(f["sessions/sess"].attrs["jsonl_line_cursor"]) == 3
        tc = f["sessions/sess/tool_calls"]
        assert tc["tool_use_id"].shape[0] == 1
        assert tc["name"][0].decode() == "Read"   # matched across the two syncs


def test_resync_no_new_lines_is_noop(tmp_path):
    jsonl = tmp_path / "s.jsonl"
    h5 = tmp_path / "s.h5"
    _write_jsonl(jsonl, [USER, ASSISTANT, TOOL_RESULT])
    _sync_jsonl_to_hdf5(jsonl, h5, "sess")
    _sync_jsonl_to_hdf5(jsonl, h5, "sess")   # nothing new
    s = HDF5Session(h5, session_id="sess")
    assert s.turn_count() == 3
    s.close()
