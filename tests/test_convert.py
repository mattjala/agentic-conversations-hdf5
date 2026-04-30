"""End-to-end tests for the JSONL → HDF5 converter."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from agentic_conversations_hdf5 import HDF5Session, SCHEMA_VERSION
from agentic_conversations_hdf5.convert import convert_jsonl, _flatten_content

FIXTURE = Path(__file__).parent / "fixtures" / "sample.jsonl"


def test_flatten_content_handles_block_types():
    assert _flatten_content("hi") == "hi"
    assert _flatten_content(None) == ""
    blocks = [
        {"type": "text", "text": "hello"},
        {"type": "thinking", "thinking": "thought"},
        {"type": "tool_use", "id": "t1", "name": "X", "input": {}},
        {"type": "tool_result", "tool_use_id": "t1", "content": "out"},
    ]
    flat = _flatten_content(blocks)
    assert "hello" in flat
    assert "thought" in flat
    assert "out" in flat


def test_convert_writes_expected_shape(tmp_path):
    out = tmp_path / "session.h5"
    counts = convert_jsonl(FIXTURE, out, overwrite=True)

    assert counts["messages"] == 4
    assert counts["tool_uses"] == 1
    assert counts["tool_results"] == 1
    assert counts["skipped"] == 1  # the permission-mode line

    with h5py.File(out, "r") as f:
        assert int(f.attrs["schema_version"]) == SCHEMA_VERSION
        assert f.attrs["source"].decode() == "claude-code" \
            if isinstance(f.attrs["source"], bytes) else \
            f.attrs["source"] == "claude-code"

        sg = f["sessions/sample"]
        assert sg["messages/uuid"].shape == (4,)
        assert sg["tool_calls/tool_use_id"].shape == (1,)

        # Session-level metadata captured.
        cwd = sg.attrs["cwd"]
        if isinstance(cwd, bytes):
            cwd = cwd.decode()
        assert cwd == "/tmp/example"


def test_token_usage_aggregation(tmp_path):
    out = tmp_path / "session.h5"
    convert_jsonl(FIXTURE, out, overwrite=True)

    sess = HDF5Session(out, session_id="sample", mode="r")
    try:
        totals = sess.total_usage()
    finally:
        sess.close()

    assert totals["input_tokens"] == 10 + 5
    assert totals["output_tokens"] == 25 + 15
    assert totals["cache_creation_input_tokens"] == 100
    assert totals["cache_read_input_tokens"] == 500 + 600


def test_tool_use_joins_to_result(tmp_path):
    out = tmp_path / "session.h5"
    convert_jsonl(FIXTURE, out, overwrite=True)

    with h5py.File(out, "r") as f:
        tg = f["sessions/sample/tool_calls"]
        tu_id = tg["tool_use_id"][0]
        if isinstance(tu_id, bytes):
            tu_id = tu_id.decode()
        result = tg["result_text"][0]
        if isinstance(result, bytes):
            result = result.decode()
        name = tg["name"][0]
        if isinstance(name, bytes):
            name = name.decode()
        is_err = int(tg["is_error"][0])

    assert tu_id == "toolu_aaa"
    assert name == "Bash"
    assert "README.md" in result
    assert is_err == 0


def test_parent_uuid_preserved(tmp_path):
    """The DAG structure must round-trip — flatten only at read time."""
    out = tmp_path / "session.h5"
    convert_jsonl(FIXTURE, out, overwrite=True)

    with h5py.File(out, "r") as f:
        mg = f["sessions/sample/messages"]
        uuids = [u.decode() if isinstance(u, bytes) else u
                 for u in mg["uuid"][:]]
        parents = [p.decode() if isinstance(p, bytes) else p
                   for p in mg["parent_uuid"][:]]

    # First message has no parent.
    assert parents[0] == ""
    # Each subsequent message points at the previous in this linear fixture.
    for i in range(1, 4):
        assert parents[i] == uuids[i - 1]


def test_content_json_roundtrip(tmp_path):
    """content_json must be parseable and contain the original blocks."""
    out = tmp_path / "session.h5"
    convert_jsonl(FIXTURE, out, overwrite=True)

    with h5py.File(out, "r") as f:
        cj = f["sessions/sample/messages/content_json"][1]
        if isinstance(cj, bytes):
            cj = cj.decode()
    blocks = json.loads(cj)
    types = [b.get("type") for b in blocks]
    assert "text" in types
    assert "tool_use" in types
