"""Backend-interface conformance tests across HDF5 / SQLite / JSON."""
from __future__ import annotations

import numpy as np
import pytest

from agentic_conversations_hdf5 import HDF5Session, SQLiteSession, JSONSession


def _make_hdf5(tmp_path):
    return HDF5Session(tmp_path / "s.h5", session_id="t")


def _make_sqlite(tmp_path):
    return SQLiteSession(tmp_path / "s.db", session_id="t")


def _make_json(tmp_path):
    return JSONSession(tmp_path / "s_json", session_id="t")


@pytest.fixture(params=[_make_hdf5, _make_sqlite, _make_json],
                ids=["hdf5", "sqlite", "json"])
def make_session(request, tmp_path):
    return lambda: request.param(tmp_path)


def test_add_turn_increments_count(make_session):
    sess = make_session()
    try:
        assert sess.turn_count() == 0
        sess.add_turn("user", "hello")
        sess.add_turn("assistant", "hi back")
        assert sess.turn_count() == 2
    finally:
        sess.close()


def test_recent_context_returns_chronological(make_session):
    sess = make_session()
    try:
        sess.add_turn("user", "first")
        sess.add_turn("assistant", "second")
        sess.add_turn("user", "third")
        ctx = sess.get_recent_context(n=2)
        contents = [c["content"] for c in ctx]
        assert contents == ["second", "third"]
    finally:
        sess.close()


def test_artifact_round_trip(make_session):
    sess = make_session()
    try:
        data = np.arange(50, dtype=np.float32).reshape(5, 10)
        sess.store_artifact("a", data)
        out = sess.get_artifact("a")
        assert out is not None
        np.testing.assert_array_equal(out, data)
        assert sess.get_artifact("missing") is None
    finally:
        sess.close()


def test_embedding_optional(make_session):
    sess = make_session()
    try:
        sess.add_turn("user", "no embedding")
        sess.add_turn("user", "with embedding",
                      embedding=np.ones(8, dtype=np.float32))
        ctx = sess.get_recent_context(n=2)
        assert "embedding" not in ctx[0]
        assert "embedding" in ctx[1]
        np.testing.assert_array_equal(ctx[1]["embedding"], np.ones(8, dtype=np.float32))
    finally:
        sess.close()
