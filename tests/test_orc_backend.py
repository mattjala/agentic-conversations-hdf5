"""Tests for the ORC backend (batch and rewrite modes). Needs the 'orc' extra."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyarrow")

from agentic_conversations_hdf5 import ORCSession


@pytest.mark.parametrize("mode", ["batch", "rewrite"])
def test_roundtrip(tmp_path, mode):
    d = tmp_path / f"orc_{mode}"
    s = ORCSession(d, session_id="t", mode=mode)
    expected = []
    for i in range(15):
        emb = np.arange(4, dtype=np.float32) + i if i % 2 == 0 else None
        s.add_turn("user" if i % 2 == 0 else "assistant", f"m{i}", embedding=emb)
        expected.append((f"m{i}", emb))
    s.store_artifact("art", np.arange(6, dtype=np.float32).reshape(2, 3))
    s.close()

    s2 = ORCSession(d, session_id="t", mode=mode)
    assert s2.turn_count() == 15
    ctx = s2.get_recent_context(15)
    for got, (content, emb) in zip(ctx, expected):
        assert got["content"] == content
        if emb is None:
            assert "embedding" not in got
        else:
            np.testing.assert_array_equal(got["embedding"], emb)
    np.testing.assert_array_equal(
        s2.get_artifact("art"), np.arange(6, dtype=np.float32).reshape(2, 3))
    assert s2.get_artifact("missing") is None
    s2.close()


def test_invalid_mode(tmp_path):
    with pytest.raises(ValueError):
        ORCSession(tmp_path / "x", session_id="t", mode="bogus")
