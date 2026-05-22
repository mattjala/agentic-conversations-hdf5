"""Tests for HDF5Session: batched writes and consolidated embeddings."""
from __future__ import annotations

import h5py
import numpy as np
import pytest

from agentic_conversations_hdf5 import HDF5Session


@pytest.mark.parametrize("batch", [1, 10, 100])
@pytest.mark.parametrize("n", [0, 1, 7, 257])  # 257 crosses the 256 pre-alloc boundary
def test_batched_writes_roundtrip(tmp_path, batch, n):
    p = tmp_path / f"b{batch}_{n}.h5"
    s = HDF5Session(p, session_id="t", batch_size=batch)
    for i in range(n):
        s.add_turn("user" if i % 2 == 0 else "assistant", f"msg-{i}")
        assert s.turn_count() == i + 1  # buffered rows are counted
    s.close()

    s2 = HDF5Session(p, session_id="t", batch_size=batch)
    assert s2.turn_count() == n  # reopen must not over/under-count
    ctx = s2.get_recent_context(n)
    assert [c["content"] for c in ctx] == [f"msg-{i}" for i in range(n)]
    s2.close()


def test_read_sees_buffered_rows(tmp_path):
    s = HDF5Session(tmp_path / "x.h5", session_id="t", batch_size=50)
    for i in range(10):
        s.add_turn("user", f"m{i}")
    # fewer than batch_size written; a read must flush and return them
    ctx = s.get_recent_context(10)
    assert [c["content"] for c in ctx] == [f"m{i}" for i in range(10)]
    s.close()


@pytest.mark.parametrize("batch", [1, 10])
def test_sparse_embeddings(tmp_path, batch):
    p = tmp_path / f"e{batch}.h5"
    s = HDF5Session(p, session_id="t", batch_size=batch)
    expected = []
    for i in range(20):
        emb = np.arange(8, dtype=np.float32) + i if i % 3 == 0 else None
        s.add_turn("user", f"m{i}", embedding=emb)
        expected.append(emb)
    s.close()

    s2 = HDF5Session(p, session_id="t")
    for got, emb in zip(s2.get_recent_context(20), expected):
        if emb is None:
            assert "embedding" not in got
        else:
            np.testing.assert_array_equal(got["embedding"], emb)
    s2.close()


def test_no_embeddings_means_no_dataset(tmp_path):
    p = tmp_path / "t.h5"
    s = HDF5Session(p, session_id="t", batch_size=10)
    for i in range(15):
        s.add_turn("user", f"m{i}")
    s.close()
    with h5py.File(p, "r") as f:
        mg = f["sessions/t/messages"]
        assert "embeddings" not in mg          # not created when unused
        assert "has_embedding" in mg
        assert int(mg["has_embedding"][:].sum()) == 0


def test_batch1_equivalent_to_default(tmp_path):
    """batch_size=1 must produce the same logical content as any other batch."""
    contents = [f"turn-{i}" for i in range(30)]
    paths = {}
    for batch in (1, 7, 30):
        p = tmp_path / f"cmp{batch}.h5"
        s = HDF5Session(p, session_id="t", batch_size=batch)
        for c in contents:
            s.add_turn("user", c)
        s.close()
        s2 = HDF5Session(p, session_id="t")
        paths[batch] = [c["content"] for c in s2.get_recent_context(30)]
        s2.close()
    assert paths[1] == paths[7] == paths[30] == contents
