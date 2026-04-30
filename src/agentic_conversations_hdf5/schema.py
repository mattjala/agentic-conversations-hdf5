"""Schema constants for agentic-conversations-hdf5.

See docs/schema.md for the full layout.
"""
from __future__ import annotations

import h5py
import numpy as np

SCHEMA_VERSION = 1

VLEN_STR = h5py.string_dtype(encoding="utf-8")

# Chunk size for the per-message parallel-array datasets. Tuned so each chunk
# holds ~64 messages — small enough that appending a single turn writes a
# bounded number of bytes, large enough that a context-reconstruction read
# pulls one or two chunks at most.
CHUNK_ROWS = 64

# Compound dtype for per-message token usage. Stored as a single numeric
# dataset so analytical queries (total tokens, cache hit rate over time) are
# a single hyperslab read with no JSON parsing.
USAGE_DTYPE = np.dtype([
    ("input_tokens", "i8"),
    ("output_tokens", "i8"),
    ("cache_creation_input_tokens", "i8"),
    ("cache_read_input_tokens", "i8"),
])

USAGE_FIELDS = USAGE_DTYPE.names

# Default compression for VLEN string and numeric datasets.
DEFAULT_COMPRESSION = "gzip"
DEFAULT_COMPRESSION_OPTS = 4
