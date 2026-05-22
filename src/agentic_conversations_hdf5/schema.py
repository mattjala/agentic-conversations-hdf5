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

# Default compression for string and numeric datasets.
DEFAULT_COMPRESSION = "gzip"
DEFAULT_COMPRESSION_OPTS = 4

# Variable-length text (message content, tool args/results) is stored in flat
# uint8 byte buffers that gzip compresses in full, with a compact offset/length
# index per row pointing into the buffer.

# Byte-buffer chunk size: 64 KB balances write latency against compression
# ratio — large enough for gzip to find cross-message patterns, small enough
# that a single-message append rarely crosses more than one chunk.
CONTENT_CHUNK_BYTES = 64 * 1024

# Per-message index into the message content buffer. *_off are absolute byte
# offsets (u8, files up to 16 EiB); *_len are byte counts (u4, up to 4 GB).
CONTENT_INDEX_DTYPE = np.dtype([
    ("text_off", "<u8"),
    ("text_len", "<u4"),
    ("json_off", "<u8"),
    ("json_len", "<u4"),
])

# Per-call index into the tool-call content buffer.
CALL_INDEX_DTYPE = np.dtype([
    ("args_off", "<u8"),
    ("args_len", "<u4"),
    ("result_off", "<u8"),
    ("result_len", "<u4"),
])
