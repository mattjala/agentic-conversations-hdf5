"""Schema constants for the HDF5 packed-bytes variant.

The packed variant replaces VLEN string datasets for variable-length text
(content_text, content_json in messages; args_json, result_text in tool_calls)
with flat uint8 byte buffers plus compact compound offset-length index datasets.

Why this matters: VLEN strings are stored in the HDF5 global heap (H5HG),
which is written separately from the chunked dataset storage. The filter
pipeline (gzip, shuffle) only runs on chunk data, never on global heap objects.
uint8 datasets are fixed-width and fully compressible.

Layout per group:

  messages/
    uuid, parent_uuid, type, role, model   VLEN str  (N,)   [unchanged]
    timestamp                              float64   (N,)   [unchanged]
    usage                                  compound  (N,)   [unchanged]
    content_index                          compound  (N,)   text/json offsets+lengths
    content_bytes                          uint8     (B,)   all content bytes

  tool_calls/
    tool_use_id, message_uuid, name, result_uuid  VLEN str  (M,)   [unchanged]
    timestamp                                     float64   (M,)   [unchanged]
    is_error                                      uint8     (M,)   [unchanged]
    call_index                                    compound  (M,)   args/result offsets+lengths
    call_bytes                                    uint8     (C,)   all call bytes
"""
from __future__ import annotations

import numpy as np

# Byte-buffer chunk size: 64 KB balances write latency against compression
# ratio — large enough for gzip to find patterns across multiple messages,
# small enough that a single-message append rarely crosses more than one chunk.
CONTENT_CHUNK_BYTES = 64 * 1024

# Per-message index into content_bytes.
# text_off / json_off are absolute byte offsets; text_len / json_len are byte
# counts. u4 supports up to 4 GB per field; u8 offsets handle files up to
# 16 EiB.
CONTENT_INDEX_DTYPE = np.dtype([
    ("text_off", "<u8"),
    ("text_len", "<u4"),
    ("json_off", "<u8"),
    ("json_len", "<u4"),
])

# Per-call index into call_bytes.
CALL_INDEX_DTYPE = np.dtype([
    ("args_off", "<u8"),
    ("args_len", "<u4"),
    ("result_off", "<u8"),
    ("result_len", "<u4"),
])
