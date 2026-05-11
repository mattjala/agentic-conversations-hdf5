"""Schema constants for the HDF5 compound-dataset variant.

Each group (messages, tool_calls) is a single compound dataset — one row
per record — rather than nine parallel 1-D arrays.

String field strategy
---------------------
* Bounded identifier fields (uuid, type, role, model, tool names) use
  fixed-length ASCII storage ('Sn' in numpy).  Fixed-length bytes are stored
  inline in the chunk data, so gzip runs on them.

* Unbounded content fields (content_text, content_json, args_json,
  result_text) use VLEN UTF-8 strings — h5py.string_dtype() → numpy
  object dtype.  These still go to the HDF5 global heap (same limitation
  as the original parallel-array layout), so gzip does not compress them.
  Truncating or capping them to a fixed length is not acceptable for a
  lossless session log.

Trade-offs vs. parallel-array layouts
--------------------------------------
* Row access (get_recent_context): one compound slice read vs. N separate
  dataset reads in the parallel layouts — compound should win here.
* Column access (total_usage): h5py field selection reads all chunk bytes
  then discards non-selected fields — parallel layout wins.
* File size: fixed-length fields are compressed; VLEN content is not
  (same as original).  In practice ~equal to the original layout.
"""
from __future__ import annotations

import h5py
import numpy as np

# Fixed-length sizes (bytes).  These are inline in chunks → compressible.
UUID_LEN  = 40   # covers full 36-char UUID plus slack
TYPE_LEN  = 16   # "user" / "assistant" / "summary"
ROLE_LEN  = 12   # "user" / "assistant"
MODEL_LEN = 64   # e.g. "claude-sonnet-4-6-20251001"
NAME_LEN  = 64   # tool name, e.g. "NotebookEdit", "WebSearch"

# Chunk rows: same as the parallel-array layouts.
CHUNK_ROWS = 64

# ---------------------------------------------------------------------------
# Compound dtype: one message row
# ---------------------------------------------------------------------------
# Fixed-length string fields ('Sn') are stored inline; the two content fields
# are VLEN (h5py.string_dtype() → dtype('O') in numpy) because their length
# is unbounded.  Usage counts are flattened into the compound to avoid a
# nested compound type (simpler numpy access, same semantics).

MSG_DTYPE = np.dtype([
    ("uuid",                         f"S{UUID_LEN}"),
    ("parent_uuid",                  f"S{UUID_LEN}"),
    ("type",                         f"S{TYPE_LEN}"),
    ("role",                         f"S{ROLE_LEN}"),
    ("model",                        f"S{MODEL_LEN}"),
    ("timestamp",                    "<f8"),
    ("content_text",                 h5py.string_dtype()),   # VLEN UTF-8
    ("content_json",                 h5py.string_dtype()),   # VLEN UTF-8
    ("input_tokens",                 "<i8"),
    ("output_tokens",                "<i8"),
    ("cache_creation_input_tokens",  "<i8"),
    ("cache_read_input_tokens",      "<i8"),
])

USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)

# ---------------------------------------------------------------------------
# Compound dtype: one tool-call row
# ---------------------------------------------------------------------------

TOOL_DTYPE = np.dtype([
    ("tool_use_id",   f"S{UUID_LEN}"),
    ("message_uuid",  f"S{UUID_LEN}"),
    ("name",          f"S{NAME_LEN}"),
    ("result_uuid",   f"S{UUID_LEN}"),
    ("timestamp",     "<f8"),
    ("is_error",      "u1"),
    ("args_json",     h5py.string_dtype()),   # VLEN UTF-8
    ("result_text",   h5py.string_dtype()),   # VLEN UTF-8
])
