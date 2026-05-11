# HDF5 Schemas

This document covers two independent uses of HDF5 in this project:

- **Session file schemas** — three alternative layouts for storing raw agent
  conversation logs (messages, tool calls, artifacts).
- **Vector store schemas** — three alternative layouts for the embedding store
  that backs claude-mem memory.

Each family has three variants that differ primarily in how variable-length
string data is handled. All variants within a family expose the same interface
and store the same logical data.

---

# Session File Schemas

A session file is a single `.h5` file containing one or more agent
conversations. The layout favours append-mostly writes, range reads for
context reconstruction, and columnar reads for analytical queries (token
usage, tool-call statistics).

## Shared structure (all three layouts)

The group hierarchy and attribute conventions are identical across all layouts.

```
/                                     attrs: schema_version (int = 1)
                                             source (str, e.g. "claude-code")
/sessions/
/sessions/<session_id>/               attrs: model, created_at, summary, cwd,
                                             git_branch, agent_version,
                                             permission_mode,
                                             session_id_original
/sessions/<sid>/messages/             dataset group — one row per message
/sessions/<sid>/messages/embeddings/  optional — one float32 (D,) dataset per
                                       message uuid, only when embedding given
/sessions/<sid>/tool_calls/           dataset group — one row per tool call
/sessions/<sid>/tool_calls/result_data/  optional — one typed array dataset per
                                          tool_use_id, for array-valued results
/sessions/<sid>/artifacts/            free-form — one named dataset per artifact
```

Within each dataset group, all datasets share the same length, use
`maxshape=(None,)` (extendable), and `chunks=(64,)`. Numeric datasets add the
shuffle filter; the three layouts differ in how string-valued columns are
stored.

## Messages columns

| Column         | Type (see layouts below) | Description                              |
|----------------|--------------------------|------------------------------------------|
| `uuid`         | str                      | Stable message id from the source log    |
| `parent_uuid`  | str                      | Parent message uuid; forms a DAG         |
| `type`         | str                      | "user", "assistant", "summary", …        |
| `role`         | str                      | API-level role inside the message block  |
| `timestamp`    | float64                  | Unix seconds (UTC)                       |
| `content_text` | str                      | Best-effort plain-text view of content   |
| `content_json` | str                      | Full content blocks JSON (lossless)      |
| `model`        | str                      | Per-message model id (assistant rows)    |
| `usage`        | compound (4 × int64)     | input / output / cache_creation / cache_read tokens |

## Tool-call columns

| Column         | Type (see layouts below) | Description                              |
|----------------|--------------------------|------------------------------------------|
| `tool_use_id`  | str                      | Anthropic `toolu_…` id                   |
| `message_uuid` | str                      | UUID of the assistant message that issued the call |
| `name`         | str                      | Tool name (e.g. "Read", "Bash")          |
| `args_json`    | str                      | Tool input as JSON                       |
| `result_text`  | str                      | Plain-text view of the result            |
| `result_uuid`  | str                      | UUID of the user message carrying the result |
| `timestamp`    | float64                  | Unix seconds                             |
| `is_error`     | uint8                    | 1 if the tool reported an error          |

---

## Layout 1: VLEN parallel arrays (schema.py)

Every string column is a separate 1-D dataset using HDF5's variable-length
UTF-8 string dtype. Each column is its own chunked dataset; `usage` is a
compound numeric dataset.

```
messages/
    uuid, parent_uuid, type, role, model   VLEN str  (N,)
    timestamp                              float64   (N,)
    content_text, content_json             VLEN str  (N,)
    usage                                  compound  (N,)

tool_calls/
    tool_use_id, message_uuid, name,
    result_uuid, args_json, result_text    VLEN str  (M,)
    timestamp                              float64   (M,)
    is_error                               uint8     (M,)
```

**Trade-offs:** Simplest layout. VLEN strings are stored in HDF5's global
heap, which the gzip filter does not touch — content fields are uncompressed.
Reading a single column (e.g. `usage` for token accounting) loads only that
dataset; analytical column access is efficient. An append touches one dataset
per column.

---

## Layout 2: Packed byte buffer (schema_packed.py)

Identifier string columns (`uuid`, `type`, `role`, `model`, etc.) remain VLEN.
The large unbounded content columns — `content_text` and `content_json` in
messages, `args_json` and `result_text` in tool calls — are replaced with flat
`uint8` byte buffers plus compound offset/length index datasets.

```
messages/
    uuid, parent_uuid, type, role, model   VLEN str  (N,)   [unchanged]
    timestamp                              float64   (N,)   [unchanged]
    usage                                  compound  (N,)   [unchanged]
    content_index                          compound  (N,)   (text_off u8, text_len u4,
                                                             json_off u8, json_len u4)
    content_bytes                          uint8     (B,)   all content bytes concatenated

tool_calls/
    tool_use_id, message_uuid, name,
    result_uuid                            VLEN str  (M,)   [unchanged]
    timestamp                              float64   (M,)   [unchanged]
    is_error                               uint8     (M,)   [unchanged]
    call_index                             compound  (M,)   (args_off u8, args_len u4,
                                                             result_off u8, result_len u4)
    call_bytes                             uint8     (C,)   all call bytes concatenated
```

To read `content_text` for message at row `i`: load `content_index[i]`, then
slice `content_bytes[text_off : text_off + text_len]` and decode as UTF-8.
New content is appended to the byte buffer; old bytes for overwritten rows
become dead space (acceptable for append-mostly logs).

**Trade-offs:** The byte buffer is a regular chunked dataset, so gzip
compresses the content columns. Identifier columns remain VLEN and uncompressed.
Read path for content requires two dataset accesses instead of one.

---

## Layout 3: Compound dataset (schema_compound.py)

Each group is a single compound dataset — one row per record — rather than
parallel 1-D arrays. String fields are split by boundedness:

- **Bounded identifiers** (`uuid`, `type`, `role`, `model`, tool names) use
  fixed-length byte-string types (`S40`, `S16`, etc.). These are stored inline
  in the chunk data and are compressible.
- **Unbounded content** (`content_text`, `content_json`, `args_json`,
  `result_text`) use VLEN UTF-8 strings. These still go to the HDF5 global heap
  and are not compressed by gzip.

```
messages/   compound (N,)  fields: uuid S40, parent_uuid S40, type S16,
                                    role S12, model S64, timestamp f8,
                                    content_text VLEN, content_json VLEN,
                                    input_tokens i8, output_tokens i8,
                                    cache_creation_input_tokens i8,
                                    cache_read_input_tokens i8

tool_calls/ compound (M,)  fields: tool_use_id S40, message_uuid S40,
                                    name S64, timestamp f8, is_error u1,
                                    args_json VLEN, result_text VLEN,
                                    result_uuid S40
```

**Trade-offs:** Row access (e.g. reading the last N messages for context) is
one compound slice instead of N separate dataset reads — compound wins here.
Column access (e.g. summing token usage) reads all chunk bytes and discards
non-selected fields, so the parallel layouts are faster for purely analytical
queries. File size is roughly equal to the VLEN layout since content fields
are still uncompressed.

---

# Vector Store Schemas

These schemas store embeddings and metadata for semantic search. They are
used by the claude-mem MCP shim (`claude-mem-vectors/`), not by the session
file reader. One `.h5` file per named collection.

**Note:** The original document text is not stored in any of these layouts.
Text is passed to the embedder at upsert time and then discarded; only the
resulting float32 vectors and metadata are persisted. A separate SQLite sidecar
(`texts.db`) stores the original text when needed (as in the MCP server).

## Shared structure (all three layouts)

```
/                         attrs: format_version (int), embedding_dim (int), n_used (int)
/embeddings  (N, D)       float32   chunked (chunk_rows × D) + gzip + shuffle
/tombstoned  (N,)         uint8     0 = live, 1 = soft-deleted
```

Deleted rows are tombstoned rather than removed; their slots are reused by
subsequent inserts. Metadata fields that claude-mem filters on are treated as
*indexed columns* (`doc_type`, `sqlite_id`, `project`, `field_type`,
`created_at_epoch`); anything else goes into an `extras_json` blob.

---

## Layout 1: VLEN (hdf5_store.py)

All string fields are separate 1-D VLEN string datasets — one dataset per
column, parallel to `/embeddings`.

```
/ids                   VLEN str  (N,)
/meta/doc_type         VLEN str  (N,)
/meta/sqlite_id        int64     (N,)
/meta/project          VLEN str  (N,)
/meta/field_type       VLEN str  (N,)
/meta/created_at_epoch int64     (N,)
/meta/extras_json      VLEN str  (N,)
```

**Trade-offs:** Simplest. VLEN strings land in the global heap and are not
compressed. An upsert touches 9 datasets.

---

## Layout 2: Packed (hdf5_packed_store.py)

Bounded metadata strings use fixed-length byte-string types (`S24`, `S48`).
Unbounded strings (document IDs, `extras_json`) are packed into a flat `uint8`
byte buffer with a compound offset/length index — the same technique as the
session file packed layout.

```
/doc_type              S24       (N,)   fixed-length, compressible
/project               S48       (N,)   fixed-length, compressible
/field_type            S24       (N,)   fixed-length, compressible
/sqlite_id             int64     (N,)
/created_at_epoch      int64     (N,)
/str_bytes             uint8     (B,)   all id + extras bytes concatenated
/str_index             compound  (N,)   (id_off u64, id_len u32, ex_off u64, ex_len u32)
```

**Trade-offs:** All string data is in regular chunked datasets, so gzip
compresses both fixed-length columns and the byte buffer. Upsert still touches
9 datasets. On overwrite, new bytes are appended and the old bytes become dead
space.

---

## Layout 3: Compound (hdf5_compound_store.py)

All per-document metadata is collapsed into a single compound dataset. The
fixed-length fields are stored inline in chunks (compressible); VLEN fields
for document ID and extras still go to the global heap.

```
/metadata  (N,)  compound:
    tombstoned          u1
    sqlite_id           int64
    created_at_epoch    int64
    doc_type            S24     inline in chunk, compressible
    project             S48     inline in chunk, compressible
    field_type          S24     inline in chunk, compressible
    id                  VLEN    unbounded doc-id string
    extras_json         VLEN    JSON blob
```

**Trade-offs:** An upsert touches 2 datasets instead of 9 (metadata +
embeddings); cold-start cache rebuild reads 1 compound slice instead of 7
column reads. VLEN fields are still uncompressed.
