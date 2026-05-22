# HDF5 Schemas

This document covers two independent uses of HDF5 in this project:

- **Session file schema** — storing raw agent conversation logs (messages,
  tool calls, artifacts).
- **Vector store schemas** — three alternative layouts for the embedding store
  that backs claude-mem memory.

---

# Session File Schemas

A session file is a single `.h5` file containing one or more agent
conversations. The layout favours append-mostly writes, range reads for
context reconstruction, and columnar reads for analytical queries (token
usage, tool-call statistics).

## Group hierarchy

```
/                                     attrs: schema_version (int = 1)
                                             source (str, e.g. "claude-code")
/sessions/
/sessions/<session_id>/               attrs: model, created_at, summary, cwd,
                                             git_branch, agent_version,
                                             permission_mode, session_id_original
/sessions/<sid>/messages/             dataset group — one row per message
/sessions/<sid>/tool_calls/           dataset group — one row per tool call
/sessions/<sid>/tool_calls/result_data/  optional — one typed array dataset per
                                          tool_use_id, for array-valued results
/sessions/<sid>/artifacts/            free-form — one named dataset per artifact
```

Per-row datasets share the same length and use `maxshape=(None,)` (extendable;
capacity is pre-allocated and doubled on growth, then truncated on close).
Numeric and byte datasets are chunked (`chunks=(64,)` for rows) with shuffle +
gzip; identifier columns are VLEN UTF-8 strings.

## Messages

Variable-length content is packed into a flat `uint8` byte buffer with a
compound offset/length index, so gzip compresses it in full.

```
messages/
    uuid, parent_uuid, type, role, model   VLEN str  (N,)
    timestamp                              float64   (N,)
    usage                                  compound  (N,)   input/output/cache_creation/cache_read (i8)
    content_index                          compound  (N,)   text_off u8, text_len u4, json_off u8, json_len u4
    content_bytes                          uint8     (B,)   all message content bytes
    has_embedding                          uint8     (N,)   1 if this row carries an embedding
    embeddings                             float32   (N, D) optional; created on the first embedding
```

To read `content_text` for row `i`: load `content_index[i]`, then slice
`content_bytes[text_off : text_off + text_len]` and decode UTF-8. Embeddings,
when present, are one consolidated `(N, D)` dataset row-aligned with the messages
(zero-filled for rows without one), so a range of embeddings is a single slice
read; `has_embedding` says which rows are real.

## Tool calls

Tool args and result text are packed into a second byte buffer, mirroring the
message content layout.

```
tool_calls/
    tool_use_id, message_uuid, name, result_uuid   VLEN str  (M,)
    timestamp                                       float64   (M,)
    is_error                                        uint8     (M,)
    call_index                                      compound  (M,)   args_off u8, args_len u4, result_off u8, result_len u4
    call_bytes                                      uint8     (C,)   all tool args/result bytes
```

Token usage is stored as a standalone compound numeric dataset so analytical
queries (total tokens, cache hit rate) are a single hyperslab read with no JSON
parsing. `parent_uuid` is preserved verbatim, so forks in the source log survive
the round-trip.

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
