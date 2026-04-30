# Schema (v1)

A single `.h5` file is a self-contained store of one or more agent sessions.
The layout favours **append-mostly writes**, **range reads** for context
reconstruction, and **columnar reads** for analytical queries (token usage,
tool-call statistics).

## Top-level

```
/                                     attrs:
                                          schema_version : int    (= 1)
                                          source         : str    e.g. "claude-code"
/sessions/                            group
/sessions/<session_id>/               group   (one per logical conversation)
```

Multiple sessions may live side-by-side in one file.

## Per-session attributes

```
/sessions/<sid>/                      attrs:
    model              : str    primary model used (best-effort)
    created_at         : f64    unix timestamp at first write
    summary            : str    short human-readable description (optional)
    cwd                : str    captured from JSONL (Claude Code only)
    git_branch         : str    captured from JSONL
    agent_version      : str    e.g. "2.1.109"
    permission_mode    : str    captured from JSONL state entries (optional)
    session_id_original: str    UUID from the source log (optional)
```

## `/sessions/<sid>/messages/` — parallel-array dataset group

All datasets are 1-D, share the same length *N*, are extendable
(`maxshape=(None,)`), and use chunked storage (`chunks=(64,)`) with gzip
compression. VLEN strings are gzipped without shuffle; numeric datasets
add the shuffle filter (helps gzip on integer columns).

| Dataset        | dtype             | Description                                              |
|----------------|-------------------|----------------------------------------------------------|
| `uuid`         | VLEN str          | Stable message id from the source log.                   |
| `parent_uuid`  | VLEN str          | Parent message uuid (empty for the root). Forms a DAG.   |
| `type`         | VLEN str          | "user", "assistant", "summary", ...                      |
| `role`         | VLEN str          | API-level role inside the `message` block.               |
| `timestamp`    | f64               | Unix seconds (UTC).                                      |
| `content_text` | VLEN str          | Best-effort plain-text view of the content.              |
| `content_json` | VLEN str          | Full content blocks JSON for lossless round-trip.        |
| `model`        | VLEN str          | Per-message model id (assistant rows only).              |
| `usage`        | compound (4 × i8) | Per-message token counts — see below.                    |

`usage` compound fields:

```
input_tokens                  : i8
output_tokens                 : i8
cache_creation_input_tokens   : i8
cache_read_input_tokens       : i8
```

Storing usage as a numeric compound dataset is the key win over JSONL: a
question like "total cache-read tokens for this session" is a single hyperslab
read followed by `arr["cache_read_input_tokens"].sum()` — no JSON parsing, no
per-row dict overhead.

### `/sessions/<sid>/messages/embeddings/`

Optional. One dataset per message (keyed by `uuid`), float32 of shape (D,).
Only written when an embedding is supplied. Sparse population is cheap because
HDF5 doesn't pay for absent groups/datasets at read time.

## `/sessions/<sid>/tool_calls/` — parallel-array dataset group

Same shape and storage rules as `/messages/`. Length *M*, one row per tool
invocation. The link to its parent assistant message and its result message is
explicit, so analytical queries don't need to walk JSON.

| Dataset        | dtype     | Description                                          |
|----------------|-----------|------------------------------------------------------|
| `tool_use_id`  | VLEN str  | Anthropic `toolu_…` id.                              |
| `message_uuid` | VLEN str  | UUID of the assistant message that issued the call.  |
| `name`         | VLEN str  | Tool name (e.g. "Read", "Bash").                     |
| `args_json`    | VLEN str  | Tool input as JSON.                                  |
| `result_text`  | VLEN str  | Plain-text view of the result.                       |
| `result_uuid`  | VLEN str  | UUID of the user message carrying the tool result.   |
| `timestamp`    | f64       | Unix seconds.                                        |
| `is_error`     | u1        | 1 if the tool reported an error, else 0.             |

### `/sessions/<sid>/tool_calls/result_data/`

Optional. One dataset per tool call (keyed by `tool_use_id`) for tool results
that are naturally arrays (numpy ndarrays, sensor traces, etc.). This is where
the co-location story for scientific agents lives.

## `/sessions/<sid>/artifacts/`

Free-form. One named dataset per artifact the agent produces during the
session (figures, intermediate arrays, processed outputs). Same file, same
session, no separate filesystem to keep in sync.

## Design notes / trade-offs

- **Parallel arrays vs. compound dataset for messages.** A single compound
  dataset over all message columns reads more cleanly but VLEN strings inside
  HDF5 compounds compress poorly and are awkward to extend. Splitting into
  parallel arrays lets gzip work per-column (which is the right granularity:
  `model` and `type` compress to almost nothing, `content_*` compress well).
- **No indexes.** Lookups by uuid use linear scan. For typical session sizes
  (≤ a few thousand messages) this is fast enough; if it ever isn't, an
  optional `/messages/_uuid_index` could be added without a schema break.
- **`content_text` vs. `content_json`.** Keeping both costs space but means a
  preview/scan never has to parse JSON, while a full round-trip never loses
  information (thinking blocks, tool_use args, signatures, etc.).
- **`parent_uuid`.** Preserved verbatim so forks/branches in the source log
  survive the round-trip. A flat-list view is recoverable by sorting on
  timestamp.
- **Schema version.** Bumped on any breaking change. Readers should refuse
  files with `schema_version > 1` rather than guess.
