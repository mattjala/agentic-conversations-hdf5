# agentic-conversations-hdf5

[![CI](https://github.com/mattjala/agentic-conversations-hdf5/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/mattjala/agentic-conversations-hdf5/actions/workflows/ci.yml)

This repo explores two ways to use HDF5 in agentic AI pipelines.

The main body of the repo focuses on storing Claude Code session logs as HDF5 instead of JSONL. This shows improvement on recovering recent messages through hyperslab reads instead of full-file scans, and also supports faster aggregate information computation. Tool call data lives in the same file as any numerical artifacts the agent produced.

The subfolder `claude-mem-vectors` focuses on using HDF5 as a drop-in vector store backend for [claude-mem](https://github.com/badlogic/claude-mem), replacing ChromaDB. Three HDF5 layout variants (VLEN, packed, compound) are measured against SQLite+BLOB and in-memory baselines on the exact interface claude-mem's `ChromaSync` exercises.

---

## Conversation Log Storage

### Install

```bash
pip install -e .
# for benchmarks:
pip install -e ".[bench]"
```

### Live session recording (hook)

The easiest way to capture sessions is the live hook, which writes incrementally to HDF5 as Claude Code runs — no post-hoc conversion needed.

```bash
pip install -e .
agentic-conversations-hdf5 setup-hook
```

That one command patches `~/.claude/settings.json` to register hooks on `UserPromptSubmit` and `Stop`. Sessions are written to `~/.claude/hdf5-sessions/<session-id>.h5` by default.

```bash
# Inspect a live session (while Claude Code is running or after):
agentic-conversations-hdf5 inspect ~/.claude/hdf5-sessions/<session-id>.h5
```

To write files to a different directory:

```bash
agentic-conversations-hdf5 setup-hook --output-dir ~/my-sessions
# or set the env var when the hook runs:
export AGENTIC_HDF5_DIR=~/my-sessions
```

To remove the hook:

```bash
agentic-conversations-hdf5 teardown-hook
```

The hook never blocks Claude Code — all errors are swallowed silently so a broken HDF5 install cannot interrupt your session.

### Converting a Claude Code session (post-hoc)

```bash
agentic-conversations-hdf5 convert \
    ~/.claude/projects/<encoded-cwd>/<session-id>.jsonl \
    -o session.h5

agentic-conversations-hdf5 inspect session.h5
agentic-conversations-hdf5 tail session.h5 <session-id> -n 5
```

Multiple sessions can share one file:

```bash
agentic-conversations-hdf5 convert \
    ~/.claude/projects/<encoded-cwd>/*.jsonl \
    -o all-sessions.h5
```

### Backends

`HDF5Session` implements the `SessionBackend` interface; SQLite and JSON+NumPy backends implement the same interface for benchmark comparison.

Identifier columns are VLEN UTF-8 strings; unbounded content (`content_text`, `content_json`, tool args/results) is packed into flat `uint8` byte buffers with a compound offset/length index, so gzip compresses it in full. Token usage is a standalone compound numeric dataset for one-read analytical queries. Embeddings, when present, are a single consolidated `(N, D)` dataset.

### Schema

Full schema is in [`docs/schema.md`](docs/schema.md). The short version:

```
/sessions/<sid>/
    messages/       — uuid, parent_uuid, type, role, model (VLEN str), timestamp,
                      usage (compound), content_index + content_bytes (packed text),
                      has_embedding, embeddings (N, D)
    tool_calls/     — one row per tool invocation, joined by tool_use_id;
                      args/result packed into call_index + call_bytes
    artifacts/      — arbitrary binary outputs (figures, arrays, etc.)
```

`parent_uuid` is preserved verbatim, so forks in the source log survive the round-trip. The `usage` compound dataset is the main analytical win: total cache tokens for a session is `arr["cache_read_input_tokens"].sum()` — one read, no JSON parsing.

### Benchmarks

```bash
python benchmarks/benchmark.py --quick
```

Synthetic sessions are generated via `benchmarks/gen_synthetic.py`. Test fixtures at three scales (2 MB, 25 MB, 250 MB) live in `tests/fixtures/`.

---

## HDF5 as a Vector Store for claude-mem

Source and benchmarks are in `claude-mem-vectors/`. The `VectorStore` ABC in `claude-mem-vectors/store/vector_store.py` mirrors exactly the interface `ChromaSync` calls: `upsert`, `delete`, `query` with metadata `where` filters, `list_ids`, and `update_metadata`. Swapping backends requires no changes above the store layer.

The same three layout variants from the conversation log portion appear here: VLEN, packed, and compound, applied to embedding metadata rather than conversation turns.

### Using HDF5 as your claude-mem backend

An MCP shim server that acts as a drop-in replacement for `chroma-mcp` is in
`claude-mem-vectors/mcp_server/`. It exposes the same `chroma_*` tool interface
claude-mem calls, backed by your choice of HDF5 or SQLite. No changes to
claude-mem are required — you change one line in your MCP config.

See [`claude-mem-vectors/mcp_server/README.md`](claude-mem-vectors/mcp_server/README.md)
for installation and configuration instructions.

### Results and benchmarks

Full benchmark tables, design comparison, and reproduction instructions are in
[`claude-mem-vectors/results.md`](claude-mem-vectors/results.md).

The short version: for claude-mem's hook-driven write pattern (one document per
hook call), SQLite is the practical choice — `h5py.flush()` dominates upsert
cost regardless of layout, putting all three HDF5 variants ~200× behind SQLite
at batch size 1. HDF5 earns its place if sessions also store large numerical
artifacts alongside embeddings, which is the scenario where its hierarchical
structure adds something SQLite cannot match.

---

## Tests

```bash
pip install -e ".[dev]"
pytest
```
