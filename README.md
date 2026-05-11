# agentic-conversations-hdf5

[![CI](https://github.com/mattjala/agentic-conversations-hdf5/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/mattjala/agentic-conversations-hdf5/actions/workflows/ci.yml)

This repo explores two ways to use HDF5 in agentic AI pipelines.

The main body of the repo focuses on storing Claude Code session logs as HDF5 instead of JSONL. This shows improvement on recovering recent messages through hyperslab reads instead of full-file scans, and also supports faster aggregate information computation. Tool call data lives in the same file as any numerical artifacts the agent produced. Three HDF5 layout variants (VLEN, packed, compound) are implemented and tested.

The subfolder `claude-mem-vectors` focuses on using HDF5 as a drop-in vector store backend for [claude-mem](https://github.com/badlogic/claude-mem), replacing ChromaDB. Three HDF5 layout variants (VLEN, packed, compound) are measured against SQLite+BLOB and in-memory baselines on the exact interface claude-mem's `ChromaSync` exercises.

---

## Conversation Log Storage

### Install

```bash
pip install -e .
# for benchmarks:
pip install -e ".[bench]"
```

### Converting a Claude Code session

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

### Three layouts

This repo contains three HDF5 backends and two comparison backends (SQLite, JSON+NumPy), all implementing the same `SessionBackend` interface.

**VLEN** (`HDF5Session`) is the baseline HDF5 layout: one dataset per column, VLEN UTF-8 strings, gzip compression. VLEN strings live in HDF5's global heap and are incompressible regardless of dataset-level compression settings — so `content_text` and `content_json` compress less than you'd expect.

**Packed** (`HDF5PackedSession`) stores unbounded string columns as flat `uint8` byte buffers with a separate offset index, so gzip can actually compress them. Fixed-width fields (role, type, model) become `S24`/`S48` byte-string datasets. The tradeoff is schema complexity and higher reconstruction cost per row.

**Compound** (`HDF5CompoundSession`) stores all metadata for a message in a single compound dataset — one row per message, one read to load all columns. Write cost drops from 9 dataset writes to 2 per upsert. Fixed-length string fields are inline in the chunk and compressible; VLEN fields (uuid, content) are not.

### Schema

Full schema is in [`docs/schema.md`](docs/schema.md). The short version:

```
/sessions/<sid>/
    messages/       — parallel 1-D datasets: uuid, parent_uuid, type, role,
                      timestamp, content_text, content_json, model, usage
                      usage is compound (input/output/cache_creation/cache_read tokens)
    tool_calls/     — parallel 1-D datasets per tool invocation, joined by tool_use_id
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
