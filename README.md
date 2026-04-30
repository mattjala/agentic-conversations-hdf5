# agentic-conversations-hdf5

Store agent conversation logs (e.g. Claude Code's JSONL session files) as
HDF5, with the goal of demonstrating measurable wins on:

1. **Storage size** — gzipped VLEN strings + columnar token-usage records
   beat raw JSONL on long sessions, especially ones with verbose tool output.
2. **Context reconstruction latency** — recovering the last *N* messages is
   one chunked hyperslab read instead of a full-file JSONL scan.
3. **Analytical queries** — total tokens, cache-hit rate over time, tool
   error rate, and similar are single hyperslab reads with no JSON parsing.
4. **Self-contained provenance** — one file holds the conversation, tool
   calls, token usage, *and* any binary artifacts the agent produced, all
   inspectable by standard tooling (`h5dump`, `h5ls`, `h5py`).

This is an alternative to formats like
[claude-mem](https://github.com/...) (SQLite) for the same logs.

## Install

```bash
pip install -e .
```

## Convert Claude Code JSONL → HDF5

```bash
agentic-conversations-hdf5 convert \
    ~/.claude/projects/<encoded-cwd>/<session-id>.jsonl \
    -o session.h5

agentic-conversations-hdf5 inspect session.h5
agentic-conversations-hdf5 tail session.h5 <session-id> -n 5
```

Multiple sessions can share a single file:

```bash
agentic-conversations-hdf5 convert \
    ~/.claude/projects/<encoded-cwd>/*.jsonl \
    -o all-sessions.h5
```

## Schema

See [`docs/schema.md`](docs/schema.md). The short version:

- `/sessions/<sid>/messages/` — parallel-array dataset group
  (uuid, parent_uuid, type, role, timestamp, content_text, content_json,
  model, usage[compound: input/output/cache_creation/cache_read tokens]).
- `/sessions/<sid>/tool_calls/` — parallel-array dataset group
  joining tool_use blocks to their tool_result by `tool_use_id`.
- `/sessions/<sid>/artifacts/` — arbitrary binary outputs.

`parent_uuid` preserves the conversation DAG (Claude Code allows forks),
so reconstructed sessions are not flattened.

## Benchmarks

```bash
pip install -e ".[bench]"
python benchmarks/benchmark.py --quick
```

Compares HDF5 against SQLite and JSON+NumPy on synthetic sessions across
text-only and array-heavy scenarios.

## Tests

```bash
pip install -e ".[dev]"
pytest
```
