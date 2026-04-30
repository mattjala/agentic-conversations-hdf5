# Option B: HDF5 as an Agent Session File

## Big Idea

A typical agent session produces scattered artifacts: JSON logs, SQLite event records,
numpy arrays from tool outputs, maybe a separate vector DB for in-session RAG. None of
these formats know about each other. Reconstructing full context requires joins, file
lookups, and format conversions.

HDF5 is hierarchical, self-describing, and handles both text and binary data. A single
`.h5` file can hold an entire agent session: conversation turns, embeddings, tool call
records, and any data artifacts produced along the way. The pitch: portable, inspectable,
self-contained sessions — and fast context reconstruction because everything is co-located.

This case is strongest for agents working with scientific/numerical data, where tool
outputs are naturally arrays and co-location has the most payoff.

## Target Metrics

1. **Session file size** — HDF5 vs. JSON + SQLite combo vs. single SQLite DB. Win is
   modest on text-only sessions; meaningful when tool outputs include arrays.

2. **Context reconstruction latency** — time to reload the last N turns including tool
   results, at varying session lengths. HDF5 wins when results contain large data
   (one read vs. JOIN + blob deserialize).

3. **Co-located artifact retrieval** — "give me the dataset from turn 12 and its
   description" — one hyperslab read vs. separate DB query + file open.

4. **Session portability** — qualitative: one file to copy, inspect with h5dump, share.
   Not a graph, but part of the narrative.

Graphs: bar charts for (2) and (3) at fixed session size, line charts scaling session
length for (1) and (2). Separate series for text-only vs. array-heavy sessions.

## Broad Sketch

### Schema

```
/sessions/{session_id}/
    /turns/{turn_id}/
        role          # string attribute
        content       # variable-length string dataset
        timestamp     # float64 attribute
        embedding     # float32 dataset, shape (D,) — optional
    /tool_calls/{call_id}/
        name          # string attribute
        args          # JSON string dataset
        result_text   # variable-length string dataset — optional
        result_data   # float32 or other typed dataset — optional, for array results
        timestamp     # float64 attribute
    /artifacts/{name}   # any HDF5-storable data produced during session
    /attrs:
        model         # LLM model used
        created_at    # unix timestamp
        summary       # short string description
```

### Interface

```python
session = HDF5Session("session.h5", session_id="abc123")
session.add_turn(role, content, embedding=None)
session.add_tool_call(name, args, result)
session.get_recent_context(n=20)       # reconstruct last N turns for prompt injection
session.store_artifact(name, data)
```

### Comparison Backends

- JSON files (one per session) + separate numpy files for arrays
- SQLite with tables: turns, tool_calls, artifacts (blobs)

### Benchmark Harness

Synthetic agent sessions at varying lengths and array-heaviness. Measure write time,
context reconstruction time, file size. Separate "text session" vs. "data session"
scenarios to show where HDF5 advantage grows.

## Honest Limitations to Document

- Story is weak for text-only sessions — JSON/SQLite are fine there
- No concurrent session writes
- Inspectability argument requires users to have h5dump / h5ls / this tooling
- Strongest case requires the domain (scientific agents) to be a fit
