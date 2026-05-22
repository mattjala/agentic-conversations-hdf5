# ORC vs. HDF5 as a conversation-log backend

`ORCSession` (`backends/orc_backend.py`) stores the message/tool-call log as
Apache ORC columnar tables via pyarrow, with dense vectors (embeddings, tool
result arrays) and artifacts in row-aligned `.npy`/`.npz` sidecars — the same
table-vs-blob split HDF5 uses. Benchmarked with `benchmark.py` against the
HDF5 backend on the same machine (HDF5 2.0.0 + zlib-ng h5py build).

ORC is **write-once**: a stripe is sealed when written and there is no efficient
in-place row append. Two modes bracket this:

- **batch** — buffer every turn in RAM, write the ORC file once at `close()`.
- **rewrite** — rewrite the whole file on every `add_turn` (the only way to get
  "live" persistence from a sealed format); O(N²) total.

## Write throughput (t/s)

| Backend                 | scenario | N=1000 | N=5000 |
|-------------------------|----------|--------|--------|
| ORC (batch)             | text     | 6,205  | 162,734 |
| ORC (batch)             | array    | 10,065 | 10,484 |
| HDF5 (batch=100) | text     | 20,883 | 26,342 |
| HDF5 (batch=100) | array    | 428    | 406    |
| HDF5 (batch=1)   | text     | 389    | 405    |
| HDF5 (batch=1)   | array    | 184    | 184    |

ORC per-turn **rewrite** (small N only — the O(N²) cliff):

| scenario | N=50 | N=200 | N=500 |
|----------|------|-------|-------|
| text     | 145  | 457   | 279   |
| array    | 132  | 98    | 55    |

## Read latency & size (N=5000)

| Backend                 | scenario | ctx ms | art ms | size kB |
|-------------------------|----------|--------|--------|---------|
| ORC (batch)             | text     | 0.81   | 1.15   | 858     |
| ORC (batch)             | array    | 1.05   | 1.14   | 53,279  |
| HDF5 (batch=100) | text     | 1.70   | 0.34   | 758     |
| HDF5 (batch=100) | array    | 2.40   | 0.37   | 55,886  |

## Reading the numbers

- **ORC-batch write t/s is huge but misleading as "live" throughput.** In batch
  mode `add_turn` only appends to Python lists; *all* I/O happens in one
  `orc.write_table` at close. So it measures "buffer all N in RAM + one bulk
  columnar write." That has large fixed overhead (only 6,205 t/s at N=1000)
  amortized over more rows (162,734 at N=5000). It is effectively HDF5
  with `batch_size = N`, minus durability: **nothing is persisted until close,
  and memory grows unbounded** — unacceptable for a live agent log.
- **ORC cannot do durable incremental append.** The honest "live" mode is
  rewrite, and it falls off a cliff: array-heavy 132 → 55 t/s from N=50 to 500.
  Unusable past a few hundred turns.
- **HDF5 gives a tunable durability/throughput knob** (`batch_size`):
  every batch is flushed and persisted incrementally, bounded memory, crash-safe
  to the last flushed batch. ORC offers only "all-or-nothing at close."
- **Reads:** ORC columnar reads are fast (0.8–1.1 ms) and actually beat
  HDF5 here — but `get_recent_context` reads the whole table then slices
  (pyarrow ORC has no cheap row-range tail), so this advantage erodes as files
  grow far beyond these sizes. HDF5's hyperslab read is true random access.
- **Artifacts:** HDF5 stores them natively (~0.34 ms); ORC needs an `.npz`
  sidecar (~1.15 ms).
- **Size:** comparable; ORC slightly larger for text, slightly smaller for array.

## Verdict

ORC is a strong **archival / bulk-export** format for a *finished* conversation
(one-shot columnar dump, fast scans, good compression) but a poor **live-logging**
backend: it has no durable incremental append, so you either buffer everything in
RAM (batch) or pay O(N²) (rewrite). HDF5 remains the better fit for the
incremental write-as-you-go agent-logging use case. A reasonable hybrid would be
HDF5 for live capture + an ORC export step for downstream analytics.
