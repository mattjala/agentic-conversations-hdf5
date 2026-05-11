# claude-mem Vector Store Benchmark Results

Three HDF5 layout variants (VLEN, packed, compound) plus SQLite-blob and in-memory,
all measured against the same `VectorStore` interface.  
Embeddings: `FakeHashEmbedder`, d=384. Metadata: synthetic claude-mem document mix.

---

## Upsert Throughput (N=10,000 total)

In claude-mem's hook-based write path, bs=1 is the realistic case (one doc per hook call).

| Backend        | bs=1 (d/s) | bs=1 p50 (ms) | bs=1000 (d/s) | bs=1000 p50 (ms) |
|----------------|------------|---------------|---------------|------------------|
| in-memory      | 37,613     | 0.02          | 40,038        | 23.2             |
| SQLite + BLOB  | 7,847      | 0.07          | 26,428        | 37.7             |
| HDF5 VLEN      | 39         | 25.4          | 6,529         | 155.8            |
| HDF5 compound  | 39         | 25.3          | 6,365         | 159.9            |
| HDF5 packed    | 37         | 27.3          | 3,631         | 278.4            |

**Key finding:** All three HDF5 layouts are essentially identical at bs=1 (~39 d/s, ~25ms).
`h5py.flush()` dominates; reducing dataset writes from 9 (VLEN) to 2 (compound) gives no speedup.
SQLite is ~200× faster at bs=1. At packed bs=1000, packed is 2× slower than VLEN/compound
due to the additional byte-buffer append operations.

---

## Query Latency (k=20, 64 queries)

### N=10,000

| Backend        | no filter p50 | project p50 | project+type p50 |
|----------------|---------------|-------------|------------------|
| SQLite + BLOB  | 89.7 ms       | 24.6 ms     | 18.9 ms          |
| HDF5 VLEN      | 79.6 ms       | 55.5 ms     | 55.6 ms          |
| HDF5 compound  | 83.0 ms       | 59.0 ms     | 59.5 ms          |
| HDF5 packed    | 110.8 ms      | 86.6 ms     | 90.2 ms          |

### N=1,000

| Backend        | no filter p50 | project p50 | project+type p50 |
|----------------|---------------|-------------|------------------|
| SQLite + BLOB  | 4.3 ms        | 1.1 ms      | 0.8 ms           |
| HDF5 VLEN      | 4.4 ms        | 7.1 ms      | 6.1 ms           |
| HDF5 compound  | 4.8 ms        | 7.5 ms      | 6.4 ms           |
| HDF5 packed    | 22.3 ms       | 25.0 ms     | 24.1 ms          |

**Key finding:** SQLite wins on filtered queries — SQL WHERE prunes candidates before scan.
VLEN and compound are nearly identical; both use an in-memory numpy metadata cache so filter
cost is the same regardless of HDF5 layout. Packed is slowest: result reconstruction requires
per-row byte-buffer reads that bypass the cache. At N=1,000 packed is already 5× slower than
VLEN on unfiltered queries.

---

## File Size (N=100,000)

| Backend               | Total (MB) | Per-doc (B) | vs SQLite |
|-----------------------|------------|-------------|-----------|
| SQLite + BLOB         | 215.8      | 2,159       | baseline  |
| HDF5 VLEN (no comp.)  | 179.9      | 1,799       | −17%      |
| HDF5 VLEN gzip1       | 156.6      | 1,566       | −27%      |
| HDF5 VLEN gzip4       | 155.6      | 1,556       | −28%      |
| HDF5 VLEN gzip9       | 153.9      | 1,539       | −29%      |
| HDF5 compound         | 139.8      | 1,398       | −35%      |
| HDF5 packed           | 132.5      | 1,325       | −39%      |

**Key finding:** Packed achieves the smallest files. Fixed-length string columns (S24/S48)
compress better than VLEN strings stored in HDF5's global heap (which is incompressible
regardless of dataset compression settings). Compound is ~5% larger than packed (VLEN fields
for `id` and `extras_json`). At N=1,000, all HDF5 layouts are within 1% of each other —
the size advantage only materialises at scale.

---

## Summary Table

| Metric                        | Winner        | Notes                                              |
|-------------------------------|---------------|----------------------------------------------------|
| Write throughput (bs=1)       | SQLite        | HDF5 flush bottleneck; ~200× gap, all layouts      |
| Write throughput (bs=1000)    | SQLite        | Gap narrows; SQLite 4–7× faster                   |
| Query, filtered (N=10k)       | SQLite        | SQL WHERE; VLEN/compound ~2× slower                |
| Query, unfiltered (N=10k)     | HDF5 VLEN     | Contiguous chunk read beats BLOB fetch             |
| Query, any (N=1k)             | SQLite        | Small N; HDF5 open overhead visible in packed      |
| File size                     | HDF5 packed   | 39% smaller than SQLite; 15% smaller than VLEN     |

---

## Design Comparison

| Property                    | VLEN              | Compound          | Packed            |
|-----------------------------|-------------------|-------------------|-------------------|
| HDF5 datasets written/upsert| 9                 | 2                 | 6                 |
| Flush cost matters?         | Yes (dominates)   | Yes (dominates)   | Yes (dominates)   |
| String storage              | VLEN (global heap)| VLEN + S24/S48    | S24/S48 + byte buf|
| Compressible strings?       | No                | Partially         | Yes               |
| Query reconstruction cost   | Low (numpy cache) | Low (numpy cache) | Higher (buf reads)|
| Schema complexity           | Low               | Medium            | High              |
| File size at 100k           | 1,556 B/doc       | 1,398 B/doc       | 1,325 B/doc       |

---

## Recommendation

For claude-mem's hook-driven write pattern (bs=1 is realistic), **SQLite + BLOB** is the
practical choice. All three HDF5 layouts share the same flush bottleneck and are ~200× slower
at single-doc writes.

Among the HDF5 layouts: **VLEN and compound are preferred over packed** for query performance.
Compound gains 10% file size reduction over VLEN with no query penalty; use it if storage
matters. Packed saves another 5% over compound but regresses query latency significantly —
avoid it unless writes are batched externally and storage is the primary constraint.

HDF5 would be the right call if sessions stored large numerical artifacts alongside embeddings
(the original Option-B "agent session file" concept), where HDF5's hierarchical layout adds
genuine value that SQLite cannot match.

---

## Reproducing

All benchmarks are in `benchmarks/`. Run from `claude-mem-vectors/benchmarks/`:

    python bench_upsert_throughput.py
    python bench_query_latency.py
    python bench_file_size.py

Each writes `results/<name>.json`. Backends can be selected via `--backends` flag.
All three new backends (`hdf5`, `hdf5_packed`, `hdf5_compound`) are registered in `_common.py`.
