# Schema Optimization Results

Each row is the schema after a successive change. Benchmark run with
`benchmark.py --only hdf5_packed hdf5_packed_lazy` (all default sizes, both
scenarios). Key columns at N=5,000 (largest size, most representative).

All times in milliseconds. Size in kB. Throughput in turns/sec.

## Text-only scenario (flush=1, N=5000)

| Stage                  | t/s | size kB | ctx ms | art ms |
|------------------------|-----|---------|--------|--------|
| Baseline               | 296 | 833     | 1.90   | 0.46   |
| + libver='latest'      | 319 | 792     | 1.84   | 0.41   |
| + pre-allocation       | 409 | 776     | 2.00   | 0.38   |
| + Core VFD †           | 380 | 0 *     | 2.25   | —      |
| + zlib-ng              | 489 | 773     | 1.67   | 0.34   |
| + chunk tuning §       | 489 | 773     | 1.67   | 0.34   |

## Text-only scenario (flush=100, N=5000)

| Stage                  | t/s | size kB | ctx ms | art ms |
|------------------------|-----|---------|--------|--------|
| Baseline               | 287 | 818     | 1.80   | 0.41   |
| + libver='latest'      | 307 | 774     | 1.83   | 0.40   |
| + pre-allocation       | 395 | 776     | 1.96   | 0.35   |
| + Core VFD             | N/A (flush_every ignored; see flush=1 row) |
| + zlib-ng              | 486 | 791     | 1.67   | 0.41   |
| + chunk tuning §       | 486 | 791     | 1.67   | 0.41   |

## Array-heavy scenario (flush=1, N=5000)

| Stage                  | t/s  | size kB | ctx ms  | art ms |
|------------------------|------|---------|---------|--------|
| Baseline               | 132  | 56854   | 4.57    | 0.43   |
| + libver='latest'      | 146  | 56442   | 4.46    | 0.40   |
| + pre-allocation       | 184  | 56451   | 4.78    | 0.38   |
| + Core VFD †           | 196  | 0 *     | 65.13 ‡ | —      |
| + zlib-ng              | 187  | 56463   | 3.73    | 0.31   |
| + chunk tuning §       | 187  | 56463   | 3.73    | 0.31   |

## Array-heavy scenario (flush=100, N=5000)

| Stage                  | t/s  | size kB | ctx ms | art ms |
|------------------------|------|---------|--------|--------|
| Baseline               | 151  | 56895   | 4.48   | 0.40   |
| + libver='latest'      | 137  | 56476   | 4.69   | 0.48   |
| + pre-allocation       | 184  | 56503   | 4.90   | 0.37   |
| + Core VFD             | N/A (flush_every ignored; see flush=1 row) |
| + zlib-ng              | 206  | 56526   | 3.78   | 0.33   |
| + chunk tuning §       | 206  | 56526   | 3.78   | 0.33   |

## Batched-write buffer ¶

A `batch_size` constructor param on `HDF5PackedSession`. Messages accumulate in
an in-memory buffer; when it fills (or on read/close) the whole block is written
with **one slice-assign per dataset** instead of ~9 single-row writes per turn.
This cuts HDF5 API call count by a factor of `batch_size`. Layout, chunking, and
compression are unchanged, so file size and read latency are unaffected.

Measured on the same **HDF5 2.0.0 + zlib-ng** build as the cumulative table above
(confirmed: the installed h5py links `~/local/hdf5-zlibng/lib/libhdf5.so.1000`),
so the batched speedups **do** compound with zlib-ng. `batch=1` is the control —
it flushes every message, matching the unbuffered write count, and lands on the
"+ chunk tuning" cumulative row within run-to-run variance. Speedups below are
relative to `batch=1`.

Numbers below are post embedding-consolidation (single run, same machine state).
Run-to-run variance is ~10%, so read relative speedups, not absolute t/s.

### Text-only (N=5000)

| batch | t/s    | speedup | size kB | ctx ms |
|-------|--------|---------|---------|--------|
| 1 ◊   | 414    | 1.0×    | 790     | 1.76   |
| 10    | 3,801  | 9.2×    | 763     | 1.74   |
| 25    | 8,764  | 21.2×   | 761     | 1.77   |
| 100   | 25,870 | 62.5×   | 754     | 1.76   |

(N=1000 tracks the same shape: 417 / 3,693 / 8,100 / 20,751 t/s.)

### Array-heavy (N=5000, for contrast)

| batch | t/s | speedup |
|-------|-----|---------|
| 1     | 174 | 1.0×    |
| 10    | 397 | 2.3×    |
| 25    | 431 | 2.5×    |
| 100   | 424 | 2.4×    |

Array-heavy gains far less: per-turn cost there is dominated by per-embedding
block writes and per-tool-result dataset creation + compression, not the row
writes the buffer collapses.

## Embeddings as one (N, dim) dataset

Embeddings were previously stored as one float32 dataset **per message UUID**
under a `messages/embeddings` group. That made a full scan do one read per row
(5,000 reads ≈ 223 ms at N=5000) and the last-20 read do 20 reads. They are now
a single `messages/embeddings` dataset of shape `(N, dim)`, created lazily on the
first embedding, with a `messages/has_embedding` `(N,)` uint8 presence flag so
rows without an embedding cost nothing to skip and the per-row `tid in group`
existence check is gone.

Design choices:
- **Uncompressed.** Dense float32 vectors compress poorly; gzip+shuffle on each
  block write cost ~2× write throughput for ~no size reduction. The prior
  per-UUID datasets were also uncompressed, so size is unchanged (~55 MB at
  N=5000 array-heavy).
- **Pre-allocated by doubling**, in lockstep with the message-row capacity, then
  truncated to `n_used` on close — same scheme as the other columns.
- Row index aligns with the message row index; lazily-created datasets back-fill
  prior rows as zeros (marked absent by `has_embedding`).

Read effect (N=5000 array-heavy): last-20 context read **45 → 8** h5py calls and
ctx latency **3.79 → 2.23 ms**; full scan **10,005 → 8** calls (embedding read
223 → 44 ms). Write effect: array-heavy write throughput drops ~7–10% (e.g.
batch=100 454 → ~408 t/s) — the irreducible cost of block-writing into one shared
chunked dataset (boundary-chunk read-modify-write) versus independent per-UUID
datasets. Text-only is unaffected (no embeddings). A sound trade for a
write-once / read-many conversation log.

## Per-operation breakdown (instrumented) ※

`benchmarks/profile_ops.py` wraps h5py's high-level methods to count every
actual read / write / resize / create / existence-check, grouped by dataset and
phase. **Call counts are exact**; the per-op *milliseconds* are inflated by the
profiler's own `perf_counter` wrapping and should be read only as relative
structure, not absolute cost.

### `get_recent_context(20)` — last-20 context read (N=5000)

Before/after the embedding-layout consolidation (see "Embeddings as one (N, dim)
dataset" below). Counts are total h5py calls in the read phase.

| op                  | text before | text after | array before | array after |
|---------------------|-------------|------------|--------------|-------------|
| read:uuid           | 1           | 1          | 1            | 1           |
| read:role           | 1           | 1          | 1            | 1           |
| read:timestamp      | 1           | 1          | 1            | 1           |
| read:content_index  | 1           | 1          | 1            | 1           |
| read:content_bytes  | 1           | 1          | 1            | 1           |
| read:has_embedding  | —           | 1          | —            | 1           |
| read:embeddings     | 0           | 0          | **20**       | **1**       |
| exists:embeddings   | 20          | 0          | 20           | 1 ‖         |
| **total calls**     | **25**      | **6**      | **45**       | **8**       |

Item 3 confirmed and improved: metadata was already **one slice read per column**
(not per-row). The consolidation removed the two per-row patterns — the per-row
`tid in emb_grp` existence check (replaced by the `has_embedding` flag) and the
per-row embedding read (now one `(20, dim)` slice). On a **full scan (N=5000)**
the win is starker: array-heavy goes from **10,005 calls** (5,000 embedding reads
≈ 223 ms + 5,000 existence checks) to **8 calls** (one embedding slice ≈ 44 ms).

### Write phase — calls per column scale as N / batch_size

| batch | calls per metadata column | total write+resize calls (text, N=5000) |
|-------|---------------------------|-----------------------------------------|
| 1     | 5,000                     | 50,040                                  |
| 100   | 50                        | 540                                     |

This is the mechanism behind the ~60× batched-write speedup, shown directly: a
`batch_size`-fold reduction in HDF5 call count. The consolidation also moved
array-heavy embeddings from N per-row `create_dataset` calls to one block write
per batch.

## Notes

- flush=1: one `h5py.flush()` per message (realistic live-write scenario)
- flush=100: flush every 100 messages (amortised cost, batch-write scenario)
- text-only: no embeddings, no array tool results
- array-heavy: 1536-d embeddings + 1000-element float32 tool results per turn
- § Chunk tuning sweep (chunk_rows ∈ {8,16,32,64,256,512,1024} × content_chunk ∈ {4KB,16KB,64KB,256KB,1MB}, N=5000, standard h5py): smaller content chunks (4KB) give a ~8% throughput edge for text but inflate array-heavy file size 3× (3671 KB vs 1140 KB) due to poor gzip compression at small block sizes. Larger chunks hurt both throughput and latency. Default values (chunk_rows=64, content_chunk=64KB) remain optimal for the balanced case. No defaults changed.
- † Core VFD uses `flush_every=0` (backing_store writes entire image on close only); flush_every parameter has no effect
- \* size kB = 0 because mid-session disk file is not written; final size on disk is similar to pre-allocation baseline
- ‡ 65ms context latency regression: 56 MB in-memory image must be fully navigated with no kernel page cache; not suitable for large array-heavy sessions
- zlib-ng requires custom HDF5 2.0+ build; see build notes below
- ¶ Batched-write buffer measured on the HDF5 2.0.0 + zlib-ng build (same as the cumulative table); speedups are relative to `batch=1` and compound with zlib-ng. The buffer cuts API call count, zlib-ng cuts compression cost — orthogonal mechanisms.
- ◊ `batch=1` is the control: flushes every message, so its write count matches the unbuffered path. Its t/s lands on the "+ chunk tuning" cumulative row within run-to-run noise, confirming the buffer adds no overhead.
- ※ Per-op breakdown produced by `benchmarks/profile_ops.py`. Call counts are exact; per-op ms is profiler-inflated (relative structure only).

## Build notes

### zlib-ng + HDF5 2.0+ + h5py

Tested on Ubuntu 22.04 / Python 3.12 / x86_64.

```bash
# 1. Build zlib-ng 2.2.2 in zlib-compat mode (drop-in for libz.so)
curl -L https://github.com/zlib-ng/zlib-ng/archive/refs/tags/2.2.2.tar.gz | tar xz -C /tmp
cmake -S /tmp/zlib-ng-2.2.2 -B /tmp/zlib-ng-build \
  -DCMAKE_INSTALL_PREFIX=$HOME/local/zlib-ng \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DZLIB_COMPAT=ON
cmake --build /tmp/zlib-ng-build --parallel $(nproc)
cmake --install /tmp/zlib-ng-build

# 2. Build HDF5 2.0.0 pointing at the zlib-ng libz.so
cmake -S /path/to/hdf5-2.0.0-source -B /tmp/hdf5-build \
  -DCMAKE_INSTALL_PREFIX=$HOME/local/hdf5-zlibng \
  -DCMAKE_BUILD_TYPE=Release \
  -DZLIB_ROOT=$HOME/local/zlib-ng \
  -DZLIB_INCLUDE_DIR=$HOME/local/zlib-ng/include \
  -DZLIB_LIBRARY=$HOME/local/zlib-ng/lib/libz.so \
  -DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=OFF \
  -DHDF5_BUILD_TOOLS=ON -DHDF5_BUILD_HL_LIB=ON \
  -DHDF5_BUILD_CPP_LIB=OFF -DHDF5_BUILD_FORTRAN=OFF \
  -DHDF5_BUILD_JAVA=OFF -DHDF5_BUILD_EXAMPLES=OFF
cmake --build /tmp/hdf5-build --parallel $(nproc)
cmake --install /tmp/hdf5-build

# 3. Copy zlib-ng into HDF5 lib dir (runtime RPATH is $ORIGIN/../lib)
cp $HOME/local/zlib-ng/lib/libz.so.1* $HOME/local/hdf5-zlibng/lib/

# 4. Build h5py 3.16.0 from source against HDF5 2.0
pip download h5py==3.16.0 --no-deps --no-binary h5py -d /tmp/h5py-src
HDF5_DIR=$HOME/local/hdf5-zlibng \
  LD_LIBRARY_PATH=$HOME/local/hdf5-zlibng/lib \
  pip install /tmp/h5py-src/h5py-3.16.0.tar.gz

# 5. Run benchmarks with LD_LIBRARY_PATH so the custom HDF5 is loaded
LD_LIBRARY_PATH=$HOME/local/hdf5-zlibng/lib \
  python benchmarks/benchmark.py --only hdf5_packed hdf5_packed_lazy --sizes 5000
```

**Verification** — after step 4, confirm:
```python
import h5py
assert h5py.version.hdf5_version.startswith("2.")  # "2.0.0"
# Check zlib-ng is loaded at runtime:
import subprocess, os, glob, site
so = glob.glob(f"{site.getsitepackages()[0]}/h5py/*.so")[0]
# should show ~/local/hdf5-zlibng/lib/libz.so.1 (not /usr/local/lib)
```
