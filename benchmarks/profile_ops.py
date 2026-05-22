"""Per-operation HDF5 profiler for the schema backend.

Wraps h5py's high-level dataset/group methods to count and time every actual
read / write / resize / create / existence-check, grouped by dataset and by
phase (write vs. context-read vs. full-scan). Call counts are exact; the per-op
times include the wrapper's own overhead, so read them as relative.

Usage
-----
    python benchmarks/profile_ops.py                 # default: text+array, batch 1 & 100
    python benchmarks/profile_ops.py --n 2000
    python benchmarks/profile_ops.py --scenario text --batch 100
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from agentic_conversations_hdf5 import HDF5Session

# Reuse the benchmark's synthetic data generators for consistency.
sys.path.insert(0, str(Path(__file__).parent))
from benchmark import turn_text, turn_role, rand_embedding, rand_tool_result  # noqa: E402


# ---------------------------------------------------------------------------
# Profiler: monkeypatches h5py to record (phase, op, dataset) -> calls/time/bytes
# ---------------------------------------------------------------------------

class OpProfiler:
    def __init__(self) -> None:
        # stats[phase][(op, name)] = [calls, seconds, bytes]
        self.stats: dict[str, dict[tuple[str, str], list]] = defaultdict(
            lambda: defaultdict(lambda: [0, 0.0, 0])
        )
        self.phase = "(unprofiled)"
        self._enabled = False
        self._orig: dict = {}

    @staticmethod
    def _norm(name: str) -> str:
        """Collapse per-UUID sub-datasets into their parent group bucket."""
        parts = name.strip("/").split("/")
        last = parts[-1]
        parent = parts[-2] if len(parts) >= 2 else ""
        if parent in ("embeddings", "result_data", "artifacts"):
            return parent  # aggregate the many per-uuid datasets
        return last

    def _record(self, op: str, name: str, dt: float, nbytes: int) -> None:
        if not self._enabled:
            return
        rec = self.stats[self.phase][(op, self._norm(name))]
        rec[0] += 1
        rec[1] += dt
        rec[2] += nbytes

    def install(self) -> None:
        D, G = h5py.Dataset, h5py.Group
        self._orig = {
            "dget": D.__getitem__, "dset": D.__setitem__, "drsz": D.resize,
            "gcreate": G.create_dataset, "gcontains": G.__contains__,
        }
        prof = self

        def dget(self, key):
            t = time.perf_counter()
            r = prof._orig["dget"](self, key)
            dt = time.perf_counter() - t
            prof._record("read", self.name, dt, getattr(r, "nbytes", 0))
            return r

        def dset(self, key, val):
            t = time.perf_counter()
            r = prof._orig["dset"](self, key, val)
            dt = time.perf_counter() - t
            prof._record("write", self.name, dt, getattr(val, "nbytes", 0))
            return r

        def drsz(self, *a, **k):
            t = time.perf_counter()
            r = prof._orig["drsz"](self, *a, **k)
            prof._record("resize", self.name, time.perf_counter() - t, 0)
            return r

        def gcreate(self, name, *a, **k):
            t = time.perf_counter()
            r = prof._orig["gcreate"](self, name, *a, **k)
            dt = time.perf_counter() - t
            prof._record("create", f"{self.name}/{name}", dt,
                         getattr(r, "nbytes", 0))
            return r

        def gcontains(self, name):
            t = time.perf_counter()
            r = prof._orig["gcontains"](self, name)
            prof._record("exists", self.name or "/", time.perf_counter() - t, 0)
            return r

        D.__getitem__, D.__setitem__, D.resize = dget, dset, drsz
        G.create_dataset, G.__contains__ = gcreate, gcontains

    def uninstall(self) -> None:
        D, G = h5py.Dataset, h5py.Group
        D.__getitem__ = self._orig["dget"]
        D.__setitem__ = self._orig["dset"]
        D.resize = self._orig["drsz"]
        G.create_dataset = self._orig["gcreate"]
        G.__contains__ = self._orig["gcontains"]

    @contextmanager
    def profile(self, phase: str):
        self.phase, self._enabled = phase, True
        try:
            yield
        finally:
            self._enabled = False

    def print_phase(self, phase: str, header: str) -> None:
        rows = self.stats.get(phase, {})
        print(f"\n  {header}")
        print(f"    {'op':<22} {'calls':>8} {'total ms':>10} {'MB':>8}")
        print(f"    {'-'*22} {'-'*8} {'-'*10} {'-'*8}")
        # sort by total time descending
        for (op, name), (calls, secs, nbytes) in sorted(
            rows.items(), key=lambda kv: -kv[1][1]
        ):
            print(f"    {op + ':' + name:<22} {calls:>8,} "
                  f"{secs*1e3:>10.2f} {nbytes/1e6:>8.2f}")
        tot_calls = sum(v[0] for v in rows.values())
        tot_ms = sum(v[1] for v in rows.values()) * 1e3
        print(f"    {'TOTAL':<22} {tot_calls:>8,} {tot_ms:>10.2f}")


# ---------------------------------------------------------------------------
# Profiled run
# ---------------------------------------------------------------------------

def run(scenario: str, batch: int, n: int, store_root: Path) -> None:
    prof = OpProfiler()
    prof.install()
    try:
        sp = store_root / f"prof_{scenario}_b{batch}_{n}.h5"
        if sp.exists():
            sp.unlink()
        with_arrays = scenario == "data"

        # ---- WRITE phase ----
        sess = HDF5Session(sp, session_id="prof", batch_size=batch)
        with prof.profile("write"):
            for i in range(n):
                emb = rand_embedding() if with_arrays else None
                sess.add_turn(turn_role(i), turn_text(i), embedding=emb)
                if with_arrays:
                    sess.add_tool_call(
                        name="analyse", args={"turn": i},
                        result_text=f"{i%7} anomalies",
                        result_data=rand_tool_result(),
                    )
            # flush the trailing partial batch inside the write phase
            sess._flush_msg_buffer()
        sess.close()

        # ---- CONTEXT-READ phase (cold open + last-20) ----
        sess = HDF5Session(sp, session_id="prof", batch_size=batch)
        with prof.profile("ctx_read"):
            _ = sess.get_recent_context(20)
        sess.close()

        # ---- FULL-SCAN read phase ----
        sess = HDF5Session(sp, session_id="prof", batch_size=batch)
        with prof.profile("full_scan"):
            _ = sess.get_recent_context(n)
        sess.close()
    finally:
        prof.uninstall()

    label = "text-only" if scenario == "text" else "array-heavy"
    print(f"\n{'='*60}\n[{label}, batch={batch}, N={n:,}]\n{'='*60}")
    prof.print_phase("write", f"WRITE phase (N={n:,})")
    prof.print_phase("ctx_read", "CONTEXT-READ phase (last 20)")
    prof.print_phase("full_scan", f"FULL-SCAN phase (all {n:,})")


def parse_args():
    p = argparse.ArgumentParser(description="Per-op HDF5 profiler (the backend)")
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--scenario", choices=["text", "data", "both"], default="both")
    p.add_argument("--batch", type=int, nargs="+", default=[1, 100])
    p.add_argument("--outdir", type=Path, default=Path("results/profile"))
    return p.parse_args()


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    scenarios = ["text", "data"] if args.scenario == "both" else [args.scenario]
    for scenario in scenarios:
        for batch in args.batch:
            run(scenario, batch, args.n, args.outdir)


if __name__ == "__main__":
    main()
