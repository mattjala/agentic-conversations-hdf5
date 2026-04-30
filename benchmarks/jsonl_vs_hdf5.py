"""Compare raw Claude Code JSONL logs against the converted HDF5 form.

Metrics:
    1. on-disk size (raw bytes, gzipped, hdf5)
    2. cold "load last N messages" latency
    3. cold "sum total tokens used in this session" latency

The point: HDF5 wins (1) on long sessions and (2,3) by a lot, because
JSONL forces a full-file scan with per-line JSON.parse where HDF5 is a
single chunked read with no parsing.

Usage:
    python benchmarks/jsonl_vs_hdf5.py [--n-context 20]
                                       [--logs ~/.claude/projects/.../*.jsonl]
                                       [--limit 10]
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentic_conversations_hdf5 import HDF5Session
from agentic_conversations_hdf5.convert import convert_jsonl


def _file_size(p: Path) -> int:
    return p.stat().st_size if p.exists() else 0


def _gzip_size(p: Path) -> int:
    with open(p, "rb") as f_in:
        data = f_in.read()
    return len(gzip.compress(data, compresslevel=4))


def _jsonl_tail(p: Path, n: int) -> list[dict]:
    msgs: list[dict] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") in {"user", "assistant", "summary"}:
                msgs.append(rec)
    return msgs[-n:]


def _jsonl_total_usage(p: Path) -> dict[str, int]:
    totals = {
        "input_tokens": 0, "output_tokens": 0,
        "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
    }
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") != "assistant":
                continue
            usage = (rec.get("message") or {}).get("usage") or {}
            for k in totals:
                v = usage.get(k)
                if v is not None:
                    totals[k] += int(v)
    return totals


def _hdf5_tail(p: Path, sid: str, n: int) -> list[dict]:
    sess = HDF5Session(p, session_id=sid, mode="r")
    try:
        return sess.get_recent_context(n)
    finally:
        sess.close()


def _hdf5_total_usage(p: Path, sid: str) -> dict[str, int]:
    sess = HDF5Session(p, session_id=sid, mode="r")
    try:
        return sess.total_usage()
    finally:
        sess.close()


def _time(fn, repeats: int = 3) -> float:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--logs", nargs="*", type=Path,
                   help="JSONL files to benchmark (default: pick from ~/.claude/projects)")
    p.add_argument("--limit", type=int, default=8,
                   help="how many JSONL files to use if --logs is omitted")
    p.add_argument("--n-context", type=int, default=20)
    p.add_argument("--outdir", type=Path, default=Path("results"))
    return p.parse_args()


def find_logs(limit: int) -> list[Path]:
    root = Path.home() / ".claude" / "projects"
    if not root.exists():
        return []
    candidates = sorted(root.rglob("*.jsonl"), key=_file_size, reverse=True)
    return candidates[:limit]


def main() -> int:
    args = parse_args()
    logs = args.logs or find_logs(args.limit)
    if not logs:
        print("no JSONL logs found", file=sys.stderr)
        return 1

    args.outdir.mkdir(parents=True, exist_ok=True)
    work = args.outdir / "hdf5_out"
    work.mkdir(exist_ok=True)

    print(f"{'session':<40} {'jsonl kB':>10} {'gz kB':>8} {'h5 kB':>8} "
          f"{'tail jl ms':>11} {'tail h5 ms':>11} "
          f"{'usage jl ms':>12} {'usage h5 ms':>12}")
    print("-" * 120)

    rows = []
    for src in logs:
        h5p = work / (src.stem + ".h5")
        if h5p.exists():
            h5p.unlink()
        convert_jsonl(src, h5p, session_id=src.stem)

        jsonl_kb = _file_size(src) / 1e3
        gz_kb = _gzip_size(src) / 1e3
        h5_kb = _file_size(h5p) / 1e3

        tail_jl = _time(lambda: _jsonl_tail(src, args.n_context)) * 1000
        tail_h5 = _time(lambda: _hdf5_tail(h5p, src.stem, args.n_context)) * 1000

        usage_jl = _time(lambda: _jsonl_total_usage(src)) * 1000
        usage_h5 = _time(lambda: _hdf5_total_usage(h5p, src.stem)) * 1000

        name = src.stem[:38]
        print(f"{name:<40} {jsonl_kb:>10.1f} {gz_kb:>8.1f} {h5_kb:>8.1f} "
              f"{tail_jl:>11.2f} {tail_h5:>11.2f} "
              f"{usage_jl:>12.2f} {usage_h5:>12.2f}")
        rows.append({
            "session": src.stem,
            "jsonl_bytes": _file_size(src),
            "gzip_bytes": int(gz_kb * 1000),
            "hdf5_bytes": _file_size(h5p),
            "tail_jsonl_ms": tail_jl,
            "tail_hdf5_ms": tail_h5,
            "usage_jsonl_ms": usage_jl,
            "usage_hdf5_ms": usage_h5,
        })

    out = args.outdir / "jsonl_vs_hdf5.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nraw data: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
