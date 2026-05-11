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

from agentic_conversations_hdf5 import HDF5Session, HDF5PackedSession, HDF5CompoundSession
from agentic_conversations_hdf5.convert import convert_jsonl
from agentic_conversations_hdf5.convert_packed import convert_jsonl_packed
from agentic_conversations_hdf5.convert_compound import convert_jsonl_compound


def _file_size(p: Path) -> int:
    return p.stat().st_size if p.exists() else 0


def _gzip_size(p: Path) -> int:
    with open(p, "rb") as f_in:
        data = f_in.read()
    return len(gzip.compress(data, compresslevel=4))


def _write_gzip(src: Path, dest: Path) -> None:
    with open(src, "rb") as f_in, gzip.open(dest, "wb", compresslevel=4) as f_out:
        f_out.write(f_in.read())


def _gzip_tail(gz_path: Path, n: int) -> list[dict]:
    msgs: list[dict] = []
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
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


def _gzip_total_usage(gz_path: Path) -> dict[str, int]:
    totals = {
        "input_tokens": 0, "output_tokens": 0,
        "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
    }
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
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


def _packed_tail(p: Path, sid: str, n: int) -> list[dict]:
    sess = HDF5PackedSession(p, session_id=sid, mode="r")
    try:
        return sess.get_recent_context(n)
    finally:
        sess.close()


def _packed_total_usage(p: Path, sid: str) -> dict[str, int]:
    sess = HDF5PackedSession(p, session_id=sid, mode="r")
    try:
        return sess.total_usage()
    finally:
        sess.close()


def _compound_tail(p: Path, sid: str, n: int) -> list[dict]:
    sess = HDF5CompoundSession(p, session_id=sid, mode="r")
    try:
        return sess.get_recent_context(n)
    finally:
        sess.close()


def _compound_total_usage(p: Path, sid: str) -> dict[str, int]:
    sess = HDF5CompoundSession(p, session_id=sid, mode="r")
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

    print(f"{'session':<36} {'jl kB':>7} {'gz kB':>7} {'h5 kB':>7} {'pk kB':>7} {'cp kB':>7} "
          f"{'tail jl':>8} {'tail gz':>8} {'tail h5':>8} {'tail pk':>8} {'tail cp':>8} "
          f"{'use jl':>8} {'use gz':>8} {'use h5':>8} {'use pk':>8} {'use cp':>8}")
    print("-" * 180)

    rows = []
    for src in logs:
        h5p  = work / (src.stem + ".h5")
        pkp  = work / (src.stem + "_packed.h5")
        cpp  = work / (src.stem + "_compound.h5")
        gzp  = work / (src.stem + ".jsonl.gz")
        for p in (h5p, pkp, cpp):
            if p.exists():
                p.unlink()
        convert_jsonl(src, h5p, session_id=src.stem)
        convert_jsonl_packed(src, pkp, session_id=src.stem)
        convert_jsonl_compound(src, cpp, session_id=src.stem)
        _write_gzip(src, gzp)

        jsonl_kb = _file_size(src) / 1e3
        gz_kb    = _file_size(gzp) / 1e3
        h5_kb    = _file_size(h5p) / 1e3
        pk_kb    = _file_size(pkp) / 1e3
        cp_kb    = _file_size(cpp) / 1e3

        tail_jl = _time(lambda: _jsonl_tail(src, args.n_context)) * 1000
        tail_gz = _time(lambda: _gzip_tail(gzp, args.n_context)) * 1000
        tail_h5 = _time(lambda: _hdf5_tail(h5p, src.stem, args.n_context)) * 1000
        tail_pk = _time(lambda: _packed_tail(pkp, src.stem, args.n_context)) * 1000
        tail_cp = _time(lambda: _compound_tail(cpp, src.stem, args.n_context)) * 1000

        usage_jl = _time(lambda: _jsonl_total_usage(src)) * 1000
        usage_gz = _time(lambda: _gzip_total_usage(gzp)) * 1000
        usage_h5 = _time(lambda: _hdf5_total_usage(h5p, src.stem)) * 1000
        usage_pk = _time(lambda: _packed_total_usage(pkp, src.stem)) * 1000
        usage_cp = _time(lambda: _compound_total_usage(cpp, src.stem)) * 1000

        name = src.stem[:34]
        print(f"{name:<36} {jsonl_kb:>7.1f} {gz_kb:>7.1f} {h5_kb:>7.1f} {pk_kb:>7.1f} {cp_kb:>7.1f} "
              f"{tail_jl:>8.2f} {tail_gz:>8.2f} {tail_h5:>8.2f} {tail_pk:>8.2f} {tail_cp:>8.2f} "
              f"{usage_jl:>8.2f} {usage_gz:>8.2f} {usage_h5:>8.2f} {usage_pk:>8.2f} {usage_cp:>8.2f}")
        rows.append({
            "session":           src.stem,
            "jsonl_bytes":       _file_size(src),
            "gzip_bytes":        _file_size(gzp),
            "hdf5_bytes":        _file_size(h5p),
            "packed_bytes":      _file_size(pkp),
            "compound_bytes":    _file_size(cpp),
            "tail_jsonl_ms":     tail_jl,
            "tail_gzip_ms":      tail_gz,
            "tail_hdf5_ms":      tail_h5,
            "tail_packed_ms":    tail_pk,
            "tail_compound_ms":  tail_cp,
            "usage_jsonl_ms":    usage_jl,
            "usage_gzip_ms":     usage_gz,
            "usage_hdf5_ms":     usage_h5,
            "usage_packed_ms":   usage_pk,
            "usage_compound_ms": usage_cp,
        })

    out = args.outdir / "jsonl_vs_hdf5.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nraw data: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
