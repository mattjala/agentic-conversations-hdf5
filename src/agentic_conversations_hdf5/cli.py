"""Command-line entry point: agentic-conversations-hdf5."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .convert import convert_jsonl, convert_many
from .backends.hdf5 import HDF5Session


def _cmd_convert(args: argparse.Namespace) -> int:
    inputs = [Path(p) for p in args.input]
    if not inputs:
        print("error: no input files", file=sys.stderr)
        return 2
    output = Path(args.output)
    if len(inputs) == 1:
        counts = convert_jsonl(
            inputs[0], output,
            session_id=args.session_id,
            overwrite=args.overwrite,
        )
        print(json.dumps({inputs[0].name: counts}, indent=2))
    else:
        if args.session_id:
            print("error: --session-id only valid with a single input",
                  file=sys.stderr)
            return 2
        counts = convert_many(inputs, output, overwrite=args.overwrite)
        print(json.dumps(counts, indent=2))
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    import h5py
    with h5py.File(args.path, "r") as f:
        info: dict = {
            "schema_version": int(f.attrs.get("schema_version", 0)),
            "source": str(f.attrs.get("source", "")),
            "sessions": {},
        }
        if "sessions" not in f:
            print(json.dumps(info, indent=2))
            return 0
        for sid in f["sessions"]:
            sg = f[f"sessions/{sid}"]
            n_msgs = sg["messages/uuid"].shape[0] if "messages" in sg else 0
            n_tools = (sg["tool_calls/tool_use_id"].shape[0]
                       if "tool_calls" in sg else 0)
            usage = sg["messages/usage"][:] if n_msgs else None
            tokens = (
                {f: int(usage[f].sum()) for f in usage.dtype.names}
                if usage is not None else {}
            )
            info["sessions"][sid] = {
                "messages": int(n_msgs),
                "tool_calls": int(n_tools),
                "model": str(sg.attrs.get("model", "")),
                "cwd": str(sg.attrs.get("cwd", "")),
                "git_branch": str(sg.attrs.get("git_branch", "")),
                "agent_version": str(sg.attrs.get("agent_version", "")),
                "tokens": tokens,
            }
    print(json.dumps(info, indent=2))
    return 0


def _cmd_tail(args: argparse.Namespace) -> int:
    sess = HDF5Session(args.path, session_id=args.session_id, mode="r")
    try:
        ctx = sess.get_recent_context(args.n)
        for turn in ctx:
            role = turn["role"] or turn.get("type", "")
            content = (turn["content"] or "").replace("\n", " ")
            if len(content) > 200:
                content = content[:200] + "..."
            print(f"[{role}] {content}")
    finally:
        sess.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="agentic-conversations-hdf5")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_conv = sub.add_parser("convert", help="convert Claude Code JSONL to HDF5")
    p_conv.add_argument("input", nargs="+", help="JSONL file(s) to convert")
    p_conv.add_argument("-o", "--output", required=True, help="output .h5 path")
    p_conv.add_argument("--session-id", default=None,
                        help="override session id (single input only)")
    p_conv.add_argument("--overwrite", action="store_true")
    p_conv.set_defaults(func=_cmd_convert)

    p_ins = sub.add_parser("inspect", help="summarise an HDF5 session file")
    p_ins.add_argument("path")
    p_ins.set_defaults(func=_cmd_inspect)

    p_tail = sub.add_parser("tail", help="print the last N messages of a session")
    p_tail.add_argument("path")
    p_tail.add_argument("session_id")
    p_tail.add_argument("-n", type=int, default=20)
    p_tail.set_defaults(func=_cmd_tail)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
