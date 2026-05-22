"""Command-line entry point: agentic-conversations-hdf5."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .convert import convert_jsonl, convert_many

_HOOK_CMD = "python3 -m agentic_conversations_hdf5.hooks.live_session"
_HOOK_EVENTS = ("UserPromptSubmit", "Stop")


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


def _cmd_setup_hook(args: argparse.Namespace) -> int:
    settings_path = Path("~/.claude/settings.json").expanduser()
    output_dir = Path(args.output_dir).expanduser()

    settings: dict = {}
    if settings_path.exists():
        with open(settings_path) as f:
            settings = json.load(f)

    hooks = settings.setdefault("hooks", {})
    hook_entry = {"type": "command", "command": _HOOK_CMD}
    if args.output_dir != "~/.claude/hdf5-sessions":
        hook_entry["command"] = (
            f"AGENTIC_HDF5_DIR={args.output_dir} {_HOOK_CMD}"
        )

    added = []
    for event in _HOOK_EVENTS:
        entries = hooks.setdefault(event, [])
        cmds = [h.get("command", "") for e in entries for h in e.get("hooks", [])]
        if not any("agentic_conversations_hdf5" in c for c in cmds):
            entries.append({"hooks": [hook_entry]})
            added.append(event)

    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=2)
        f.write("\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    if added:
        print(f"Hook registered for: {', '.join(added)}")
    else:
        print("Hook already registered (no changes made).")
    print(f"HDF5 session files will be written to: {output_dir}")
    print()
    print("To inspect a live session:")
    print("  agentic-conversations-hdf5 inspect <path-to-.h5>")
    return 0


def _cmd_teardown_hook(args: argparse.Namespace) -> int:
    settings_path = Path("~/.claude/settings.json").expanduser()
    if not settings_path.exists():
        print("No settings.json found.")
        return 0

    with open(settings_path) as f:
        settings = json.load(f)

    hooks = settings.get("hooks", {})
    removed = []
    for event in _HOOK_EVENTS:
        if event not in hooks:
            continue
        before = len(hooks[event])
        hooks[event] = [
            e for e in hooks[event]
            if not any(
                "agentic_conversations_hdf5" in h.get("command", "")
                for h in e.get("hooks", [])
            )
        ]
        if len(hooks[event]) < before:
            removed.append(event)
        if not hooks[event]:
            del hooks[event]
    if not hooks:
        del settings["hooks"]

    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=2)
        f.write("\n")

    if removed:
        print(f"Hook removed from: {', '.join(removed)}")
    else:
        print("Hook not found (no changes made).")
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

    p_setup = sub.add_parser(
        "setup-hook",
        help="register live-session hook in ~/.claude/settings.json",
    )
    p_setup.add_argument(
        "--output-dir",
        default="~/.claude/hdf5-sessions",
        help="directory for HDF5 session files (default: ~/.claude/hdf5-sessions)",
    )
    p_setup.set_defaults(func=_cmd_setup_hook)

    p_tear = sub.add_parser(
        "teardown-hook",
        help="remove live-session hook from ~/.claude/settings.json",
    )
    p_tear.set_defaults(func=_cmd_teardown_hook)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
