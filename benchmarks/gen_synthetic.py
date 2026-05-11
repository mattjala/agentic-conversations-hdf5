"""Generate synthetic Claude Code JSONL session files for benchmarking.

Usage
-----
    python benchmarks/gen_synthetic.py                     # default: 10x and 100x
    python benchmarks/gen_synthetic.py --sizes 5 50        # custom MB targets
    python benchmarks/gen_synthetic.py --outdir /tmp/data  # custom output dir
    python benchmarks/gen_synthetic.py --seed 7            # reproducible output

Defaults produce two files in tests/fixtures/:
  synthetic_10x.jsonl   ~25 MB  (≈10× the median real session of ~2.5 MB)
  synthetic_100x.jsonl  ~250 MB (≈100× the median real session)

The format exactly mirrors real Claude Code session files:
  - All line types: user, assistant, system, file-history-snapshot,
    attachment, last-prompt, queue-operation, permission-mode
  - Realistic assistant messages: thinking blocks + text + tool_use
  - Realistic user messages: plain text prompts + tool_result payloads
  - Proper UUID chaining via parentUuid
  - Realistic token-usage numbers and cache fields
"""

from __future__ import annotations

import argparse
import json
import random
import textwrap
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Seed for reproducibility
# ---------------------------------------------------------------------------
random.seed(42)

# ---------------------------------------------------------------------------
# Content libraries
# ---------------------------------------------------------------------------

USER_PROMPTS = [
    "Can you look at the HDF5 converter and add support for compound datatypes?",
    "Run the benchmark suite and summarise the results.",
    "The packed backend is crashing when chunk_size is larger than the dataset. Fix it.",
    "Read the schema file and explain the design decisions.",
    "Add a progress bar to the conversion loop.",
    "The tests are failing on Python 3.12. Debug and fix.",
    "Refactor convert.py to reduce duplication between the three backends.",
    "What's the on-disk size ratio of HDF5 vs gzipped JSONL for a 10 MB session?",
    "Add type annotations throughout the backends package.",
    "Write a helper that streams messages from an HDF5 file without loading all of them.",
    "Check if the session_id is stored correctly after a round-trip.",
    "Profile the hot path in _jsonl_tail and optimise it.",
    "Update pyproject.toml to add h5py as a required dependency.",
    "The benchmark is not picking up new JSONL files. Fix the glob pattern.",
    "Add a --dry-run flag that reports what would be converted without writing.",
]

THINKING_SNIPPETS = [
    "Let me think through the best approach here. The compound datatype backend uses H5T_COMPOUND "
    "which lets us store each message as a struct. I need to check whether h5py exposes this via "
    "numpy structured arrays or via the low-level API. Looking at the existing code in "
    "hdf5_compound.py, it seems to use numpy dtypes, which map cleanly to H5T_COMPOUND.",
    "The crash is likely caused by the chunk cache being smaller than the requested chunk. "
    "h5py sets chunk_cache_mem_size at dataset creation time, and if the chunk dimensions "
    "exceed the cache, reads fall back to direct I/O which can surface a different code path. "
    "I should check what chunk_size values are being passed and whether there's a guard.",
    "Looking at the benchmark results, HDF5 wins on random-access latency by ~40× but "
    "loses on sequential full-file reads for very small sessions (<50 KB) because HDF5 "
    "metadata overhead dominates. I should note that in the summary.",
    "The test failure on 3.12 is probably a match-statement syntax issue or a changed "
    "default in the struct module. Let me read the traceback first before guessing.",
    "Refactoring the three backends: convert.py, convert_packed.py, convert_compound.py "
    "all share the same JSONL parsing loop. I can extract that into a shared generator "
    "in a new file, then have each backend call it.",
]

FILE_CONTENTS = {
    "src/agentic_conversations_hdf5/convert.py": textwrap.dedent("""\
        \"\"\"Convert Claude Code JSONL session files to HDF5 (flat-array backend).\"\"\"
        from __future__ import annotations
        import json
        from pathlib import Path
        import numpy as np
        import h5py

        _STR_DTYPE = h5py.string_dtype(encoding="utf-8")

        def convert_jsonl(src: Path, dst: Path, *, session_id: str | None = None) -> None:
            session_id = session_id or src.stem
            records = []
            with open(src, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("type") not in ("user", "assistant"):
                        continue
                    msg = obj.get("message", {})
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        content = json.dumps(content)
                    usage = msg.get("usage", {})
                    records.append({
                        "uuid":       obj.get("uuid", ""),
                        "parent":     obj.get("parentUuid") or "",
                        "role":       obj.get("type", ""),
                        "timestamp":  obj.get("timestamp", ""),
                        "content":    content,
                        "in_tok":     usage.get("input_tokens", 0),
                        "out_tok":    usage.get("output_tokens", 0),
                        "cache_cr":   usage.get("cache_creation_input_tokens", 0),
                        "cache_rd":   usage.get("cache_read_input_tokens", 0),
                    })
            if not records:
                return
            n = len(records)
            with h5py.File(dst, "w") as hf:
                grp = hf.require_group(f"sessions/{session_id}")
                for field in ("uuid", "parent", "role", "timestamp", "content"):
                    data = [r[field].encode("utf-8") for r in records]
                    grp.create_dataset(field, data=data, dtype=_STR_DTYPE,
                                       chunks=(min(n, 256),), compression="gzip")
                for field in ("in_tok", "out_tok", "cache_cr", "cache_rd"):
                    data = np.array([r[field] for r in records], dtype=np.int32)
                    grp.create_dataset(field, data=data,
                                       chunks=(min(n, 256),), compression="gzip")
        """),
    "src/agentic_conversations_hdf5/backends/hdf5_packed.py": textwrap.dedent("""\
        \"\"\"Packed-string HDF5 backend.  Each message is stored as a single
        JSON blob in a variable-length UTF-8 string dataset.\"\"\"
        from __future__ import annotations
        import json
        from pathlib import Path
        import h5py

        _STR_DTYPE = h5py.string_dtype(encoding="utf-8")

        def convert_jsonl_packed(src: Path, dst: Path, *, session_id: str | None = None) -> None:
            session_id = session_id or src.stem
            blobs: list[bytes] = []
            with open(src, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("type") not in ("user", "assistant"):
                        continue
                    blobs.append(json.dumps(obj).encode("utf-8"))
            if not blobs:
                return
            n = len(blobs)
            with h5py.File(dst, "a") as hf:
                grp = hf.require_group(f"sessions/{session_id}")
                grp.create_dataset("packed", data=blobs, dtype=_STR_DTYPE,
                                   chunks=(min(n, 256),), compression="gzip")
        """),
    "benchmarks/benchmark.py": textwrap.dedent("""\
        \"\"\"Benchmark harness for HDF5 vs. SQLite vs. JSON+NumPy session backends.\"\"\"
        from __future__ import annotations
        import argparse, json, time
        from pathlib import Path
        import h5py, numpy as np

        def _tail_jsonl(p: Path, n: int) -> list[dict]:
            lines = p.read_text().splitlines()
            out = []
            for line in reversed(lines):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("type") in ("user", "assistant"):
                    out.append(obj)
                    if len(out) >= n:
                        break
            return list(reversed(out))

        def bench_jsonl_tail(p: Path, n: int = 20) -> float:
            t0 = time.perf_counter()
            _tail_jsonl(p, n)
            return time.perf_counter() - t0

        def bench_hdf5_tail(p: Path, session_id: str, n: int = 20) -> float:
            t0 = time.perf_counter()
            with h5py.File(p, "r") as hf:
                grp = hf[f"sessions/{session_id}"]
                total = grp["role"].shape[0]
                start = max(0, total - n)
                _ = grp["content"][start:total]
            return time.perf_counter() - t0
        """),
    "tests/test_convert.py": textwrap.dedent("""\
        import json, tempfile
        from pathlib import Path
        import h5py, pytest
        from agentic_conversations_hdf5.convert import convert_jsonl

        SAMPLE = '''
        {\"type\":\"permission-mode\",\"permissionMode\":\"default\",\"sessionId\":\"s1\"}
        {\"parentUuid\":null,\"type\":\"user\",\"message\":{\"role\":\"user\",\"content\":\"hello\"},\"uuid\":\"u1\",\"timestamp\":\"2026-01-01T00:00:00Z\",\"sessionId\":\"s1\"}
        {\"parentUuid\":\"u1\",\"type\":\"assistant\",\"message\":{\"role\":\"assistant\",\"content\":[{\"type\":\"text\",\"text\":\"hi\"}],\"usage\":{\"input_tokens\":5,\"output_tokens\":3}},\"uuid\":\"a1\",\"timestamp\":\"2026-01-01T00:00:01Z\",\"sessionId\":\"s1\"}
        '''.strip()

        def test_round_trip():
            with tempfile.TemporaryDirectory() as d:
                src = Path(d) / "sess.jsonl"
                dst = Path(d) / "sess.h5"
                src.write_text(SAMPLE)
                convert_jsonl(src, dst, session_id="s1")
                with h5py.File(dst, "r") as hf:
                    grp = hf["sessions/s1"]
                    roles = [r.decode() for r in grp["role"][:]]
                    assert roles == ["user", "assistant"]
        """),
}

BASH_COMMANDS = [
    ("ls -la src/", "total 24\ndrwxr-xr-x 3 user user 4096 May 1 10:00 .\ndrwxr-xr-x 8 user user 4096 May 1 09:00 ..\ndrwxr-xr-x 2 user user 4096 May 1 10:00 agentic_conversations_hdf5\n"),
    ("python -m pytest tests/ -x -q",
     "......................                                              [100%]\n22 passed in 1.43s\n"),
    ("python benchmarks/jsonl_vs_hdf5.py --limit 5",
     "file                                 jsonl   gz   hdf5    packed  compound  tail_j  tail_h  tok_j  tok_h\n"
     "62fce10c-7c43.jsonl               2461.2  312.4  891.3   743.1   822.0     0.182   0.004  0.071  0.001\n"
     "b4c7b708-cd0f.jsonl               2699.2  341.1  973.4   812.3   901.2     0.201   0.005  0.078  0.001\n"
     "6980a209-9dd6.jsonl               2549.1  328.7  922.1   779.4   856.3     0.195   0.004  0.074  0.001\n"),
    ("python -c \"import h5py; print(h5py.version.info)\"",
     "summary of the h5py configuration\n    h5py    3.11.0\n    HDF5    1.14.3\n    Python  3.12.3\n    Numpy   1.26.4\n"),
    ("git log --oneline -5",
     "a3f1b2c Add compound backend\n9e2d7f1 Fix packed chunk size\n3c8a4e9 Initial scaffold\n"),
    ("python -c \"from agentic_conversations_hdf5 import __version__; print(__version__)\"",
     "0.1.0\n"),
]

EDIT_DESCRIPTIONS = [
    "Adding chunk size guard to prevent cache overflow",
    "Extracting shared JSONL parser into base module",
    "Adding progress bar using tqdm",
    "Fixing off-by-one in tail query",
    "Adding type annotations to public API",
    "Updating token-usage extraction to handle new usage schema",
]

TOOL_NAMES_READ = ["Read", "Glob", "Grep"]
TOOL_NAMES_WRITE = ["Edit", "Write", "Bash"]


# ---------------------------------------------------------------------------
# UUID helpers
# ---------------------------------------------------------------------------

def new_uuid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Line builders
# ---------------------------------------------------------------------------

def permission_line(session_id: str) -> dict:
    return {"type": "permission-mode", "permissionMode": "bypassPermissions",
            "sessionId": session_id}


def file_snapshot_line(session_id: str, message_id: str) -> dict:
    return {"type": "file-history-snapshot", "messageId": message_id,
            "snapshot": json.dumps({"messageId": message_id, "trackedFileBackups": {}}),
            "isSnapshotUpdate": False, "sessionId": session_id}


def last_prompt_line(session_id: str, prompt: str) -> dict:
    return {"type": "last-prompt", "lastPrompt": prompt[:120], "sessionId": session_id}


def queue_op_line(session_id: str, ts: str) -> dict:
    return {"type": "queue-operation", "operation": "enqueue", "timestamp": ts,
            "sessionId": session_id,
            "content": "<task-notification><task-id>synthetic</task-id><summary>background task</summary></task-notification>"}


def system_turn_line(parent_uuid: str, this_uuid: str, session_id: str, ts: str,
                     duration_ms: int, message_count: int, cwd: str, version: str,
                     git_branch: str) -> dict:
    return {"parentUuid": parent_uuid, "isSidechain": False, "type": "system",
            "subtype": "turn_duration", "durationMs": duration_ms,
            "messageCount": message_count, "timestamp": ts, "uuid": this_uuid,
            "isMeta": False, "userType": "external", "entrypoint": "cli",
            "cwd": cwd, "sessionId": session_id, "version": version,
            "gitBranch": git_branch}


def user_text_line(parent_uuid: str | None, this_uuid: str, session_id: str, ts: str,
                   text: str, cwd: str, version: str, git_branch: str,
                   *, is_first: bool = False) -> dict:
    return {"parentUuid": parent_uuid, "isSidechain": False,
            "promptId": new_uuid(),
            "type": "user", "isMeta": False,
            "message": {"role": "user", "content": text},
            "uuid": this_uuid, "timestamp": ts,
            "userType": "external", "entrypoint": "cli",
            "cwd": cwd, "sessionId": session_id, "version": version,
            "gitBranch": git_branch}


def user_tool_result_line(parent_uuid: str, this_uuid: str, session_id: str, ts: str,
                          tool_use_id: str, result_content: str, is_error: bool,
                          cwd: str, version: str, git_branch: str) -> dict:
    content_block = {"tool_use_id": tool_use_id, "type": "tool_result",
                     "content": result_content, "is_error": is_error}
    return {"parentUuid": parent_uuid, "isSidechain": False,
            "promptId": new_uuid(),
            "type": "user", "isMeta": False,
            "message": {"role": "user", "content": [content_block]},
            "uuid": this_uuid, "timestamp": ts,
            "userType": "external", "entrypoint": "cli",
            "toolUseResult": {"stdout": result_content, "stderr": "", "interrupted": False},
            "cwd": cwd, "sessionId": session_id, "version": version,
            "gitBranch": git_branch}


def assistant_line(parent_uuid: str, this_uuid: str, session_id: str, ts: str,
                   thinking: str, text: str, tool_uses: list[dict],
                   in_tok: int, out_tok: int, cache_cr: int, cache_rd: int,
                   cwd: str, version: str, git_branch: str) -> dict:
    content: list[dict] = []
    if thinking:
        content.append({"type": "thinking", "thinking": thinking})
    if text:
        content.append({"type": "text", "text": text})
    for tu in tool_uses:
        content.append({"type": "tool_use", "id": tu["id"],
                         "name": tu["name"], "input": tu["input"]})
    usage = {"input_tokens": in_tok, "output_tokens": out_tok,
             "cache_creation_input_tokens": cache_cr,
             "cache_read_input_tokens": cache_rd,
             "server_tool_use": {"web_search_requests": 0, "web_fetch_requests": 0},
             "service_tier": "standard",
             "cache_creation": {"ephemeral_1h_input_tokens": cache_cr,
                                "ephemeral_5m_input_tokens": 0},
             "speed": "standard"}
    return {"parentUuid": parent_uuid, "isSidechain": False,
            "requestId": f"req_{new_uuid().replace('-', '')[:24]}",
            "type": "assistant",
            "message": {"model": "claude-sonnet-4-6", "role": "assistant",
                        "content": content, "usage": usage},
            "uuid": this_uuid, "timestamp": ts,
            "userType": "external", "entrypoint": "cli",
            "cwd": cwd, "sessionId": session_id, "version": version,
            "gitBranch": git_branch}


# ---------------------------------------------------------------------------
# Session generator
# ---------------------------------------------------------------------------

class SessionGenerator:
    def __init__(self, session_id: str, cwd: str = "/home/user/project",
                 version: str = "2.1.116", git_branch: str = "main"):
        self.session_id = session_id
        self.cwd = cwd
        self.version = version
        self.git_branch = git_branch
        self.t = datetime(2026, 4, 15, 9, 0, 0, tzinfo=timezone.utc)
        self.prev_uuid: str | None = None
        self.message_count = 0

    def _ts(self) -> str:
        self.t += timedelta(seconds=random.uniform(1, 8))
        return self.t.strftime("%Y-%m-%dT%H:%M:%S.") + f"{self.t.microsecond // 1000:03d}Z"

    def _emit(self, obj: dict) -> str:
        return json.dumps(obj, separators=(",", ":")) + "\n"

    def generate(self, target_bytes: int) -> str:
        lines: list[str] = []

        # Header
        lines.append(self._emit(permission_line(self.session_id)))
        lines.append(self._emit(last_prompt_line(self.session_id, random.choice(USER_PROMPTS))))

        total = sum(len(l) for l in lines)
        round_num = 0

        while total < target_bytes:
            round_num += 1
            round_lines = self._generate_round(round_num)
            for l in round_lines:
                lines.append(l)
                total += len(l)
                if total >= target_bytes:
                    break

        return "".join(lines)

    def _generate_round(self, round_num: int) -> list[str]:
        lines: list[str] = []

        # --- 1. User prompt ---
        prompt = random.choice(USER_PROMPTS)
        u_uuid = new_uuid()
        ts = self._ts()
        lines.append(self._emit(user_text_line(
            self.prev_uuid, u_uuid, self.session_id, ts,
            prompt, self.cwd, self.version, self.git_branch,
            is_first=(round_num == 1),
        )))
        self.message_count += 1
        self.prev_uuid = u_uuid

        # --- 2. Assistant turn with 1-3 tool calls ---
        n_tools = random.randint(1, 3)
        tool_uses = []
        for _ in range(n_tools):
            tu_id = f"toolu_{new_uuid().replace('-', '')[:20]}"
            name, inp = self._random_tool_use()
            tool_uses.append({"id": tu_id, "name": name, "input": inp})

        a_uuid = new_uuid()
        ts = self._ts()
        thinking = random.choice(THINKING_SNIPPETS)
        text = f"I'll help with that. Let me {random.choice(['read', 'check', 'inspect', 'examine'])} the relevant files first."
        lines.append(self._emit(assistant_line(
            self.prev_uuid, a_uuid, self.session_id, ts,
            thinking, text, tool_uses,
            in_tok=random.randint(4000, 25000),
            out_tok=random.randint(200, 1200),
            cache_cr=random.randint(0, 30000),
            cache_rd=random.randint(0, 80000),
            cwd=self.cwd, version=self.version, git_branch=self.git_branch,
        )))
        self.message_count += 1
        self.prev_uuid = a_uuid

        # --- 3. Tool result(s) ---
        for tu in tool_uses:
            tr_uuid = new_uuid()
            ts = self._ts()
            result = self._random_tool_result(tu["name"])
            lines.append(self._emit(user_tool_result_line(
                self.prev_uuid, tr_uuid, self.session_id, ts,
                tu["id"], result, False,
                self.cwd, self.version, self.git_branch,
            )))
            self.message_count += 1
            self.prev_uuid = tr_uuid

        # --- 4. File snapshot (occasionally) ---
        if random.random() < 0.4:
            lines.append(self._emit(file_snapshot_line(self.session_id, a_uuid)))

        # --- 5. Final assistant response ---
        final_uuid = new_uuid()
        ts = self._ts()
        final_text = self._final_response(tool_uses)
        lines.append(self._emit(assistant_line(
            self.prev_uuid, final_uuid, self.session_id, ts,
            "", final_text, [],
            in_tok=random.randint(8000, 40000),
            out_tok=random.randint(100, 600),
            cache_cr=0,
            cache_rd=random.randint(10000, 80000),
            cwd=self.cwd, version=self.version, git_branch=self.git_branch,
        )))
        self.message_count += 1
        self.prev_uuid = final_uuid

        # --- 6. System turn-duration line (every few rounds) ---
        if random.random() < 0.3:
            sys_uuid = new_uuid()
            ts = self._ts()
            lines.append(self._emit(system_turn_line(
                self.prev_uuid, sys_uuid, self.session_id, ts,
                random.randint(1000, 120000), self.message_count,
                self.cwd, self.version, self.git_branch,
            )))
            self.prev_uuid = sys_uuid

        # --- 7. Queue operation (occasionally) ---
        if random.random() < 0.15:
            lines.append(self._emit(queue_op_line(self.session_id, ts)))

        return lines

    def _random_tool_use(self) -> tuple[str, dict]:
        choice = random.random()
        if choice < 0.25:
            # Bash
            cmd, _ = random.choice(BASH_COMMANDS)
            return "Bash", {"command": cmd}
        elif choice < 0.5:
            # Read
            path = random.choice(list(FILE_CONTENTS.keys()))
            return "Read", {"file_path": path}
        elif choice < 0.65:
            # Edit
            path = random.choice(list(FILE_CONTENTS.keys()))
            old = "    records.append({"
            new_text = "    # accumulate parsed record\n    records.append({"
            return "Edit", {"file_path": path, "old_string": old, "new_string": new_text}
        elif choice < 0.8:
            # Grep
            return "Grep", {"pattern": random.choice(["convert_jsonl", "chunk_size", "session_id", "h5py"]),
                            "path": "src/"}
        elif choice < 0.9:
            # Glob
            return "Glob", {"pattern": "src/**/*.py"}
        else:
            # Write
            path = f"tests/test_synthetic_{random.randint(1,99)}.py"
            return "Write", {"file_path": path, "content": "# synthetic test\n"}

    def _random_tool_result(self, tool_name: str) -> str:
        if tool_name == "Bash":
            _, output = random.choice(BASH_COMMANDS)
            return output
        elif tool_name == "Read":
            path = random.choice(list(FILE_CONTENTS.keys()))
            return FILE_CONTENTS[path]
        elif tool_name == "Edit":
            return "The file has been edited successfully."
        elif tool_name == "Grep":
            return "\n".join([
                f"src/agentic_conversations_hdf5/convert.py:{random.randint(1,100)}: {random.choice(['convert_jsonl', 'chunk_size', 'session_id'])}",
                f"src/agentic_conversations_hdf5/backends/hdf5_packed.py:{random.randint(1,80)}: {random.choice(['convert_jsonl', 'chunk_size'])}",
            ])
        elif tool_name == "Glob":
            return "\n".join([
                "src/agentic_conversations_hdf5/__init__.py",
                "src/agentic_conversations_hdf5/convert.py",
                "src/agentic_conversations_hdf5/convert_packed.py",
                "src/agentic_conversations_hdf5/convert_compound.py",
                "src/agentic_conversations_hdf5/backends/hdf5_packed.py",
                "src/agentic_conversations_hdf5/backends/hdf5_compound.py",
            ])
        elif tool_name == "Write":
            return "File written successfully."
        return "Done."

    def _final_response(self, tool_uses: list[dict]) -> str:
        names = [tu["name"] for tu in tool_uses]
        if "Edit" in names or "Write" in names:
            return (
                f"I've made the requested changes. {random.choice(EDIT_DESCRIPTIONS)}. "
                "The modification preserves backward compatibility — existing callers "
                "pass through unchanged. You may want to run `python -m pytest tests/ -x` "
                "to confirm the suite is still green."
            )
        elif "Bash" in names:
            return (
                "The command completed successfully. "
                f"{'All tests passed.' if random.random() > 0.2 else 'There are 2 failures — see the traceback above.'} "
                "Let me know if you'd like me to dig deeper into any of the output."
            )
        else:
            return (
                "I've reviewed the file. The key thing to note is that the JSONL parser "
                "iterates lines sequentially, which means random-access queries require a "
                "full scan. The HDF5 backend avoids this by storing each field as a separate "
                "chunked dataset, enabling column-level reads without touching unneeded data. "
                "Happy to expand on any part of the design."
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    default_outdir = Path(__file__).parent.parent / "tests" / "fixtures"

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", nargs="+", type=int, default=[25, 250], metavar="MB",
                   help="target file sizes in MB (default: 25 250)")
    p.add_argument("--outdir", type=Path, default=default_outdir,
                   help=f"output directory (default: {default_outdir})")
    p.add_argument("--seed", type=int, default=42,
                   help="random seed for reproducibility (default: 42)")
    args = p.parse_args()

    random.seed(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)

    for mb in args.sizes:
        filename = f"synthetic_{mb}mb.jsonl"
        target_bytes = mb * 1024 * 1024
        out_path = args.outdir / filename
        sid = str(uuid.uuid4())
        gen = SessionGenerator(session_id=sid)
        print(f"Generating {filename} (target {mb} MB)...", flush=True)
        content = gen.generate(target_bytes)
        actual_mb = len(content) / (1024 * 1024)
        out_path.write_text(content, encoding="utf-8")
        print(f"  wrote {out_path} ({actual_mb:.1f} MB, {content.count(chr(10))} lines)")

    print("Done.")


if __name__ == "__main__":
    main()
