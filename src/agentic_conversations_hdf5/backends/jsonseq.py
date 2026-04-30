"""JSON + NumPy agent session backend.

Layout
------
<base_dir>/
    session.json          — all turns and tool call metadata
    embeddings/
        {turn_id}.npy     — per-turn embedding (float32)
    result_data/
        {call_id}.npy     — per-tool-call array result
    artifacts/
        {name}.npy        — named artifacts
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .base import SessionBackend


class JSONSession(SessionBackend):
    def __init__(self, base_dir: str | Path, session_id: str = ""):
        self.base_dir = Path(base_dir)
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self._session_dir = self.base_dir / self.session_id
        self._session_dir.mkdir(parents=True, exist_ok=True)
        (self._session_dir / "embeddings").mkdir(exist_ok=True)
        (self._session_dir / "result_data").mkdir(exist_ok=True)
        (self._session_dir / "artifacts").mkdir(exist_ok=True)

        self._meta_path = self._session_dir / "session.json"
        self._data: dict[str, Any] = self._load_meta()

    def _load_meta(self) -> dict[str, Any]:
        if self._meta_path.exists():
            with open(self._meta_path) as f:
                return json.load(f)
        return {
            "session_id": self.session_id,
            "created_at": time.time(),
            "turns": [],
            "tool_calls": [],
        }

    def _save_meta(self) -> None:
        with open(self._meta_path, "w") as f:
            json.dump(self._data, f)

    # ------------------------------------------------------------------

    def add_turn(
        self,
        role: str,
        content: str,
        embedding: Optional[np.ndarray] = None,
    ) -> str:
        turn_id = str(uuid.uuid4())[:12]
        record: dict[str, Any] = {
            "turn_id": turn_id,
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "has_embedding": embedding is not None,
        }
        if embedding is not None:
            np.save(
                str(self._session_dir / "embeddings" / f"{turn_id}.npy"),
                embedding.astype(np.float32),
            )
        self._data["turns"].append(record)
        self._save_meta()
        return turn_id

    def add_tool_call(
        self,
        name: str,
        args: dict[str, Any],
        result_text: Optional[str] = None,
        result_data: Optional[np.ndarray] = None,
    ) -> str:
        call_id = str(uuid.uuid4())[:12]
        record: dict[str, Any] = {
            "call_id": call_id,
            "name": name,
            "args": args,
            "result_text": result_text,
            "timestamp": time.time(),
            "has_result_data": result_data is not None,
        }
        if result_data is not None:
            np.save(
                str(self._session_dir / "result_data" / f"{call_id}.npy"),
                result_data,
            )
        self._data["tool_calls"].append(record)
        self._save_meta()
        return call_id

    def get_recent_context(self, n: int = 20) -> list[dict[str, Any]]:
        turns = self._data["turns"][-n:]
        results = []
        for t in turns:
            d: dict[str, Any] = {
                "turn_id": t["turn_id"],
                "role": t["role"],
                "content": t["content"],
                "timestamp": t["timestamp"],
            }
            if t.get("has_embedding"):
                emb_path = self._session_dir / "embeddings" / f"{t['turn_id']}.npy"
                if emb_path.exists():
                    d["embedding"] = np.load(str(emb_path))
            results.append(d)
        return results

    def store_artifact(self, name: str, data: np.ndarray) -> None:
        np.save(str(self._session_dir / "artifacts" / f"{name}.npy"), data)

    def get_artifact(self, name: str) -> Optional[np.ndarray]:
        p = self._session_dir / "artifacts" / f"{name}.npy"
        if not p.exists():
            return None
        return np.load(str(p))

    def turn_count(self) -> int:
        return len(self._data["turns"])

    def close(self) -> None:
        # No persistent file handle to close.
        pass
