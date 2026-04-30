"""Abstract base class for agent session backends."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class Turn:
    turn_id: str
    role: str           # "user" | "assistant" | "system"
    content: str
    timestamp: float
    embedding: Optional[np.ndarray] = None   # (D,) float32, optional


@dataclass
class ToolCall:
    call_id: str
    name: str
    args: dict[str, Any]
    timestamp: float
    result_text: Optional[str] = None
    result_data: Optional[np.ndarray] = None  # array result, optional


class SessionBackend(ABC):
    """Common interface for all agent session storage backends."""

    @abstractmethod
    def add_turn(
        self,
        role: str,
        content: str,
        embedding: Optional[np.ndarray] = None,
    ) -> str:
        """Add a conversation turn. Returns turn_id."""

    @abstractmethod
    def add_tool_call(
        self,
        name: str,
        args: dict[str, Any],
        result_text: Optional[str] = None,
        result_data: Optional[np.ndarray] = None,
    ) -> str:
        """Add a tool call record. Returns call_id."""

    @abstractmethod
    def get_recent_context(self, n: int = 20) -> list[dict[str, Any]]:
        """Return the last n turns as a list of dicts (role, content, timestamp).

        Each dict has at minimum: turn_id, role, content, timestamp.
        If an embedding was stored it is included as 'embedding'.
        """

    @abstractmethod
    def store_artifact(self, name: str, data: np.ndarray) -> None:
        """Store a named array artifact in the session."""

    @abstractmethod
    def get_artifact(self, name: str) -> Optional[np.ndarray]:
        """Retrieve a named artifact. Returns None if not found."""

    @abstractmethod
    def turn_count(self) -> int:
        """Return the number of turns stored."""

    @abstractmethod
    def close(self) -> None:
        """Release any open resources."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def timed(fn, *args, **kwargs):
    """Return (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0
