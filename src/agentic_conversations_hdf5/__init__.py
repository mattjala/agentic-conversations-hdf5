"""HDF5-backed agent conversation log storage."""
from .backends.base import SessionBackend, Turn, ToolCall
from .backends.hdf5 import HDF5Session
from .backends.sqlite import SQLiteSession
from .backends.jsonseq import JSONSession
from .schema import SCHEMA_VERSION

__all__ = [
    "SessionBackend",
    "Turn",
    "ToolCall",
    "HDF5Session",
    "SQLiteSession",
    "JSONSession",
    "ORCSession",
    "SCHEMA_VERSION",
]


def __getattr__(name: str):
    # ORCSession needs pyarrow (an optional 'orc' extra). Import lazily so the
    # core package and the live-session hook work without pyarrow installed.
    if name == "ORCSession":
        from .backends.orc_backend import ORCSession
        return ORCSession
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
