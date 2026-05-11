"""Embedder implementations.

Two embedders are provided:

  FakeHashEmbedder
    Deterministic, dependency-free, ~microsecond per text. Hashes the text
    into a fixed-dim float32 vector and unit-normalises. Use this in
    benchmarks to isolate vector-store cost from embedder cost — the
    Chroma RAM-blowup is fundamentally about the embedder process, so
    benchmarks need to vary that axis independently.

  MiniLMEmbedder
    sentence-transformers all-MiniLM-L6-v2 (384 dim). The realistic
    embedder claude-mem would actually use. Imported lazily so the rest of
    the harness runs without sentence-transformers installed.
"""
from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

from .vector_store import Embedder


class FakeHashEmbedder(Embedder):
    """Deterministic hash-based pseudo-embedder.

    NOT a semantically meaningful embedding — but produces stable, distinct,
    unit-norm float32 vectors at a known dimension. Sufficient for measuring
    vector-store performance independent of model load + inference cost.
    """

    def __init__(self, dim: int = 384, seed: int = 0):
        self.dim = dim
        self.seed = seed

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            # Deterministic per-text vector: BLAKE2b stream → float32 → unit-norm.
            h = hashlib.blake2b(
                text.encode("utf-8"),
                digest_size=64,
                key=self.seed.to_bytes(8, "little"),
            ).digest()
            # Expand the 64-byte hash into `dim` float32 values by repeated hashing.
            buf = bytearray()
            counter = 0
            while len(buf) < self.dim * 4:
                buf.extend(
                    hashlib.blake2b(
                        h + counter.to_bytes(4, "little"), digest_size=64
                    ).digest()
                )
                counter += 1
            arr = np.frombuffer(bytes(buf[: self.dim * 4]), dtype=np.uint32)
            # Map uint32 → float in [-1, 1)
            v = (arr.astype(np.float64) / (2**31)) - 1.0
            v = v.astype(np.float32)
            n = np.linalg.norm(v)
            out[i] = v / n if n > 0 else v
        return out


class MiniLMEmbedder(Embedder):
    """sentence-transformers all-MiniLM-L6-v2, 384 dim. Lazy-loaded."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.dim = 384
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # noqa: WPS433
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        model = self._load()
        v = model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return v.astype(np.float32, copy=False)
