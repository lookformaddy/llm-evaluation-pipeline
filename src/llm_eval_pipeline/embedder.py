"""Deterministic, dependency-free local embedder.

Real semantic-similarity scorers would call an embedding model. For an
offline, reproducible harness we use a hashed bag-of-ngrams embedder:
tokens (word unigrams + bigrams) are hashed with SHA-256 into a fixed
dimension vector and L2-normalized. It captures lexical/semantic overlap
well enough to rank paraphrases above unrelated text, is bit-for-bit
reproducible across machines and runs (unlike salted ``hash()``), and
needs no network or model download.

Swap point: implement the :class:`Embedder` protocol with a real model
(e.g. a sentence-transformers wrapper) and inject it into
``SemanticSimilarityScorer`` — nothing else changes.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, runtime_checkable

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@runtime_checkable
class Embedder(Protocol):
    def embed(self, text: str) -> list[float]:
        ...


class HashEmbedder:
    """Hashed n-gram embedder with cosine-comparable output vectors."""

    def __init__(self, dim: int = 512) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def _add(self, vec: list[float], token: str, weight: float) -> None:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % self.dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign * weight

    def embed(self, text: str) -> list[float]:
        tokens = tokenize(text)
        vec = [0.0] * self.dim
        for tok in tokens:
            self._add(vec, tok, 1.0)
        for a, b in zip(tokens, tokens[1:]):
            self._add(vec, f"{a} {b}", 0.5)
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity; inputs are expected to be L2-normalized."""
    if len(a) != len(b):
        raise ValueError("vector dimensions differ")
    return sum(x * y for x, y in zip(a, b))
