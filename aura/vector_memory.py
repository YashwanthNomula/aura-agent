"""Zero-dependency semantic recall over AURA's persistent memory.

Each stored memory (a long-term fact or a past episode) is embedded as a
fixed-size dense vector using the "hashing trick": tokens and word bigrams
are hashed into buckets with random signs (derived deterministically from
SHA-256, so identical text always yields identical vectors on any machine).
Recall is a cosine-similarity top-k ranking over those vectors.

This is deliberately *not* a neural embedding: no downloads, no deps, no
GPU. It finds memories that share vocabulary with the query, including
rephrasings ("favorite food" ~ "likes pizza for dinner"), where exact
keyword search fails. Fast enough for thousands of memories.
"""
from __future__ import annotations

import hashlib
import math
import re

DIM = 512  # vector width; 512 keeps collisions rare at our memory sizes
_BIGRAM_WEIGHT = 1.5  # phrases ("fine tune") matter more than lone words

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, e.g. 'Fine-tune!' -> ['fine', 'tune']."""
    return _TOKEN_RE.findall(text.lower())


def _hash_unit(unit: str, dim: int) -> tuple[int, int]:
    """Deterministically map a token/bigram to (bucket_index, sign)."""
    digest = hashlib.sha256(unit.encode("utf-8")).digest()
    index = int.from_bytes(digest[:4], "big") % dim
    sign = 1 if (digest[4] & 1) else -1
    return index, sign


def embed(text: str, dim: int = DIM) -> list[float]:
    """Deterministic L2-normalized embedding of *text* (zero-dep)."""
    vec = [0.0] * dim
    tokens = tokenize(text)
    units: list[tuple[str, float]] = [(t, 1.0) for t in tokens]
    units += [(" ".join(pair), _BIGRAM_WEIGHT)
              for pair in zip(tokens, tokens[1:])]
    for unit, weight in units:
        index, sign = _hash_unit(unit, dim)
        vec[index] += sign * weight
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two (already normalized) vectors."""
    return sum(x * y for x, y in zip(a, b))


class VectorIndex:
    """In-memory index of documents, ranked by semantic similarity."""

    def __init__(self, dim: int = DIM):
        self.dim = dim
        self._docs: list[tuple[str, str, list[float]]] = []  # (id, text, vec)

    def __len__(self) -> int:
        return len(self._docs)

    def add(self, doc_id: str, text: str) -> None:
        self._docs.append((doc_id, text, embed(text, self.dim)))

    def query(self, text: str, k: int = 5) -> list[tuple[str, str, float]]:
        """Top-k (doc_id, text, cosine_score) most similar to *text*."""
        q = embed(text, self.dim)
        ranked = sorted(
            ((doc_id, doc, cosine(q, vec)) for doc_id, doc, vec in self._docs),
            key=lambda r: r[2],
            reverse=True,
        )
        return ranked[: max(k, 0)]


def index_memory(memory) -> VectorIndex:
    """Build a semantic index over a :class:`aura.memory.Memory`.

    Facts are indexed as ``"key: value"``; episodes as ``"task result"``.
    """
    index = VectorIndex()
    for key, value in memory.facts.items():
        index.add(f"fact:{key}", f"{key}: {value}")
    for i, ep in enumerate(memory.episodes):
        index.add(f"episode:{i}", f"{ep.get('task', '')} {ep.get('result', '')}")
    return index
