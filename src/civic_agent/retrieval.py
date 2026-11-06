"""Hybrid search: BM25 keyword ranking fused with embedding similarity, filtered by who is asking."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from importlib import resources
from typing import Protocol

import numpy as np

# Which roles may read which access levels. Filtering happens before ranking, so a document a role
# may not see can never leak through a snippet or a citation.
ACCESS = {"citizen": {"public"}, "caseworker": {"public", "staff"}}

TOKEN = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class Doc:
    id: str
    title: str
    access: str
    text: str


@dataclass(frozen=True)
class Hit:
    doc: Doc
    score: float


def load_documents() -> list[Doc]:
    raw = resources.files("civic_agent.data").joinpath("documents.json").read_text(encoding="utf-8")
    return [Doc(**d) for d in json.loads(raw)]


STOPWORDS = set("""a an and are at be by can do does for from get how i in is it my of on or the to what when where
which who why will with you your me much many need ہے ہیں کی کا کے میں کیا کتنی کتنے کتنا کو سے اور یہ""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in (w.lower() for w in TOKEN.findall(text)) if t not in STOPWORDS]


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


class HashEmbedder:
    """Deterministic bag-of-words hashing. Offline and instant: used by tests and when no model is available."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in tokenize(text):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        return out


class Model2VecEmbedder:
    """Static sentence embeddings (numpy only, runs on any CPU). Downloads the model on first use."""

    def __init__(self, name: str = "minishlab/potion-base-8M"):
        from model2vec import StaticModel

        self.model = StaticModel.from_pretrained(name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts), dtype=np.float32)


def default_embedder() -> Embedder:
    return Model2VecEmbedder() if os.getenv("CIVIC_EMBEDDER", "model2vec") == "model2vec" else HashEmbedder()


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.avgdl = sum(map(len, docs)) / max(len(docs), 1)
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in docs]

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for tf, d in zip(self.tf, self.docs):
            s = 0.0
            for t in query:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / self.avgdl))
            out.append(s)
        return out


def _normalise(m: np.ndarray) -> np.ndarray:
    return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)


class HybridIndex:
    """Reciprocal rank fusion of BM25 and cosine similarity (k=60, the usual constant)."""

    def __init__(self, docs: list[Doc], embedder: Embedder | None = None, rrf_k: int = 60, min_semantic: float = 0.55):
        self.docs = docs
        self.min_semantic = min_semantic
        self.rrf_k = rrf_k
        self.embedder = embedder or default_embedder()
        self.bm25 = BM25([tokenize(d.title + " " + d.text) for d in docs])
        self.vectors = _normalise(self.embedder.encode([d.title + ". " + d.text for d in docs]))

    def search(self, query: str, role: str, k: int = 3) -> list[Hit]:
        allowed = ACCESS.get(role, set())
        visible = [i for i, d in enumerate(self.docs) if d.access in allowed]
        if not visible or not query.strip():
            return []
        kw = self.bm25.scores(tokenize(query))
        qv = _normalise(self.embedder.encode([query]))[0]
        sem = self.vectors @ qv
        fused: dict[int, float] = {}
        for ranking in (sorted(visible, key=lambda i: -kw[i]), sorted(visible, key=lambda i: -sem[i])):
            for rank, i in enumerate(ranking):
                fused[i] = fused.get(i, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        # With no shared keyword, a document needs strong semantic similarity. 0.55 was measured on potion-base-8M:
        # real matches scored 0.62-0.98, an off-topic "passport" question 0.47 (see README, Retrieval).
        top = sorted(fused, key=lambda i: -fused[i])[:k]
        return [Hit(self.docs[i], round(fused[i], 5)) for i in top if kw[i] > 0 or sem[i] >= self.min_semantic]
