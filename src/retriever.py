"""Retrieve the most relevant document chunks for a query."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.metrics.pairwise import cosine_similarity

from .ingest import Chunk


@dataclass
class Hit:
    chunk: Chunk
    score: float


def search(index: dict, query: str, k: int = 4) -> list[Hit]:
    q = index["vectorizer"].transform([query])
    sims = cosine_similarity(q, index["matrix"]).ravel()
    order = sims.argsort()[::-1][:k]
    return [Hit(index["chunks"][i], float(sims[i])) for i in order if sims[i] > 0.0]
