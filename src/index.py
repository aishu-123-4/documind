"""Build / persist a TF-IDF retrieval index over document chunks."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from .ingest import Chunk, ingest_path


def build_index(source: str | Path) -> dict:
    chunks = ingest_path(source)
    if not chunks:
        raise ValueError(f"No readable documents found at {source}")
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2),
                                 stop_words="english", min_df=1, max_df=0.95)
    matrix = vectorizer.fit_transform(c.text for c in chunks)
    return {"vectorizer": vectorizer, "matrix": matrix, "chunks": chunks,
            "source": str(source)}


def save_index(index: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**index, "chunks": [asdict(c) for c in index["chunks"]]}
    joblib.dump(payload, path)
    return path


def load_index(path: str | Path) -> dict:
    payload = joblib.load(path)
    payload["chunks"] = [Chunk(**d) for d in payload["chunks"]]
    return payload
