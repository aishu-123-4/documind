"""Tests for ingestion, chunking, indexing, retrieval (no API key needed)."""

from pathlib import Path

import pytest

from src.answer import answer_extractive
from src.index import build_index, load_index, save_index
from src.ingest import chunk_text, ingest_path
from src.retriever import search


@pytest.fixture
def docs(tmp_path: Path) -> Path:
    (tmp_path / "a.md").write_text(
        "# Coffee\n\nCoffee is a brewed drink prepared from roasted coffee "
        "beans. It contains caffeine, a stimulant.\n\n"
        "## Origins\n\nCoffee originated in Ethiopia and spread through the "
        "Arabian peninsula.\n",
        encoding="utf-8",
    )
    (tmp_path / "b.txt").write_text(
        "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. "
        "It was completed in 1889 and is 330 metres tall.",
        encoding="utf-8",
    )
    return tmp_path


def test_chunk_text_splits_and_overlaps():
    text = "\n\n".join(f"Paragraph number {i} with some filler words." for i in range(40))
    chunks = chunk_text(text, target=200, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 260 for c in chunks)  # target + a little slack

def test_chunk_text_hard_splits_long_paragraph():
    long_para = "x" * 5000
    chunks = chunk_text(long_para, target=900, overlap=200)
    assert len(chunks) >= 6

def test_ingest_reads_txt_and_md(docs):
    chunks = ingest_path(docs)
    names = {c.doc for c in chunks}
    assert any("a.md" in n for n in names)
    assert any("b.txt" in n for n in names)

def test_ingest_missing_path_raises():
    with pytest.raises(FileNotFoundError):
        ingest_path("/no/such/place-xyz")

def test_build_index_and_retrieve_coffee(docs):
    index = build_index(docs)
    hits = search(index, "where did coffee come from?", k=2)
    assert hits
    assert any("a.md" in h.chunk.doc for h in hits)

def test_retrieve_eiffel(docs):
    index = build_index(docs)
    hits = search(index, "how tall is the tower in Paris?", k=2)
    assert any("b.txt" in h.chunk.doc for h in hits)

def test_save_load_roundtrip(docs, tmp_path):
    index = build_index(docs)
    out = save_index(index, tmp_path / "idx.joblib")
    loaded = load_index(out)
    assert len(loaded["chunks"]) == len(index["chunks"])
    hits = search(loaded, "caffeine stimulant", k=2)
    assert hits

def test_extractive_answer_has_citation(docs):
    index = build_index(docs)
    hits = search(index, "coffee caffeine", k=2)
    out = answer_extractive("coffee caffeine", hits)
    assert "a.md#" in out or "b.txt#" in out

def test_build_index_empty_raises(tmp_path):
    with pytest.raises(ValueError):
        build_index(tmp_path)
