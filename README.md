# DocuMind — RAG Document Q&A  📄🧠

Ask questions over your own documents and get **citation-grounded** answers.
DocuMind ingests `.txt` / `.md` (and `.pdf` with `pypdf`), chunks them, retrieves
the most relevant passages, and answers — **offline by default**, upgrading to
Claude-generated answers when an `ANTHROPIC_API_KEY` is present.

This is the canonical "RAG app" employers ask about, built end-to-end.

## How it works

```
docs/ ─► ingest.py ─► paragraph-aware chunks (with overlap, source-tagged)
                         └─► TfidfVectorizer ─► sparse matrix ──► index/*.joblib
query ─► retriever.py ─► cosine similarity ─► top-k passages
         └─► answer.py ─► Claude (if key) grounded + cited, else extractive
```

- **Paragraph-aware chunking** with overlap so context isn't split mid-idea; each
  chunk keeps its `document#chunk` citation.
- **TF-IDF retrieval** — no model download, runs anywhere.
- **Grounded answers** — the LLM is instructed to answer *only* from retrieved
  passages and cite them, which is what keeps RAG honest (no hallucinated facts).

## Quickstart

```bash
cd 01-documind
pip install -r requirements.txt

# Bundled sample docs let you try it immediately:
python -m src ask "What are the two stages of photosynthesis?" --no-llm

# Index your own documents:
python -m src index /path/to/your/docs
python -m src ask "summarize the refund policy"

# With a key, answers are written and cited:
export ANTHROPIC_API_KEY=sk-ant-...
python -m src ask "What are the two stages of photosynthesis?"
```

Example (extractive mode):

```
[retrieved 2 passages | extractive mode]
Top passages for: "What are the two stages of photosynthesis?"
1. [photosynthesis.md#0]  (score 0.217)
   ... The light-dependent reactions occur in the thylakoid membranes ...
```

With an API key the same query returns a written answer citing
`[photosynthesis.md#0]`.

## Project layout

```
01-documind/
├── README.md
├── requirements.txt
├── docs/                 # bundled sample documents (works out of the box)
├── index/                # saved index (git-ignored)
└── src/
    ├── ingest.py         # read + paragraph-aware chunking (txt/md/pdf)
    ├── index.py          # TF-IDF index build / save / load
    ├── retriever.py      # cosine top-k search
    ├── answer.py         # Claude-or-extractive, citation-grounded
    └── __main__.py       # index / ask CLI
tests/
└── test_documind.py      # 9 tests, no API key required
```

## Tests

```bash
python -m pytest -q      # 9 tests
```

## DocuMind vs. RepoMind

DocuMind is RepoMind's sibling: same RAG backbone, different domain. RepoMind
chunks *code* by symbol (`ast`); DocuMind chunks *prose* by paragraph and reads
PDFs. Together they show the retrieval architecture generalizes.

## Possible extensions

- Dense embeddings (`sentence-transformers`) + FAISS behind the same interface.
- Hybrid BM25 + dense retrieval with reciprocal-rank fusion.
- Conversational memory (follow-up questions) and answer streaming.
- A Streamlit chat UI with clickable citations.
