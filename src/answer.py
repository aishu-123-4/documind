"""Answer a question from retrieved chunks — Claude if available, else extractive.

Key-optional: with an ANTHROPIC_API_KEY (or `ant auth login` profile) Claude
writes a grounded, cited answer; otherwise the top passages are returned with
their `document#chunk` citations.
"""

from __future__ import annotations

import os

from .retriever import Hit

MODEL = "claude-opus-5-5"

SYSTEM_PROMPT = (
    "You are DocuMind, a document question-answering assistant. Answer using "
    "ONLY the provided passages. Cite sources inline as [document#chunk]. If the "
    "passages do not contain the answer, say so — never invent facts."
)


def llm_available() -> bool:
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return bool(os.environ.get("ANTHROPIC_API_KEY")
                or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                or os.path.isdir(os.path.expanduser("~/.config/anthropic")))


def _context(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{h.chunk.citation}]\n{h.chunk.text}" for h in hits)


def answer_extractive(query: str, hits: list[Hit]) -> str:
    if not hits:
        return "No relevant passages found. Try rephrasing or re-indexing."
    lines = [f'Top passages for: "{query}"',
             "(extractive mode — set ANTHROPIC_API_KEY for a written answer)\n"]
    for i, h in enumerate(hits, 1):
        snippet = " ".join(h.chunk.text.split())
        if len(snippet) > 320:
            snippet = snippet[:320] + "…"
        lines.append(f"{i}. [{h.chunk.citation}]  (score {h.score:.3f})")
        lines.append(f"   {snippet}\n")
    return "\n".join(lines)


def answer_llm(query: str, hits: list[Hit]) -> str:
    import anthropic

    client = anthropic.Anthropic()
    user = (f"Question: {query}\n\nPassages:\n\n{_context(hits)}\n\n"
            "Answer with inline [document#chunk] citations.")
    resp = client.messages.create(
        model=MODEL, max_tokens=1200, system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def answer(query: str, hits: list[Hit], use_llm: bool | None = None) -> str:
    if use_llm is None:
        use_llm = llm_available()
    if use_llm:
        try:
            return answer_llm(query, hits)
        except Exception as exc:  # noqa: BLE001
            return (f"[LLM call failed: {exc}. Falling back to extractive.]\n\n"
                    + answer_extractive(query, hits))
    return answer_extractive(query, hits)
