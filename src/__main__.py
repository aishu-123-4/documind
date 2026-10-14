"""DocuMind CLI.

    python -m src index <path-to-docs>        # build & save the index
    python -m src ask "question?"              # retrieve + answer
    python -m src ask "..." --docs ./docs --k 5 --no-llm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .answer import answer, llm_available
from .index import build_index, load_index, save_index
from .retriever import search

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INDEX = ROOT / "index" / "docs.joblib"
DEFAULT_DOCS = ROOT / "docs"


def cmd_index(args: argparse.Namespace) -> int:
    index = build_index(args.path)
    out = save_index(index, args.out)
    docs = len({c.doc for c in index["chunks"]})
    print(f"Indexed {len(index['chunks'])} chunks from {docs} document(s).")
    print(f"Saved index -> {out}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    index_path = Path(args.index)
    if index_path.exists():
        index = load_index(index_path)
    else:
        source = args.docs or (DEFAULT_DOCS if DEFAULT_DOCS.exists() else None)
        if not source:
            print(f"No index at {index_path}. Run `index <path>` or pass --docs.",
                  file=sys.stderr)
            return 1
        index = build_index(source)

    hits = search(index, args.question, k=args.k)
    use_llm = False if args.no_llm else None
    mode = "extractive" if args.no_llm or not llm_available() else "LLM"
    print(f"[retrieved {len(hits)} passages | {mode} mode]\n")
    print(answer(args.question, hits, use_llm=use_llm))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="documind")
    sub = p.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("index"); pi.set_defaults(func=cmd_index)
    pi.add_argument("path")
    pi.add_argument("--out", default=str(DEFAULT_INDEX))

    pa = sub.add_parser("ask"); pa.set_defaults(func=cmd_ask)
    pa.add_argument("question")
    pa.add_argument("--index", default=str(DEFAULT_INDEX))
    pa.add_argument("--docs", default=None)
    pa.add_argument("--k", type=int, default=4)
    pa.add_argument("--no-llm", action="store_true")

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
