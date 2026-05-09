from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from document_intelligence_rag.answering import build_extractive_answer
from document_intelligence_rag.retrieval.tfidf import TfidfRetriever

logger = logging.getLogger(__name__)


def answer_query(
    *,
    index_path: str | Path,
    query: str,
    top_k: int,
    output: str | Path | None = None,
) -> dict[str, Any]:
    retriever = TfidfRetriever.load(index_path)
    retrieval_results = retriever.retrieve(query, top_k=top_k)
    grounded_answer = build_extractive_answer(query, retrieval_results)
    payload = {
        "query": query,
        "top_k": top_k,
        "backend": retriever.backend,
        "index_path": str(index_path),
        **grounded_answer.to_dict(),
    }

    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Wrote grounded answer to %s", output_path)

    return payload


def _print_answer(payload: dict[str, Any]) -> None:
    print(f"query: {payload['query']}")
    print(f"backend: {payload['backend']}")
    print(f"index_path: {payload['index_path']}")
    print(f"insufficient_context: {payload['insufficient_context']}")
    print(f"confidence_score: {payload['confidence_score']:.4f}")
    print("answer:")
    print(payload["answer"])
    if payload["cited_chunk_ids"]:
        print("cited_chunk_ids: " + ", ".join(payload["cited_chunk_ids"]))
    if payload["cited_document_ids"]:
        print("cited_document_ids: " + ", ".join(payload["cited_document_ids"]))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a local grounded extractive answer.")
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    payload = answer_query(
        index_path=args.index_path,
        query=args.query,
        top_k=args.top_k,
        output=args.output,
    )
    _print_answer(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
