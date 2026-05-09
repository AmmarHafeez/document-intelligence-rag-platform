from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from document_intelligence_rag.models import RetrievalResult
from document_intelligence_rag.retrieval.preview import preview_text
from document_intelligence_rag.retrieval.tfidf import TfidfRetriever

logger = logging.getLogger(__name__)


def result_to_record(result: RetrievalResult) -> dict[str, Any]:
    chunk = result.chunk
    return {
        "rank": result.rank,
        "score": result.score,
        "document_id": chunk.document_id,
        "chunk_id": chunk.chunk_id,
        "source_path": str(chunk.source_path),
        "start_char": chunk.start_char,
        "end_char": chunk.end_char,
        "preview": preview_text(chunk.text),
        "text": chunk.text,
        "matched_terms": result.matched_terms,
    }


def query_index(
    *,
    index_path: str | Path,
    query: str,
    top_k: int,
    output: str | Path | None = None,
) -> dict[str, Any]:
    retriever = TfidfRetriever.load(index_path)
    results = retriever.retrieve(query, top_k=top_k)
    payload = {
        "query": query,
        "top_k": top_k,
        "backend": retriever.backend,
        "index_path": str(index_path),
        "results": [result_to_record(result) for result in results],
    }

    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Wrote retrieval results to %s", output_path)

    return payload


def _print_results(payload: dict[str, Any]) -> None:
    print(f"query: {payload['query']}")
    print(f"backend: {payload['backend']}")
    print(f"index_path: {payload['index_path']}")
    for result in payload["results"]:
        print(
            f"{result['rank']}. score={result['score']:.4f} "
            f"document_id={result['document_id']} source_path={result['source_path']}"
        )
        print(f"   {result['preview']}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query a saved local retrieval index.")
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    payload = query_index(
        index_path=args.index_path,
        query=args.query,
        top_k=args.top_k,
        output=args.output,
    )
    _print_results(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
