from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from document_intelligence_rag.chunking import split_documents
from document_intelligence_rag.config import load_config
from document_intelligence_rag.ingestion import load_documents
from document_intelligence_rag.retrieval import KeywordRetriever


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="document_intelligence_rag",
        description="Run local document retrieval commands.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a YAML config file.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    retrieve_parser = subparsers.add_parser("retrieve", help="Retrieve chunks for a query.")
    retrieve_parser.add_argument("query", help="Query text.")
    retrieve_parser.add_argument("--top-k", type=int, default=None, help="Number of ranked chunks.")
    return parser


def _retrieve_payload(query: str, top_k: int | None, config_path: Path | None) -> dict[str, Any]:
    config = load_config(config_path)
    documents = load_documents(config.documents_dir)
    chunks = split_documents(
        documents,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    retriever = KeywordRetriever(chunks)
    requested_top_k = top_k or config.top_k
    results = retriever.retrieve(query, top_k=requested_top_k)
    return {
        "query": query,
        "top_k": requested_top_k,
        "results": [result.to_dict() for result in results],
    }


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "retrieve":
        payload = _retrieve_payload(args.query, args.top_k, args.config)
        print(json.dumps(payload, indent=2))
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2
