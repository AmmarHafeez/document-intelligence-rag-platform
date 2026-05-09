from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from document_intelligence_rag.chunking import split_documents
from document_intelligence_rag.ingestion import load_documents
from document_intelligence_rag.retrieval.tfidf import TfidfRetriever

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BuildIndexSummary:
    document_count: int
    chunk_count: int
    backend: str
    index_path: Path


def build_index(
    *,
    documents_dir: str | Path,
    index_path: str | Path,
    chunk_size: int,
    chunk_overlap: int,
    backend: str = "tfidf",
) -> BuildIndexSummary:
    if backend != "tfidf":
        raise ValueError("Only the 'tfidf' backend is supported for saved vector indexes.")

    documents = load_documents(documents_dir)
    chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    retriever = TfidfRetriever().fit(chunks)
    saved_path = retriever.save(index_path)

    summary = BuildIndexSummary(
        document_count=len(documents),
        chunk_count=len(chunks),
        backend=backend,
        index_path=saved_path,
    )
    logger.info("Built %s index with %d documents and %d chunks", backend, len(documents), len(chunks))
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a local retrieval index.")
    parser.add_argument("--documents-dir", type=Path, required=True)
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument("--backend", choices=["tfidf"], default="tfidf")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    summary = build_index(
        documents_dir=args.documents_dir,
        index_path=args.index_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        backend=args.backend,
    )
    print(f"document_count: {summary.document_count}")
    print(f"chunk_count: {summary.chunk_count}")
    print(f"backend: {summary.backend}")
    print(f"index_path: {summary.index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
