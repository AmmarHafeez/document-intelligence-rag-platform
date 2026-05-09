from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from document_intelligence_rag.chunking import split_documents
from document_intelligence_rag.evaluation.metrics import (
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)
from document_intelligence_rag.ingestion import load_documents
from document_intelligence_rag.models import RetrievalResult, TextChunk
from document_intelligence_rag.retrieval import KeywordRetriever, TfidfRetriever
from document_intelligence_rag.retrieval.preview import preview_text

logger = logging.getLogger(__name__)


class Retriever(Protocol):
    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievalResult]:
        ...


@dataclass(frozen=True, slots=True)
class EvaluationQuery:
    query_id: str
    query: str
    relevant_document_ids: tuple[str, ...] = ()
    relevant_chunk_ids: tuple[str, ...] = ()


def _require_string(value: Any, field_name: str, item_index: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Query item {item_index} must include a non-empty '{field_name}'.")
    return value.strip()


def _optional_string_list(value: Any, field_name: str, item_index: int) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Query item {item_index} field '{field_name}' must be a list of strings.")
    return tuple(item for item in value if item)


def parse_evaluation_queries(path: str | Path) -> list[EvaluationQuery]:
    query_path = Path(path)
    try:
        payload = json.loads(query_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Evaluation query file is not valid JSON: {query_path}") from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("queries"), list):
        raise ValueError("Evaluation query file must contain a 'queries' list.")

    queries: list[EvaluationQuery] = []
    for index, item in enumerate(payload["queries"], start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Query item {index} must be an object.")

        queries.append(
            EvaluationQuery(
                query_id=_require_string(item.get("query_id"), "query_id", index),
                query=_require_string(item.get("query"), "query", index),
                relevant_document_ids=_optional_string_list(
                    item.get("relevant_document_ids"),
                    "relevant_document_ids",
                    index,
                ),
                relevant_chunk_ids=_optional_string_list(
                    item.get("relevant_chunk_ids"),
                    "relevant_chunk_ids",
                    index,
                ),
            )
        )

    return queries


def _build_retriever(backend: str, chunks: list[TextChunk]) -> Retriever:
    if backend == "keyword":
        return KeywordRetriever(chunks)
    if backend == "tfidf":
        return TfidfRetriever().fit(chunks)
    raise ValueError(f"Unsupported retrieval backend: {backend}.")


def _select_relevance(query: EvaluationQuery) -> tuple[str, tuple[str, ...]]:
    if query.relevant_chunk_ids:
        return "chunk", query.relevant_chunk_ids
    return "document", query.relevant_document_ids


def _result_record(result: RetrievalResult) -> dict[str, Any]:
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
    }


def _evaluate_query(
    *,
    evaluation_query: EvaluationQuery,
    retriever: Retriever,
    top_k: int,
) -> dict[str, Any]:
    results = retriever.retrieve(evaluation_query.query, top_k=top_k)
    retrieved_document_ids = [result.chunk.document_id for result in results]
    retrieved_chunk_ids = [result.chunk.chunk_id for result in results]
    relevance_type, relevant_ids = _select_relevance(evaluation_query)
    retrieved_ids = retrieved_chunk_ids if relevance_type == "chunk" else retrieved_document_ids

    return {
        "query_id": evaluation_query.query_id,
        "query": evaluation_query.query,
        "relevance_type": relevance_type,
        "relevant_document_ids": list(evaluation_query.relevant_document_ids),
        "relevant_chunk_ids": list(evaluation_query.relevant_chunk_ids),
        "retrieved_document_ids": retrieved_document_ids,
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "metrics": {
            "recall_at_k": recall_at_k(retrieved_ids, relevant_ids, top_k),
            "precision_at_k": precision_at_k(retrieved_ids, relevant_ids, top_k),
            "reciprocal_rank": mean_reciprocal_rank([retrieved_ids], [relevant_ids], k=top_k),
        },
        "results": [_result_record(result) for result in results],
    }


def _aggregate_metrics(per_query_results: list[dict[str, Any]], top_k: int) -> dict[str, float | int]:
    query_count = len(per_query_results)
    if query_count == 0:
        return {
            "recall_at_k": 0.0,
            "precision_at_k": 0.0,
            "mean_reciprocal_rank": 0.0,
            "query_count": 0,
            "evaluated_at_k": top_k,
        }

    recall = sum(float(item["metrics"]["recall_at_k"]) for item in per_query_results) / query_count
    precision = sum(float(item["metrics"]["precision_at_k"]) for item in per_query_results) / query_count
    reciprocal_rank = (
        sum(float(item["metrics"]["reciprocal_rank"]) for item in per_query_results) / query_count
    )

    return {
        "recall_at_k": recall,
        "precision_at_k": precision,
        "mean_reciprocal_rank": reciprocal_rank,
        "query_count": query_count,
        "evaluated_at_k": top_k,
    }


def evaluate_retrieval(
    *,
    documents_dir: str | Path,
    queries_path: str | Path,
    output_path: str | Path,
    backend: str,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
) -> dict[str, Any]:
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    queries = parse_evaluation_queries(queries_path)
    documents = load_documents(documents_dir)
    if not documents:
        raise ValueError(f"No documents found in {Path(documents_dir)}.")

    chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        raise ValueError("No chunks were generated from the provided documents.")

    retriever = _build_retriever(backend, chunks)
    per_query_results = [
        _evaluate_query(evaluation_query=query, retriever=retriever, top_k=top_k)
        for query in queries
    ]
    report = {
        "backend": backend,
        "documents_dir": str(documents_dir),
        "queries_path": str(queries_path),
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "metrics": _aggregate_metrics(per_query_results, top_k),
        "queries": per_query_results,
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote retrieval evaluation report to %s", output)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate local retrieval quality with relevance labels.")
    parser.add_argument("--documents-dir", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["keyword", "tfidf"], required=True)
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument("--top-k", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    report = evaluate_retrieval(
        documents_dir=args.documents_dir,
        queries_path=args.queries,
        output_path=args.output,
        backend=args.backend,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
    )
    metrics = report["metrics"]
    print(f"query_count: {metrics['query_count']}")
    print(f"evaluated_at_k: {metrics['evaluated_at_k']}")
    print(f"backend: {report['backend']}")
    print(f"output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
