from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from document_intelligence_rag.retrieval.keyword import tokenize

logger = logging.getLogger(__name__)

SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
DEFAULT_SUPPORT_THRESHOLD = 0.8


@dataclass(frozen=True, slots=True)
class GroundingSource:
    chunk_id: str | None
    document_id: str | None
    text: str
    context_type: str


def split_answer_sentences(answer: str) -> list[str]:
    sentences = [sentence.strip() for sentence in SENTENCE_BOUNDARY_PATTERN.split(answer)]
    return [sentence for sentence in sentences if sentence]


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _token_overlap_score(sentence: str, context: str) -> float:
    sentence_terms = set(tokenize(sentence))
    if not sentence_terms:
        return 0.0
    context_terms = set(tokenize(context))
    return len(sentence_terms & context_terms) / len(sentence_terms)


def _source_from_mapping(item: dict[str, Any], *, context_type: str, text_field: str) -> GroundingSource:
    return GroundingSource(
        chunk_id=item.get("chunk_id") if isinstance(item.get("chunk_id"), str) else None,
        document_id=item.get("document_id") if isinstance(item.get("document_id"), str) else None,
        text=str(item.get(text_field, "")),
        context_type=context_type,
    )


def _extract_sources(answer_record: dict[str, Any]) -> tuple[list[GroundingSource], bool]:
    cited_sources = answer_record.get("cited_sources")
    if isinstance(cited_sources, list):
        full_text_sources = [
            _source_from_mapping(source, context_type="cited_sources.text", text_field="text")
            for source in cited_sources
            if isinstance(source, dict) and source.get("text")
        ]
        if full_text_sources:
            return full_text_sources, True

    source_previews = answer_record.get("source_previews")
    if isinstance(source_previews, list):
        preview_sources = [
            _source_from_mapping(source, context_type="source_previews.preview", text_field="preview")
            for source in source_previews
            if isinstance(source, dict) and source.get("preview")
        ]
        if preview_sources:
            return preview_sources, False

    return [], False


def _best_support(sentence: str, sources: list[GroundingSource]) -> dict[str, Any]:
    best_source: GroundingSource | None = None
    best_score = 0.0
    exact_match = False
    normalized_sentence = _normalize_text(sentence)

    for source in sources:
        normalized_context = _normalize_text(source.text)
        if normalized_sentence and normalized_sentence in normalized_context:
            return {
                "support_score": 1.0,
                "exact_match": True,
                "chunk_id": source.chunk_id,
                "document_id": source.document_id,
                "context_type": source.context_type,
            }

        score = _token_overlap_score(sentence, source.text)
        if score > best_score:
            best_score = score
            best_source = source

    if best_source is not None:
        exact_match = False
        return {
            "support_score": best_score,
            "exact_match": exact_match,
            "chunk_id": best_source.chunk_id,
            "document_id": best_source.document_id,
            "context_type": best_source.context_type,
        }

    return {
        "support_score": 0.0,
        "exact_match": False,
        "chunk_id": None,
        "document_id": None,
        "context_type": None,
    }


def _list_strings(answer_record: dict[str, Any], *field_names: str) -> list[str]:
    for field_name in field_names:
        values = answer_record.get(field_name)
        if isinstance(values, list):
            return [value for value in values if isinstance(value, str)]
    return []


def _coverage(used_ids: set[str], cited_ids: set[str]) -> float | None:
    if not cited_ids:
        return None
    return len(used_ids & cited_ids) / len(cited_ids)


def _citation_coverage(
    *,
    answer_record: dict[str, Any],
    supported_sentence_results: list[dict[str, Any]],
    sources: list[GroundingSource],
) -> dict[str, float | None]:
    cited_chunk_ids = set(_list_strings(answer_record, "cited_chunk_ids", "cited_chunks"))
    cited_document_ids = set(_list_strings(answer_record, "cited_document_ids", "cited_documents"))
    if not cited_chunk_ids:
        cited_chunk_ids = {source.chunk_id for source in sources if source.chunk_id}
    if not cited_document_ids:
        cited_document_ids = {source.document_id for source in sources if source.document_id}

    used_chunk_ids = {
        result["chunk_id"]
        for result in supported_sentence_results
        if isinstance(result.get("chunk_id"), str)
    }
    used_document_ids = {
        result["document_id"]
        for result in supported_sentence_results
        if isinstance(result.get("document_id"), str)
    }
    chunk_coverage = _coverage(used_chunk_ids, cited_chunk_ids)
    document_coverage = _coverage(used_document_ids, cited_document_ids)
    coverage_values = [
        value for value in (chunk_coverage, document_coverage) if value is not None
    ]

    return {
        "citation_coverage": (
            sum(coverage_values) / len(coverage_values) if coverage_values else None
        ),
        "cited_chunk_coverage": chunk_coverage,
        "cited_document_coverage": document_coverage,
    }


def evaluate_answer_grounding(
    answer_record: dict[str, Any],
    *,
    support_threshold: float = DEFAULT_SUPPORT_THRESHOLD,
) -> dict[str, Any]:
    if not 0.0 <= support_threshold <= 1.0:
        raise ValueError("support_threshold must be between 0.0 and 1.0.")

    if bool(answer_record.get("insufficient_context")):
        return {
            "query": answer_record.get("query"),
            "insufficient_context": True,
            "context_available": False,
            "full_text_available": False,
            "metrics": {
                "sentence_support_rate": None,
                "citation_coverage": None,
                "cited_chunk_coverage": None,
                "cited_document_coverage": None,
            },
            "sentence_support": [],
            "unsupported_sentences": [],
        }

    answer = str(answer_record.get("answer", ""))
    sentences = split_answer_sentences(answer)
    sources, full_text_available = _extract_sources(answer_record)
    sentence_results: list[dict[str, Any]] = []
    supported_sentence_results: list[dict[str, Any]] = []
    unsupported_sentences: list[str] = []

    for sentence in sentences:
        support = _best_support(sentence, sources)
        supported = bool(support["support_score"] >= support_threshold)
        result = {
            "sentence": sentence,
            "support_score": support["support_score"],
            "supported": supported,
            "exact_match": support["exact_match"],
            "best_source_chunk_id": support["chunk_id"],
            "best_source_document_id": support["document_id"],
            "context_type": support["context_type"],
        }
        sentence_results.append(result)
        if supported:
            supported_sentence_results.append(
                {
                    "chunk_id": support["chunk_id"],
                    "document_id": support["document_id"],
                }
            )
        else:
            unsupported_sentences.append(sentence)

    sentence_support_rate = (
        len(supported_sentence_results) / len(sentences) if sentences else 0.0
    )
    coverage = _citation_coverage(
        answer_record=answer_record,
        supported_sentence_results=supported_sentence_results,
        sources=sources,
    )

    return {
        "query": answer_record.get("query"),
        "insufficient_context": False,
        "context_available": bool(sources),
        "full_text_available": full_text_available,
        "metrics": {
            "sentence_support_rate": sentence_support_rate,
            **coverage,
        },
        "sentence_support": sentence_results,
        "unsupported_sentences": unsupported_sentences,
    }


def summarize_grounding_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    answer_count = len(results)
    applicable_results = [
        result for result in results if not result.get("insufficient_context")
    ]
    insufficient_context_count = answer_count - len(applicable_results)

    support_values = [
        result["metrics"]["sentence_support_rate"]
        for result in applicable_results
        if result["metrics"]["sentence_support_rate"] is not None
    ]
    citation_values = [
        result["metrics"]["citation_coverage"]
        for result in applicable_results
        if result["metrics"]["citation_coverage"] is not None
    ]

    return {
        "answer_count": answer_count,
        "evaluated_answer_count": len(applicable_results),
        "insufficient_context_count": insufficient_context_count,
        "sentence_support_rate": (
            sum(support_values) / len(support_values) if support_values else 0.0
        ),
        "citation_coverage": (
            sum(citation_values) / len(citation_values) if citation_values else 0.0
        ),
        "unsupported_sentence_count": sum(
            len(result["unsupported_sentences"]) for result in applicable_results
        ),
        "full_text_available_count": sum(
            1 for result in applicable_results if result.get("full_text_available")
        ),
    }
