from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable

from document_intelligence_rag.models import RetrievalResult
from document_intelligence_rag.retrieval.keyword import tokenize
from document_intelligence_rag.retrieval.preview import preview_text

logger = logging.getLogger(__name__)

SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


@dataclass(frozen=True, slots=True)
class CitedSource:
    chunk_id: str
    document_id: str
    source_path: str
    text: str
    preview: str
    score: float

    def to_dict(self) -> dict[str, object]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "source_path": self.source_path,
            "text": self.text,
            "preview": self.preview,
            "score": self.score,
        }


@dataclass(frozen=True, slots=True)
class SourcePreview:
    chunk_id: str
    document_id: str
    source_path: str
    preview: str
    score: float

    def to_dict(self) -> dict[str, object]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "source_path": self.source_path,
            "preview": self.preview,
            "score": self.score,
        }


@dataclass(frozen=True, slots=True)
class GroundedAnswer:
    answer: str
    cited_chunk_ids: list[str]
    cited_document_ids: list[str]
    cited_sources: list[CitedSource]
    source_previews: list[SourcePreview]
    confidence_score: float
    insufficient_context: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "answer": self.answer,
            "cited_chunk_ids": self.cited_chunk_ids,
            "cited_document_ids": self.cited_document_ids,
            "cited_sources": [source.to_dict() for source in self.cited_sources],
            "source_previews": [source.to_dict() for source in self.source_previews],
            "confidence_score": self.confidence_score,
            "insufficient_context": self.insufficient_context,
        }


@dataclass(frozen=True, slots=True)
class _SentenceCandidate:
    sentence: str
    result: RetrievalResult
    sentence_index: int
    overlap_terms: set[str]

    @property
    def score(self) -> tuple[int, float]:
        return (len(self.overlap_terms), self.result.score)


def _question_terms(question: str) -> set[str]:
    terms = set(tokenize(question))
    content_terms = terms - STOPWORDS
    return content_terms or terms


def _split_sentences(text: str) -> list[str]:
    sentences = [sentence.strip() for sentence in SENTENCE_BOUNDARY_PATTERN.split(text)]
    return [sentence for sentence in sentences if sentence]


def _unique_append(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _insufficient_answer(results: list[RetrievalResult] | None = None) -> GroundedAnswer:
    source_previews = _source_previews(results or [])
    return GroundedAnswer(
        answer="Insufficient context to answer the question from retrieved chunks.",
        cited_chunk_ids=[],
        cited_document_ids=[],
        cited_sources=[],
        source_previews=source_previews,
        confidence_score=0.0,
        insufficient_context=True,
    )


def _source_previews(results: Iterable[RetrievalResult]) -> list[SourcePreview]:
    previews: list[SourcePreview] = []
    seen_chunks: set[str] = set()
    for result in results:
        chunk = result.chunk
        if chunk.chunk_id in seen_chunks:
            continue
        previews.append(
            SourcePreview(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                source_path=str(chunk.source_path),
                preview=preview_text(chunk.text),
                score=result.score,
            )
        )
        seen_chunks.add(chunk.chunk_id)
    return previews


def _cited_sources(results: Iterable[RetrievalResult]) -> list[CitedSource]:
    sources: list[CitedSource] = []
    seen_chunks: set[str] = set()
    for result in results:
        chunk = result.chunk
        if chunk.chunk_id in seen_chunks:
            continue
        preview = preview_text(chunk.text)
        sources.append(
            CitedSource(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                source_path=str(chunk.source_path),
                text=chunk.text,
                preview=preview,
                score=result.score,
            )
        )
        seen_chunks.add(chunk.chunk_id)
    return sources


def _rank_candidates(
    question_terms: set[str],
    results: list[RetrievalResult],
) -> list[_SentenceCandidate]:
    candidates: list[_SentenceCandidate] = []
    for result in results:
        for index, sentence in enumerate(_split_sentences(result.chunk.text)):
            sentence_terms = set(tokenize(sentence))
            overlap = question_terms & sentence_terms
            if overlap:
                candidates.append(
                    _SentenceCandidate(
                        sentence=sentence,
                        result=result,
                        sentence_index=index,
                        overlap_terms=overlap,
                    )
                )

    return sorted(
        candidates,
        key=lambda candidate: (
            -candidate.score[0],
            -candidate.score[1],
            candidate.result.rank,
            candidate.sentence_index,
        ),
    )


def build_extractive_answer(
    question: str,
    retrieved_results: Iterable[RetrievalResult],
    *,
    max_sentences: int = 3,
    max_answer_chars: int = 700,
) -> GroundedAnswer:
    results = list(retrieved_results)
    if not results:
        logger.info("No retrieved chunks available for answer generation")
        return _insufficient_answer()

    question_terms = _question_terms(question)
    if not question_terms:
        return _insufficient_answer(results)

    ranked_candidates = _rank_candidates(question_terms, results)
    if not ranked_candidates:
        logger.info("Retrieved chunks did not overlap with question terms")
        return _insufficient_answer(results)

    selected: list[_SentenceCandidate] = []
    answer_length = 0
    for candidate in ranked_candidates:
        proposed_length = answer_length + len(candidate.sentence) + (1 if selected else 0)
        if selected and proposed_length > max_answer_chars:
            continue
        selected.append(candidate)
        answer_length = proposed_length
        if len(selected) >= max_sentences:
            break

    selected.sort(key=lambda candidate: (candidate.result.rank, candidate.sentence_index))

    cited_chunk_ids: list[str] = []
    cited_document_ids: list[str] = []
    covered_terms: set[str] = set()
    for candidate in selected:
        covered_terms.update(candidate.overlap_terms)
        _unique_append(cited_chunk_ids, candidate.result.chunk.chunk_id)
        _unique_append(cited_document_ids, candidate.result.chunk.document_id)

    confidence_score = len(covered_terms) / len(question_terms)
    selected_chunk_ids = set(cited_chunk_ids)
    selected_results = [result for result in results if result.chunk.chunk_id in selected_chunk_ids]

    return GroundedAnswer(
        answer=" ".join(candidate.sentence for candidate in selected),
        cited_chunk_ids=cited_chunk_ids,
        cited_document_ids=cited_document_ids,
        cited_sources=_cited_sources(selected_results),
        source_previews=_source_previews(selected_results),
        confidence_score=confidence_score,
        insufficient_context=False,
    )
