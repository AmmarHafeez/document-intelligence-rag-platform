from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Iterable

from document_intelligence_rag.models import RetrievalResult, TextChunk

logger = logging.getLogger(__name__)

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


class KeywordRetriever:
    backend = "keyword"

    def __init__(self, chunks: Iterable[TextChunk]) -> None:
        self._chunks = list(chunks)
        self._chunk_tokens = {
            chunk.chunk_id: Counter(tokenize(chunk.text))
            for chunk in self._chunks
        }
        logger.debug("Initialized keyword retriever with %d chunks", len(self._chunks))

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def _score(self, query_terms: set[str], chunk: TextChunk) -> tuple[float, list[str]]:
        counts = self._chunk_tokens[chunk.chunk_id]
        matched_terms = sorted(term for term in query_terms if counts.get(term, 0) > 0)
        if not matched_terms:
            return 0.0, []
        score = sum(counts[term] for term in matched_terms) / len(query_terms)
        return float(score), matched_terms

    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        query_terms = set(tokenize(query))
        if not query_terms:
            raise ValueError("query must contain at least one searchable token.")

        scored: list[tuple[float, list[str], TextChunk]] = []
        for chunk in self._chunks:
            score, matched_terms = self._score(query_terms, chunk)
            if score > 0:
                scored.append((score, matched_terms, chunk))

        scored.sort(key=lambda item: (-item[0], item[2].document_id, item[2].chunk_id))
        return [
            RetrievalResult(
                chunk=chunk,
                score=score,
                rank=index,
                matched_terms=matched_terms,
            )
            for index, (score, matched_terms, chunk) in enumerate(scored[:top_k], start=1)
        ]
