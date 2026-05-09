from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from document_intelligence_rag.models import RetrievalResult, TextChunk
from document_intelligence_rag.retrieval.keyword import tokenize

logger = logging.getLogger(__name__)


class TfidfRetriever:
    backend = "tfidf"

    def __init__(self) -> None:
        self._chunks: list[TextChunk] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix: Any | None = None

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def document_count(self) -> int:
        return len({chunk.document_id for chunk in self._chunks})

    @property
    def chunks(self) -> tuple[TextChunk, ...]:
        return tuple(self._chunks)

    def fit(self, chunks: Iterable[TextChunk]) -> "TfidfRetriever":
        self._chunks = list(chunks)
        if not self._chunks:
            raise ValueError("Cannot fit TF-IDF retriever without chunks.")

        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b[A-Za-z0-9]+\b",
            norm="l2",
        )
        self._matrix = self._vectorizer.fit_transform([chunk.text for chunk in self._chunks])
        logger.info("Fit TF-IDF retriever with %d chunks", len(self._chunks))
        return self

    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")
        if not tokenize(query):
            raise ValueError("query must contain at least one searchable token.")
        if self._vectorizer is None or self._matrix is None:
            raise ValueError("TF-IDF retriever has not been fit or loaded.")

        query_vector = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self._matrix).ravel()
        query_terms = set(tokenize(query))

        ranked_indices = sorted(
            range(len(self._chunks)),
            key=lambda index: (
                -float(scores[index]),
                self._chunks[index].document_id,
                self._chunks[index].chunk_id,
            ),
        )

        results: list[RetrievalResult] = []
        for index in ranked_indices:
            score = float(scores[index])
            if score <= 0.0:
                continue
            chunk = self._chunks[index]
            chunk_terms = set(tokenize(chunk.text))
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=score,
                    rank=len(results) + 1,
                    matched_terms=sorted(query_terms & chunk_terms),
                )
            )
            if len(results) >= top_k:
                break

        return results

    def save(self, path: str | Path) -> Path:
        if self._vectorizer is None or self._matrix is None:
            raise ValueError("Cannot save TF-IDF retriever before fitting it.")

        index_path = Path(path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "backend": self.backend,
            "chunks": self._chunks,
            "vectorizer": self._vectorizer,
            "matrix": self._matrix,
        }
        joblib.dump(payload, index_path)
        logger.info("Saved TF-IDF index to %s", index_path)
        return index_path

    @classmethod
    def load(cls, path: str | Path) -> "TfidfRetriever":
        index_path = Path(path)
        payload = joblib.load(index_path)
        if not isinstance(payload, dict) or payload.get("backend") != cls.backend:
            raise ValueError(f"Index at {index_path} is not a TF-IDF index.")

        retriever = cls()
        retriever._chunks = list(payload["chunks"])
        retriever._vectorizer = payload["vectorizer"]
        retriever._matrix = payload["matrix"]
        logger.info("Loaded TF-IDF index from %s with %d chunks", index_path, retriever.chunk_count)
        return retriever
