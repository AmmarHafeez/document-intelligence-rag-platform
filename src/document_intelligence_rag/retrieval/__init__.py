from __future__ import annotations

__all__ = ["KeywordRetriever", "TfidfRetriever", "tokenize"]


def __getattr__(name: str) -> object:
    if name == "KeywordRetriever":
        from document_intelligence_rag.retrieval.keyword import KeywordRetriever

        return KeywordRetriever
    if name == "TfidfRetriever":
        from document_intelligence_rag.retrieval.tfidf import TfidfRetriever

        return TfidfRetriever
    if name == "tokenize":
        from document_intelligence_rag.retrieval.keyword import tokenize

        return tokenize
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
