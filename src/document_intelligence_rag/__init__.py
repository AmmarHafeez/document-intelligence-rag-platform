"""Local-first document retrieval components."""

from document_intelligence_rag.answering import (
    CitedSource,
    GroundedAnswer,
    SourcePreview,
    build_extractive_answer,
)
from document_intelligence_rag.chunking import split_document, split_documents
from document_intelligence_rag.evaluation import (
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)
from document_intelligence_rag.ingestion import load_document, load_documents
from document_intelligence_rag.models import Document, RetrievalResult, TextChunk
from document_intelligence_rag.retrieval import KeywordRetriever, TfidfRetriever

__all__ = [
    "Document",
    "GroundedAnswer",
    "KeywordRetriever",
    "CitedSource",
    "RetrievalResult",
    "SourcePreview",
    "TextChunk",
    "TfidfRetriever",
    "build_extractive_answer",
    "load_document",
    "load_documents",
    "mean_reciprocal_rank",
    "precision_at_k",
    "recall_at_k",
    "split_document",
    "split_documents",
]

__version__ = "0.1.0"
