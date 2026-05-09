from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from document_intelligence_rag.chunking import split_documents
from document_intelligence_rag.config import AppConfig, load_config
from document_intelligence_rag.ingestion import load_documents
from document_intelligence_rag.models import RetrievalResult
from document_intelligence_rag.retrieval import KeywordRetriever

logger = logging.getLogger(__name__)


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=100)


class ChunkMatchResponse(BaseModel):
    rank: int
    score: float
    chunk_id: str
    document_id: str
    source_path: str
    start_char: int
    end_char: int
    text: str
    matched_terms: list[str]


class RetrieveResponse(BaseModel):
    query: str
    top_k: int
    results: list[ChunkMatchResponse]


@dataclass(slots=True)
class IndexState:
    config: AppConfig
    retriever: KeywordRetriever | None
    document_count: int
    chunk_count: int
    error: str | None = None

    @property
    def ready(self) -> bool:
        return self.retriever is not None and self.error is None


def _build_index_state(config: AppConfig) -> IndexState:
    documents_dir = Path(config.documents_dir)
    if not documents_dir.exists():
        message = f"No document directory found at {documents_dir}."
        logger.info(message)
        return IndexState(config=config, retriever=None, document_count=0, chunk_count=0, error=message)

    try:
        documents = load_documents(documents_dir)
        if not documents:
            message = f"No supported documents found in {documents_dir}."
            logger.info(message)
            return IndexState(config=config, retriever=None, document_count=0, chunk_count=0, error=message)

        chunks = split_documents(
            documents,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        if not chunks:
            message = f"No chunks could be built from documents in {documents_dir}."
            logger.info(message)
            return IndexState(
                config=config,
                retriever=None,
                document_count=len(documents),
                chunk_count=0,
                error=message,
            )

        retriever = KeywordRetriever(chunks)
        return IndexState(
            config=config,
            retriever=retriever,
            document_count=len(documents),
            chunk_count=len(chunks),
        )
    except Exception as exc:
        message = f"Retrieval index is not available: {exc}"
        logger.exception(message)
        return IndexState(config=config, retriever=None, document_count=0, chunk_count=0, error=message)


def _result_to_response(result: RetrievalResult) -> ChunkMatchResponse:
    chunk = result.chunk
    return ChunkMatchResponse(
        rank=result.rank,
        score=result.score,
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        source_path=str(chunk.source_path),
        start_char=chunk.start_char,
        end_char=chunk.end_char,
        text=chunk.text,
        matched_terms=result.matched_terms,
    )


def create_app(config: AppConfig | None = None) -> FastAPI:
    resolved_config = config or load_config()
    state = _build_index_state(resolved_config)

    app = FastAPI(
        title="Document Intelligence RAG Platform",
        version="0.1.0",
        description="Local-first document retrieval API.",
    )
    app.state.index_state = state

    @app.get("/health")
    def health() -> dict[str, object]:
        current_state: IndexState = app.state.index_state
        return {
            "status": "ok",
            "index_ready": current_state.ready,
            "document_count": current_state.document_count,
            "chunk_count": current_state.chunk_count,
            "error": current_state.error,
        }

    @app.post("/retrieve", response_model=RetrieveResponse)
    def retrieve(request: RetrieveRequest) -> RetrieveResponse:
        current_state: IndexState = app.state.index_state
        if current_state.retriever is None:
            raise HTTPException(
                status_code=503,
                detail=current_state.error or "Retrieval index is not available.",
            )

        top_k = request.top_k or current_state.config.top_k
        results = current_state.retriever.retrieve(request.query, top_k=top_k)
        return RetrieveResponse(
            query=request.query,
            top_k=top_k,
            results=[_result_to_response(result) for result in results],
        )

    return app


app = create_app()
