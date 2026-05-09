from __future__ import annotations

from collections.abc import Iterable

from document_intelligence_rag.models import Document, TextChunk


class InvalidChunkingConfig(ValueError):
    """Raised when chunking parameters cannot produce stable chunks."""


def validate_chunking_config(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise InvalidChunkingConfig("chunk_size must be greater than 0.")
    if chunk_overlap < 0:
        raise InvalidChunkingConfig("chunk_overlap must be greater than or equal to 0.")
    if chunk_overlap >= chunk_size:
        raise InvalidChunkingConfig("chunk_overlap must be smaller than chunk_size.")


def split_document(
    document: Document,
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[TextChunk]:
    validate_chunking_config(chunk_size, chunk_overlap)

    text = document.text
    if not text:
        return []

    chunks: list[TextChunk] = []
    start = 0
    step = chunk_size - chunk_overlap

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_index = len(chunks)
        chunks.append(
            TextChunk(
                chunk_id=f"{document.document_id}:{chunk_index:04d}",
                document_id=document.document_id,
                source_path=document.source_path,
                text=text[start:end],
                start_char=start,
                end_char=end,
            )
        )

        if end >= len(text):
            break
        start += step

    return chunks


def split_documents(
    documents: Iterable[Document],
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[TextChunk]:
    validate_chunking_config(chunk_size, chunk_overlap)
    chunks: list[TextChunk] = []
    for document in documents:
        chunks.extend(
            split_document(
                document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )
    return chunks
