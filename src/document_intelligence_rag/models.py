from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Document:
    document_id: str
    source_path: Path
    title: str
    text: str
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TextChunk:
    chunk_id: str
    document_id: str
    source_path: Path
    text: str
    start_char: int
    end_char: int

    def metadata(self) -> dict[str, str | int]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "source_path": str(self.source_path),
            "start_char": self.start_char,
            "end_char": self.end_char,
        }


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    chunk: TextChunk
    score: float
    rank: int
    matched_terms: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "score": self.score,
            "matched_terms": self.matched_terms,
            "chunk": {
                "chunk_id": self.chunk.chunk_id,
                "document_id": self.chunk.document_id,
                "source_path": str(self.chunk.source_path),
                "start_char": self.chunk.start_char,
                "end_char": self.chunk.end_char,
                "text": self.chunk.text,
            },
        }
