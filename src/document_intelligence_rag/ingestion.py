from __future__ import annotations

import logging
import re
from pathlib import Path

from document_intelligence_rag.models import Document

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}
DOCUMENT_ID_PATTERN = re.compile(r"[^a-z0-9_]+")


class UnsupportedDocumentError(ValueError):
    """Raised when a document file extension is not supported."""


class PDFExtractionError(ValueError):
    """Raised when PDF text cannot be extracted locally."""


def _supported_extensions_message() -> str:
    return ", ".join(sorted(SUPPORTED_EXTENSIONS))


def _document_id_for(path: Path) -> str:
    stem = path.stem.strip().lower().replace("-", "_")
    document_id = DOCUMENT_ID_PATTERN.sub("_", stem).strip("_")
    return document_id or "document"


def _title_from_markdown(text: str, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            if title:
                return title
    return fallback


def _title_from_text(text: str, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:120]
    return fallback


def _derive_title(path: Path, text: str) -> str:
    fallback = path.stem.replace("_", " ").replace("-", " ").strip() or path.name
    if path.suffix.lower() == ".md":
        return _title_from_markdown(text, fallback)
    return _title_from_text(text, fallback)


def _extract_pdf_text(path: Path) -> tuple[str, dict[str, str | int | float | bool]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise PDFExtractionError("pypdf is required to ingest PDF documents.") from exc

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise PDFExtractionError(f"Could not read PDF document {path}: {exc}") from exc

    if reader.is_encrypted:
        raise PDFExtractionError(f"PDF is encrypted and cannot be read without a password: {path}")

    page_texts: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:
            raise PDFExtractionError(
                f"Could not extract text from PDF page {page_number} in {path}: {exc}"
            ) from exc
        if page_text.strip():
            page_texts.append(page_text.strip())

    text = "\n\n".join(page_texts).strip()
    if not text:
        raise PDFExtractionError(
            f"PDF has no extractable text: {path}. Scanned-image PDFs and OCR are not supported."
        )

    return text, {"file_type": "pdf", "page_count": len(reader.pages)}


def _read_document_text(path: Path) -> tuple[str, dict[str, str | int | float | bool]]:
    if path.suffix.lower() == ".pdf":
        return _extract_pdf_text(path)
    return path.read_text(encoding="utf-8"), {"file_type": path.suffix.lower().lstrip(".")}


def load_document(path: str | Path) -> Document:
    source_path = Path(path)
    extension = source_path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise UnsupportedDocumentError(
            f"Unsupported document extension '{extension}' for {source_path}. "
            f"Supported extensions: {_supported_extensions_message()}."
        )
    if not source_path.exists():
        raise FileNotFoundError(f"Document does not exist: {source_path}")
    if not source_path.is_file():
        raise ValueError(f"Document path is not a file: {source_path}")

    text, metadata = _read_document_text(source_path)
    document = Document(
        document_id=_document_id_for(source_path),
        source_path=source_path,
        title=_derive_title(source_path, text),
        text=text,
        metadata=metadata,
    )
    logger.debug("Loaded document %s from %s", document.document_id, source_path)
    return document


def iter_document_paths(directory: str | Path) -> list[Path]:
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"Document directory does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"Document path is not a directory: {root}")
    return sorted((path for path in root.rglob("*") if path.is_file()), key=lambda path: path.as_posix())


def load_documents(directory: str | Path) -> list[Document]:
    documents: list[Document] = []
    for path in iter_document_paths(directory):
        document = load_document(path)
        if any(existing.document_id == document.document_id for existing in documents):
            raise ValueError(f"Duplicate document_id '{document.document_id}' found under {directory}.")
        documents.append(document)
    return documents
