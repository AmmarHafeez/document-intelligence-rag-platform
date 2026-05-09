from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

RAG_INTRO_TEXT = """# Retrieval Augmented Generation

Retrieval augmented generation uses retrieved document chunks to ground answers in local source text.
It separates document search from answer construction so retrieval quality can be inspected directly.
"""

VECTOR_SEARCH_TEXT = """TF-IDF vector search represents documents and questions with weighted term vectors.
Cosine similarity ranks chunks that share important query terms with the question.
This local baseline is deterministic and does not require model downloads.
"""

PDF_INGESTION_TEXT = (
    "PDF ingestion extracts text from local text-based PDF files. "
    "Scanned-image PDFs require OCR and are not supported by this demo."
)


@dataclass(frozen=True, slots=True)
class DemoCorpusSummary:
    documents_written: list[Path]
    documents_skipped: list[Path]
    queries_written: int
    queries_path: Path
    documents_dir: Path


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def create_text_pdf_bytes(text: str) -> bytes:
    escaped_text = _escape_pdf_text(text)
    content = f"BT /F1 12 Tf 72 720 Td ({escaped_text}) Tj ET".encode("ascii")
    objects = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        (
            b"3 0 obj\n"
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>\n"
            b"endobj\n"
        ),
        b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
        (
            f"5 0 obj\n<< /Length {len(content)} >>\nstream\n".encode("ascii")
            + content
            + b"\nendstream\nendobj\n"
        ),
    ]
    pdf = b"%PDF-1.4\n"
    offsets = [0]
    for item in objects:
        offsets.append(len(pdf))
        pdf += item

    xref_offset = len(pdf)
    xref_entries = ["0000000000 65535 f \n"]
    xref_entries.extend(f"{offset:010d} 00000 n \n" for offset in offsets[1:])
    xref = f"xref\n0 {len(objects) + 1}\n{''.join(xref_entries)}"
    trailer = f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
    startxref = f"startxref\n{xref_offset}\n%%EOF\n"
    return pdf + xref.encode("ascii") + trailer.encode("ascii") + startxref.encode("ascii")


def _demo_documents(include_pdf: bool) -> dict[str, str | bytes]:
    documents: dict[str, str | bytes] = {
        "rag_intro.md": RAG_INTRO_TEXT,
        "vector_search.txt": VECTOR_SEARCH_TEXT,
    }
    if include_pdf:
        documents["pdf_ingestion_demo.pdf"] = create_text_pdf_bytes(PDF_INGESTION_TEXT)
    return documents


def _demo_queries(include_pdf: bool) -> dict[str, list[dict[str, Any]]]:
    queries = [
        {
            "query_id": "q_rag_intro",
            "query": "What is retrieval augmented generation?",
            "relevant_document_ids": ["rag_intro"],
            "relevant_chunk_ids": ["rag_intro:0000"],
        },
        {
            "query_id": "q_vector_search",
            "query": "How does TF-IDF vector search rank chunks?",
            "relevant_document_ids": ["vector_search"],
            "relevant_chunk_ids": ["vector_search:0000"],
        },
    ]
    if include_pdf:
        queries.append(
            {
                "query_id": "q_pdf_ingestion",
                "query": "What kind of PDF files can local ingestion read?",
                "relevant_document_ids": ["pdf_ingestion_demo"],
                "relevant_chunk_ids": ["pdf_ingestion_demo:0000"],
            }
        )
    return {"queries": queries}


def _write_file(path: Path, content: str | bytes, *, overwrite: bool) -> bool:
    if path.exists() and not overwrite:
        logger.info("Skipping existing file %s", path)
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    logger.info("Wrote demo file %s", path)
    return True


def create_demo_corpus(
    *,
    documents_dir: str | Path,
    queries_path: str | Path,
    include_pdf: bool = False,
    overwrite: bool = False,
) -> DemoCorpusSummary:
    resolved_documents_dir = Path(documents_dir)
    resolved_queries_path = Path(queries_path)
    resolved_documents_dir.mkdir(parents=True, exist_ok=True)
    resolved_queries_path.parent.mkdir(parents=True, exist_ok=True)

    documents_written: list[Path] = []
    documents_skipped: list[Path] = []
    for file_name, content in _demo_documents(include_pdf).items():
        path = resolved_documents_dir / file_name
        if _write_file(path, content, overwrite=overwrite):
            documents_written.append(path)
        else:
            documents_skipped.append(path)

    queries = _demo_queries(include_pdf)
    queries_content = json.dumps(queries, indent=2) + "\n"
    queries_written = (
        len(queries["queries"])
        if _write_file(resolved_queries_path, queries_content, overwrite=overwrite)
        else 0
    )

    return DemoCorpusSummary(
        documents_written=documents_written,
        documents_skipped=documents_skipped,
        queries_written=queries_written,
        queries_path=resolved_queries_path,
        documents_dir=resolved_documents_dir,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a tiny local demo corpus.")
    parser.add_argument("--documents-dir", type=Path, required=True)
    parser.add_argument("--queries-path", type=Path, required=True)
    parser.add_argument("--include-pdf", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _print_summary(summary: DemoCorpusSummary) -> None:
    print("documents_written: " + str(len(summary.documents_written)))
    print("documents_skipped: " + str(len(summary.documents_skipped)))
    print(f"queries_written: {summary.queries_written}")
    print(f"documents_dir: {summary.documents_dir}")
    print(f"queries_path: {summary.queries_path}")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    summary = create_demo_corpus(
        documents_dir=args.documents_dir,
        queries_path=args.queries_path,
        include_pdf=args.include_pdf,
        overwrite=args.overwrite,
    )
    _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
