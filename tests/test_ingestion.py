import pytest

from document_intelligence_rag.chunking import split_document
from document_intelligence_rag.ingestion import (
    PDFExtractionError,
    UnsupportedDocumentError,
    load_document,
    load_documents,
)


def _minimal_pdf_bytes(text: str) -> bytes:
    escaped_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
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


def test_text_and_markdown_ingestion(tmp_path):
    text_path = tmp_path / "alpha.txt"
    markdown_path = tmp_path / "beta.md"
    text_path.write_text("Alpha title\nBody text.", encoding="utf-8")
    markdown_path.write_text("# Beta Title\nMarkdown body.", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert len(documents) == 2
    assert documents[0].title == "Alpha title"
    assert documents[0].text == "Alpha title\nBody text."
    assert documents[1].title == "Beta Title"
    assert documents[1].source_path == markdown_path
    assert documents[0].document_id != documents[1].document_id


def test_unsupported_file_rejection(tmp_path):
    path = tmp_path / "data.docx"
    path.write_text("not supported", encoding="utf-8")

    with pytest.raises(UnsupportedDocumentError, match="Unsupported document extension"):
        load_document(path)


def test_pdf_ingestion_extracts_text_successfully(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(_minimal_pdf_bytes("PDF ingestion extracts local text."))

    document = load_document(pdf_path)

    assert document.document_id == "sample"
    assert document.source_path == pdf_path
    assert document.metadata["file_type"] == "pdf"
    assert document.metadata["page_count"] == 1
    assert "PDF ingestion extracts local text" in document.text


def test_unreadable_pdf_raises_clear_error(tmp_path):
    pdf_path = tmp_path / "broken.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\nnot a valid pdf\n%%EOF\n")

    with pytest.raises(PDFExtractionError, match="Could not read PDF document"):
        load_document(pdf_path)


def test_chunking_pdf_ingested_text(tmp_path):
    pdf_path = tmp_path / "chunkable.pdf"
    pdf_path.write_bytes(
        _minimal_pdf_bytes("PDF ingestion produces text that can be chunked for retrieval.")
    )
    document = load_document(pdf_path)

    chunks = split_document(document, chunk_size=30, chunk_overlap=5)

    assert chunks
    assert chunks[0].document_id == "chunkable"
    assert "PDF ingestion" in chunks[0].text
