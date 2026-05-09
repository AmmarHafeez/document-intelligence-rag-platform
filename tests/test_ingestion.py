import pytest

from document_intelligence_rag.ingestion import UnsupportedDocumentError, load_document, load_documents


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
    path = tmp_path / "data.pdf"
    path.write_text("not supported", encoding="utf-8")

    with pytest.raises(UnsupportedDocumentError, match="Unsupported document extension"):
        load_document(path)
