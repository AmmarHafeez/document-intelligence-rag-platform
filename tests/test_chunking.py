import pytest

from document_intelligence_rag.chunking import InvalidChunkingConfig, split_document
from document_intelligence_rag.models import Document


def test_chunking_uses_overlap(tmp_path):
    document = Document(
        document_id="doc-1",
        source_path=tmp_path / "doc.txt",
        title="doc",
        text="abcdefghijklmnopqrstuvwxyz",
    )

    chunks = split_document(document, chunk_size=10, chunk_overlap=3)

    assert [chunk.start_char for chunk in chunks] == [0, 7, 14, 21]
    assert [chunk.end_char for chunk in chunks] == [10, 17, 24, 26]
    assert chunks[1].text.startswith("hij")
    assert chunks[0].chunk_id == "doc-1:0000"


def test_invalid_chunk_parameters_raise(tmp_path):
    document = Document(
        document_id="doc-1",
        source_path=tmp_path / "doc.txt",
        title="doc",
        text="abc",
    )

    with pytest.raises(InvalidChunkingConfig, match="smaller than chunk_size"):
        split_document(document, chunk_size=10, chunk_overlap=10)

    with pytest.raises(InvalidChunkingConfig, match="greater than 0"):
        split_document(document, chunk_size=0, chunk_overlap=0)
