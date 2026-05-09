import json

from document_intelligence_rag.chunking import split_documents
from document_intelligence_rag.ingestion import load_document, load_documents
from document_intelligence_rag.ingestion.create_demo_corpus import create_demo_corpus, main


def test_demo_corpus_files_are_created(tmp_path, capsys):
    documents_dir = tmp_path / "data" / "raw" / "documents"
    queries_path = tmp_path / "data" / "raw" / "evaluation" / "queries.json"

    exit_code = main(
        [
            "--documents-dir",
            str(documents_dir),
            "--queries-path",
            str(queries_path),
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert (documents_dir / "rag_intro.md").exists()
    assert (documents_dir / "vector_search.txt").exists()
    assert queries_path.exists()
    assert "documents_written: 2" in output
    assert "queries_written: 2" in output


def test_queries_match_ingested_document_and_chunk_ids(tmp_path):
    documents_dir = tmp_path / "data" / "raw" / "documents"
    queries_path = tmp_path / "data" / "raw" / "evaluation" / "queries.json"
    create_demo_corpus(
        documents_dir=documents_dir,
        queries_path=queries_path,
        include_pdf=True,
    )

    documents = load_documents(documents_dir)
    chunks = split_documents(documents, chunk_size=800, chunk_overlap=120)
    document_ids = {document.document_id for document in documents}
    chunk_ids = {chunk.chunk_id for chunk in chunks}
    payload = json.loads(queries_path.read_text(encoding="utf-8"))

    for query in payload["queries"]:
        assert set(query["relevant_document_ids"]).issubset(document_ids)
        assert set(query["relevant_chunk_ids"]).issubset(chunk_ids)


def test_existing_files_are_not_overwritten_without_overwrite(tmp_path):
    documents_dir = tmp_path / "data" / "raw" / "documents"
    queries_path = tmp_path / "data" / "raw" / "evaluation" / "queries.json"
    documents_dir.mkdir(parents=True)
    queries_path.parent.mkdir(parents=True)
    existing_path = documents_dir / "rag_intro.md"
    existing_path.write_text("Existing local content.", encoding="utf-8")
    queries_path.write_text('{"queries":[]}', encoding="utf-8")

    summary = create_demo_corpus(
        documents_dir=documents_dir,
        queries_path=queries_path,
        overwrite=False,
    )

    assert existing_path.read_text(encoding="utf-8") == "Existing local content."
    assert queries_path.read_text(encoding="utf-8") == '{"queries":[]}'
    assert existing_path in summary.documents_skipped
    assert summary.queries_written == 0


def test_overwrite_replaces_existing_files(tmp_path):
    documents_dir = tmp_path / "data" / "raw" / "documents"
    queries_path = tmp_path / "data" / "raw" / "evaluation" / "queries.json"
    documents_dir.mkdir(parents=True)
    existing_path = documents_dir / "rag_intro.md"
    existing_path.write_text("Existing local content.", encoding="utf-8")

    summary = create_demo_corpus(
        documents_dir=documents_dir,
        queries_path=queries_path,
        overwrite=True,
    )

    assert "Retrieval augmented generation" in existing_path.read_text(encoding="utf-8")
    assert existing_path in summary.documents_written


def test_include_pdf_creates_readable_pdf(tmp_path):
    documents_dir = tmp_path / "data" / "raw" / "documents"
    queries_path = tmp_path / "data" / "raw" / "evaluation" / "queries.json"
    create_demo_corpus(
        documents_dir=documents_dir,
        queries_path=queries_path,
        include_pdf=True,
    )

    pdf_document = load_document(documents_dir / "pdf_ingestion_demo.pdf")
    payload = json.loads(queries_path.read_text(encoding="utf-8"))

    assert pdf_document.document_id == "pdf_ingestion_demo"
    assert pdf_document.metadata["page_count"] == 1
    assert "PDF ingestion extracts text" in pdf_document.text
    assert any(
        query["relevant_document_ids"] == ["pdf_ingestion_demo"]
        for query in payload["queries"]
    )
