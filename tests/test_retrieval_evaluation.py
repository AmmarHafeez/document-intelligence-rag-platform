import json

import pytest

from document_intelligence_rag.chunking import split_documents
from document_intelligence_rag.evaluation.evaluate_retrieval import (
    evaluate_retrieval,
    parse_evaluation_queries,
)
from document_intelligence_rag.ingestion import load_documents


def _write_documents(tmp_path):
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()
    (documents_dir / "finance.txt").write_text(
        "Invoice approval requires finance review.",
        encoding="utf-8",
    )
    (documents_dir / "retrieval.md").write_text(
        "# Retrieval\nRetrieval augmented generation uses relevant chunks.",
        encoding="utf-8",
    )
    documents = load_documents(documents_dir)
    chunks = split_documents(documents, chunk_size=120, chunk_overlap=10)
    return documents_dir, documents, chunks


def _write_queries(path, queries):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"queries": queries}), encoding="utf-8")


def test_valid_evaluation_json_is_parsed(tmp_path):
    query_path = tmp_path / "queries.json"
    _write_queries(
        query_path,
        [
            {
                "query_id": "q1",
                "query": "retrieval chunks",
                "relevant_document_ids": ["doc_rag"],
                "relevant_chunk_ids": ["doc_rag:0000"],
            }
        ],
    )

    queries = parse_evaluation_queries(query_path)

    assert len(queries) == 1
    assert queries[0].query_id == "q1"
    assert queries[0].relevant_document_ids == ("doc_rag",)
    assert queries[0].relevant_chunk_ids == ("doc_rag:0000",)


def test_invalid_query_file_raises_clear_error(tmp_path):
    query_path = tmp_path / "queries.json"
    query_path.write_text(json.dumps({"items": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="'queries' list"):
        parse_evaluation_queries(query_path)


def test_missing_query_text_raises_clear_error(tmp_path):
    query_path = tmp_path / "queries.json"
    _write_queries(query_path, [{"query_id": "q1", "relevant_document_ids": ["doc"]}])

    with pytest.raises(ValueError, match="non-empty 'query'"):
        parse_evaluation_queries(query_path)


def test_keyword_backend_evaluation_writes_metrics_and_per_query_results(tmp_path):
    documents_dir, documents, chunks = _write_documents(tmp_path)
    doc_by_name = {document.source_path.name: document for document in documents}
    chunk_by_doc = {chunk.document_id: chunk for chunk in chunks}
    query_path = tmp_path / "evaluation" / "queries.json"
    output_path = tmp_path / "reports" / "metrics" / "keyword_eval.json"
    _write_queries(
        query_path,
        [
            {
                "query_id": "q1",
                "query": "invoice approval",
                "relevant_document_ids": [doc_by_name["finance.txt"].document_id],
            },
            {
                "query_id": "q2",
                "query": "retrieval chunks",
                "relevant_chunk_ids": [
                    chunk_by_doc[doc_by_name["retrieval.md"].document_id].chunk_id
                ],
            },
        ],
    )

    report = evaluate_retrieval(
        documents_dir=documents_dir,
        queries_path=query_path,
        output_path=output_path,
        backend="keyword",
        chunk_size=120,
        chunk_overlap=10,
        top_k=2,
    )

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert output_path.exists()
    assert report["backend"] == "keyword"
    assert saved["metrics"]["query_count"] == 2
    assert saved["metrics"]["evaluated_at_k"] == 2
    assert saved["metrics"]["recall_at_k"] == 1.0
    assert saved["metrics"]["precision_at_k"] == 0.5
    assert saved["metrics"]["mean_reciprocal_rank"] == 1.0
    assert saved["queries"][0]["results"][0]["score"] > 0
    assert saved["queries"][1]["retrieved_chunk_ids"]


def test_tfidf_backend_evaluation_produces_metrics(tmp_path):
    documents_dir, documents, _chunks = _write_documents(tmp_path)
    doc_by_name = {document.source_path.name: document for document in documents}
    query_path = tmp_path / "evaluation" / "queries.json"
    output_path = tmp_path / "reports" / "metrics" / "tfidf_eval.json"
    _write_queries(
        query_path,
        [
            {
                "query_id": "q1",
                "query": "relevant retrieval chunks",
                "relevant_document_ids": [doc_by_name["retrieval.md"].document_id],
            }
        ],
    )

    report = evaluate_retrieval(
        documents_dir=documents_dir,
        queries_path=query_path,
        output_path=output_path,
        backend="tfidf",
        chunk_size=120,
        chunk_overlap=10,
        top_k=1,
    )

    assert output_path.exists()
    assert report["backend"] == "tfidf"
    assert report["metrics"]["query_count"] == 1
    assert report["metrics"]["recall_at_k"] == 1.0
    assert report["metrics"]["precision_at_k"] == 1.0
    assert report["metrics"]["mean_reciprocal_rank"] == 1.0
    assert report["queries"][0]["retrieved_document_ids"][0] == doc_by_name["retrieval.md"].document_id


def test_empty_relevance_labels_are_handled(tmp_path):
    documents_dir, _documents, _chunks = _write_documents(tmp_path)
    query_path = tmp_path / "evaluation" / "queries.json"
    output_path = tmp_path / "reports" / "metrics" / "empty_labels.json"
    _write_queries(query_path, [{"query_id": "q1", "query": "retrieval chunks"}])

    report = evaluate_retrieval(
        documents_dir=documents_dir,
        queries_path=query_path,
        output_path=output_path,
        backend="keyword",
        chunk_size=120,
        chunk_overlap=10,
        top_k=2,
    )

    assert report["metrics"]["recall_at_k"] == 0.0
    assert report["metrics"]["precision_at_k"] == 0.0
    assert report["metrics"]["mean_reciprocal_rank"] == 0.0


def test_invalid_evaluation_parameters_raise(tmp_path):
    documents_dir, _documents, _chunks = _write_documents(tmp_path)
    query_path = tmp_path / "evaluation" / "queries.json"
    output_path = tmp_path / "reports" / "metrics" / "invalid.json"
    _write_queries(query_path, [{"query_id": "q1", "query": "retrieval", "relevant_document_ids": []}])

    with pytest.raises(ValueError, match="top_k"):
        evaluate_retrieval(
            documents_dir=documents_dir,
            queries_path=query_path,
            output_path=output_path,
            backend="keyword",
            chunk_size=120,
            chunk_overlap=10,
            top_k=0,
        )

    with pytest.raises(ValueError, match="Unsupported retrieval backend"):
        evaluate_retrieval(
            documents_dir=documents_dir,
            queries_path=query_path,
            output_path=output_path,
            backend="unknown",
            chunk_size=120,
            chunk_overlap=10,
            top_k=1,
        )
