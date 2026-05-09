import json

from document_intelligence_rag.answering import build_extractive_answer
from document_intelligence_rag.answering.answer_query import main as answer_query_main
from document_intelligence_rag.models import RetrievalResult, TextChunk
from document_intelligence_rag.retrieval import TfidfRetriever


def _chunk(tmp_path, text: str) -> TextChunk:
    return TextChunk(
        chunk_id="doc_rag:0000",
        document_id="doc_rag",
        source_path=tmp_path / "doc_rag.txt",
        text=text,
        start_char=0,
        end_char=len(text),
    )


def test_answer_builder_selects_relevant_sentence_from_context(tmp_path):
    chunk = _chunk(
        tmp_path,
        "Invoices require approval. Retrieval augmented generation uses relevant chunks.",
    )
    result = RetrievalResult(
        chunk=chunk,
        score=0.9,
        rank=1,
        matched_terms=["retrieval", "augmented", "generation"],
    )

    answer = build_extractive_answer("What is retrieval augmented generation?", [result])

    assert answer.insufficient_context is False
    assert answer.answer == "Retrieval augmented generation uses relevant chunks."
    assert answer.cited_chunk_ids == ["doc_rag:0000"]
    assert answer.cited_document_ids == ["doc_rag"]
    assert answer.confidence_score > 0
    assert answer.source_previews[0].chunk_id == "doc_rag:0000"


def test_answer_builder_handles_insufficient_context():
    answer = build_extractive_answer("What is retrieval augmented generation?", [])

    assert answer.insufficient_context is True
    assert "Insufficient context" in answer.answer
    assert answer.cited_chunk_ids == []
    assert answer.cited_document_ids == []
    assert answer.confidence_score == 0.0


def test_answer_cli_writes_json(tmp_path, capsys):
    index_path = tmp_path / "indexes" / "tfidf_index.joblib"
    output_path = tmp_path / "reports" / "artifacts" / "answer_result.json"
    chunk = _chunk(
        tmp_path,
        "Retrieval augmented generation uses relevant chunks for grounded answers.",
    )
    TfidfRetriever().fit([chunk]).save(index_path)

    exit_code = answer_query_main(
        [
            "--index-path",
            str(index_path),
            "--query",
            "What is retrieval augmented generation?",
            "--top-k",
            "1",
            "--output",
            str(output_path),
        ]
    )

    printed = capsys.readouterr().out
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert output_path.exists()
    assert "answer:" in printed
    assert payload["insufficient_context"] is False
    assert payload["cited_chunk_ids"] == ["doc_rag:0000"]
    assert payload["cited_document_ids"] == ["doc_rag"]
    assert "Retrieval augmented generation" in payload["answer"]
