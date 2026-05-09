import json

from document_intelligence_rag.evaluation.evaluate_grounding import main as grounding_main
from document_intelligence_rag.evaluation.grounding import evaluate_answer_grounding


def _answer_record(answer: str):
    return {
        "query": "What is retrieval augmented generation?",
        "answer": answer,
        "cited_chunk_ids": ["doc_rag:0000"],
        "cited_document_ids": ["doc_rag"],
        "cited_sources": [
            {
                "chunk_id": "doc_rag:0000",
                "document_id": "doc_rag",
                "source_path": "doc_rag.txt",
                "text": "Retrieval augmented generation uses relevant chunks.",
                "preview": "Retrieval augmented generation uses relevant chunks.",
                "score": 0.9,
            }
        ],
        "source_previews": [
            {
                "chunk_id": "doc_rag:0000",
                "document_id": "doc_rag",
                "source_path": "doc_rag.txt",
                "preview": "Retrieval augmented generation uses relevant chunks.",
                "score": 0.9,
            }
        ],
        "insufficient_context": False,
    }


def test_fully_supported_answer_gets_high_support_rate():
    result = evaluate_answer_grounding(
        _answer_record("Retrieval augmented generation uses relevant chunks.")
    )

    assert result["metrics"]["sentence_support_rate"] == 1.0
    assert result["metrics"]["citation_coverage"] == 1.0
    assert result["unsupported_sentences"] == []
    assert result["sentence_support"][0]["exact_match"] is True


def test_unsupported_sentence_is_detected():
    result = evaluate_answer_grounding(
        _answer_record(
            "Retrieval augmented generation uses relevant chunks. It always improves every answer."
        )
    )

    assert result["metrics"]["sentence_support_rate"] == 0.5
    assert result["unsupported_sentences"] == ["It always improves every answer."]


def test_citation_coverage_is_computed():
    record = _answer_record("Retrieval augmented generation uses relevant chunks.")
    record["cited_chunk_ids"] = ["doc_rag:0000", "doc_unused:0000"]
    record["cited_document_ids"] = ["doc_rag", "doc_unused"]
    record["cited_sources"].append(
        {
            "chunk_id": "doc_unused:0000",
            "document_id": "doc_unused",
            "source_path": "doc_unused.txt",
            "text": "Invoices require finance approval.",
            "preview": "Invoices require finance approval.",
            "score": 0.2,
        }
    )

    result = evaluate_answer_grounding(record)

    assert result["metrics"]["cited_chunk_coverage"] == 0.5
    assert result["metrics"]["cited_document_coverage"] == 0.5
    assert result["metrics"]["citation_coverage"] == 0.5


def test_preview_fallback_is_used_when_full_text_is_missing():
    record = _answer_record("Retrieval augmented generation uses relevant chunks.")
    del record["cited_sources"]

    result = evaluate_answer_grounding(record)

    assert result["full_text_available"] is False
    assert result["context_available"] is True
    assert result["sentence_support"][0]["context_type"] == "source_previews.preview"
    assert result["metrics"]["sentence_support_rate"] == 1.0


def test_insufficient_context_response_is_handled_cleanly():
    result = evaluate_answer_grounding(
        {
            "query": "What is retrieval augmented generation?",
            "answer": "Insufficient context to answer the question from retrieved chunks.",
            "cited_chunk_ids": [],
            "cited_document_ids": [],
            "source_previews": [],
            "insufficient_context": True,
        }
    )

    assert result["insufficient_context"] is True
    assert result["metrics"]["sentence_support_rate"] is None
    assert result["metrics"]["citation_coverage"] is None
    assert result["unsupported_sentences"] == []


def test_grounding_cli_writes_json(tmp_path, capsys):
    answers_path = tmp_path / "reports" / "artifacts" / "answer_result.json"
    output_path = tmp_path / "reports" / "metrics" / "grounding_eval.json"
    answers_path.parent.mkdir(parents=True)
    answers_path.write_text(
        json.dumps(_answer_record("Retrieval augmented generation uses relevant chunks.")),
        encoding="utf-8",
    )

    exit_code = grounding_main(
        [
            "--answers",
            str(answers_path),
            "--output",
            str(output_path),
        ]
    )

    printed = capsys.readouterr().out
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert output_path.exists()
    assert "sentence_support_rate" in printed
    assert report["metrics"]["answer_count"] == 1
    assert report["answers"][0]["metrics"]["sentence_support_rate"] == 1.0
