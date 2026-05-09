import json

from document_intelligence_rag.retrieval.build_index import main as build_index_main
from document_intelligence_rag.retrieval.query_index import main as query_index_main


def test_build_index_cli_creates_index_file(tmp_path, capsys):
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()
    (documents_dir / "guide.txt").write_text(
        "Retrieval quality improves when chunks match the question.",
        encoding="utf-8",
    )
    index_path = tmp_path / "indexes" / "tfidf_index.joblib"

    exit_code = build_index_main(
        [
            "--documents-dir",
            str(documents_dir),
            "--index-path",
            str(index_path),
            "--chunk-size",
            "80",
            "--chunk-overlap",
            "10",
            "--backend",
            "tfidf",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert index_path.exists()
    assert "document_count: 1" in output
    assert "backend: tfidf" in output


def test_query_index_cli_prints_and_writes_json(tmp_path, capsys):
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()
    (documents_dir / "guide.txt").write_text(
        "Retrieval quality improves when chunks match the question.",
        encoding="utf-8",
    )
    index_path = tmp_path / "indexes" / "tfidf_index.joblib"
    output_path = tmp_path / "reports" / "artifacts" / "retrieval_results.json"
    build_index_main(
        [
            "--documents-dir",
            str(documents_dir),
            "--index-path",
            str(index_path),
            "--chunk-size",
            "80",
            "--chunk-overlap",
            "10",
        ]
    )
    capsys.readouterr()

    exit_code = query_index_main(
        [
            "--index-path",
            str(index_path),
            "--query",
            "retrieval quality",
            "--top-k",
            "1",
            "--output",
            str(output_path),
        ]
    )

    printed = capsys.readouterr().out
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert "score=" in printed
    assert output_path.exists()
    assert payload["query"] == "retrieval quality"
    assert payload["results"][0]["rank"] == 1
