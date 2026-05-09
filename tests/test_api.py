from fastapi.testclient import TestClient

from document_intelligence_rag.api import create_app
from document_intelligence_rag.config import AppConfig


def test_health_endpoint_works_without_documents(tmp_path):
    app = create_app(AppConfig(documents_dir=tmp_path / "missing"))
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["index_ready"] is False


def test_retrieve_endpoint_returns_ranked_chunks(tmp_path):
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()
    (documents_dir / "alpha.txt").write_text(
        "Alpha retrieval systems rank relevant chunks first.",
        encoding="utf-8",
    )
    (documents_dir / "beta.md").write_text(
        "# Beta\nChunk overlap keeps context available.",
        encoding="utf-8",
    )
    app = create_app(
        AppConfig(
            documents_dir=documents_dir,
            chunk_size=80,
            chunk_overlap=10,
            top_k=2,
        )
    )
    client = TestClient(app)

    response = client.post("/retrieve", json={"query": "chunk overlap", "top_k": 1})

    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "chunk overlap"
    assert payload["top_k"] == 1
    assert len(payload["results"]) == 1
    assert payload["results"][0]["score"] > 0
    assert "Chunk overlap" in payload["results"][0]["text"]


def test_retrieve_endpoint_returns_service_error_without_index(tmp_path):
    app = create_app(AppConfig(documents_dir=tmp_path / "missing"))
    client = TestClient(app)

    response = client.post("/retrieve", json={"query": "anything", "top_k": 1})

    assert response.status_code == 503
    assert "document directory" in response.json()["detail"]
