from fastapi.testclient import TestClient

from document_intelligence_rag.api import create_app
from document_intelligence_rag.config import AppConfig
from document_intelligence_rag.models import TextChunk
from document_intelligence_rag.retrieval import TfidfRetriever


def test_health_endpoint_works_without_index(tmp_path):
    app = create_app(AppConfig(index_path=tmp_path / "missing.joblib"))
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["index_ready"] is False
    assert payload["backend"] == "tfidf"


def test_retrieve_endpoint_returns_ranked_chunks_from_tfidf_index(tmp_path):
    index_path = tmp_path / "indexes" / "tfidf_index.joblib"
    chunks = [
        TextChunk(
            chunk_id="c1",
            document_id="d1",
            source_path=tmp_path / "alpha.txt",
            text="Alpha retrieval systems rank relevant chunks first.",
            start_char=0,
            end_char=52,
        ),
        TextChunk(
            chunk_id="c2",
            document_id="d2",
            source_path=tmp_path / "beta.md",
            text="Chunk overlap keeps context available.",
            start_char=0,
            end_char=38,
        ),
    ]
    TfidfRetriever().fit(chunks).save(index_path)
    app = create_app(
        AppConfig(
            index_path=index_path,
            retriever_backend="tfidf",
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
    app = create_app(AppConfig(index_path=tmp_path / "missing.joblib"))
    client = TestClient(app)

    response = client.post("/retrieve", json={"query": "anything", "top_k": 1})

    assert response.status_code == 503
    assert "TF-IDF index" in response.json()["detail"]
