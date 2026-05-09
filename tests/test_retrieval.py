from document_intelligence_rag.models import TextChunk
from document_intelligence_rag.retrieval import KeywordRetriever, tokenize


def test_tokenize_is_deterministic():
    assert tokenize("Alpha, beta! ALPHA") == ["alpha", "beta", "alpha"]


def test_retrieval_ranking_prefers_more_token_overlap(tmp_path):
    chunks = [
        TextChunk(
            chunk_id="c1",
            document_id="d1",
            source_path=tmp_path / "one.txt",
            text="alpha beta",
            start_char=0,
            end_char=10,
        ),
        TextChunk(
            chunk_id="c2",
            document_id="d2",
            source_path=tmp_path / "two.txt",
            text="gamma only",
            start_char=0,
            end_char=10,
        ),
        TextChunk(
            chunk_id="c3",
            document_id="d3",
            source_path=tmp_path / "three.txt",
            text="alpha beta beta",
            start_char=0,
            end_char=15,
        ),
    ]
    retriever = KeywordRetriever(chunks)

    results = retriever.retrieve("alpha beta", top_k=2)

    assert [result.chunk.chunk_id for result in results] == ["c3", "c1"]
    assert results[0].score > results[1].score
    assert results[0].matched_terms == ["alpha", "beta"]
