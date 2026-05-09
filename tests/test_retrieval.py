from document_intelligence_rag.models import TextChunk
from document_intelligence_rag.retrieval import KeywordRetriever, TfidfRetriever, tokenize


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


def test_tfidf_retriever_ranks_relevant_chunk_higher(tmp_path):
    chunks = [
        TextChunk(
            chunk_id="c1",
            document_id="d1",
            source_path=tmp_path / "policy.txt",
            text="Relevant invoices require approval before payment processing.",
            start_char=0,
            end_char=61,
        ),
        TextChunk(
            chunk_id="c2",
            document_id="d2",
            source_path=tmp_path / "retrieval.txt",
            text="Retrieval augmented generation uses relevant document chunks.",
            start_char=0,
            end_char=61,
        ),
    ]
    retriever = TfidfRetriever().fit(chunks)

    results = retriever.retrieve("relevant retrieval chunks", top_k=2)

    assert results[0].chunk.chunk_id == "c2"
    assert results[0].score > results[1].score


def test_tfidf_save_load_preserves_retrieval_behavior(tmp_path):
    chunks = [
        TextChunk(
            chunk_id="c1",
            document_id="d1",
            source_path=tmp_path / "alpha.txt",
            text="Apples and pears are fruit.",
            start_char=0,
            end_char=28,
        ),
        TextChunk(
            chunk_id="c2",
            document_id="d2",
            source_path=tmp_path / "beta.txt",
            text="Contracts and invoices are business documents.",
            start_char=0,
            end_char=47,
        ),
    ]
    index_path = tmp_path / "indexes" / "tfidf_index.joblib"
    TfidfRetriever().fit(chunks).save(index_path)

    loaded = TfidfRetriever.load(index_path)
    results = loaded.retrieve("business invoice documents", top_k=1)

    assert results[0].chunk.chunk_id == "c2"
