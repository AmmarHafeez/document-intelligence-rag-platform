# Results

The initial version includes unit-tested local retrieval components for ingestion, chunking, keyword retrieval, TF-IDF vector retrieval, saved index loading, retrieval evaluation, evaluation metrics, and FastAPI serving.

## Local Smoke-Test Results

These results come from a tiny local demo corpus with 2 documents, 2 chunks, and 3 evaluation queries from `data/raw/evaluation/queries.json`. They are a smoke test for the retrieval workflow, not a real benchmark.

The run used `evaluated_at_k: 3`. Generated documents, evaluation labels, and metrics JSON remain local and ignored by Git.

| Backend | Documents | Chunks | recall_at_k | precision_at_k | mean_reciprocal_rank | Metrics file |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| keyword | 2 | 2 | 1.0 | 0.3333 | 1.0 | `reports/metrics/retrieval_eval_keyword.json` |
| tfidf | 2 | 2 | 1.0 | 0.3333 | 1.0 | `reports/metrics/retrieval_eval_tfidf.json` |

Both retrievers ranked the relevant chunk first for the simple demo queries. Broader evaluation requires a larger document set and more diverse relevance labels.

Generated indexes, derived reports, metrics output, and raw private documents should remain ignored by Git.
