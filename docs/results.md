# Results

The initial version includes unit-tested local retrieval components for text, Markdown, and PDF ingestion, chunking, keyword retrieval, TF-IDF vector retrieval, saved index loading, extractive grounded answer generation, deterministic grounding evaluation, retrieval evaluation, evaluation metrics, and FastAPI serving.

## Local Smoke-Test Results

These results come from a tiny local demo corpus with 2 documents, 2 chunks, and 3 evaluation queries from `data/raw/evaluation/queries.json`. They are a smoke test for the retrieval workflow, not a real benchmark.

The run used `evaluated_at_k: 3`. Generated documents, evaluation labels, and metrics JSON remain local and ignored by Git.

| Backend | Documents | Chunks | recall_at_k | precision_at_k | mean_reciprocal_rank | Metrics file |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| keyword | 2 | 2 | 1.0 | 0.3333 | 1.0 | `reports/metrics/retrieval_eval_keyword.json` |
| tfidf | 2 | 2 | 1.0 | 0.3333 | 1.0 | `reports/metrics/retrieval_eval_tfidf.json` |

Both retrievers ranked the relevant chunk first for the simple demo queries. Broader evaluation requires a larger document set and more diverse relevance labels.

## Local Grounding Smoke-Test Result

This grounding result comes from one local answer JSON at `reports/artifacts/answer_result.json`, with metrics written to `reports/metrics/grounding_eval.json`. Both generated files remain local and ignored by Git. This is a tiny local smoke test, not human evaluation or a broad benchmark.

The evaluator used cited source previews because full cited source text was not available in this generated answer JSON. Both answer sentences were matched to cited source previews, and no unsupported sentence was detected.

| answer_count | evaluated_answer_count | insufficient_context_count | sentence_support_rate | citation_coverage | unsupported_sentence_count | full_text_available_count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 0 | 1.0 | 1.0 | 0 | 0 |

Generated indexes, derived reports, metrics output, and raw private documents should remain ignored by Git.
