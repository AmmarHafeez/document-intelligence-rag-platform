# Results

The initial version includes unit-tested local retrieval components for text, Markdown, and PDF ingestion, chunking, keyword retrieval, TF-IDF vector retrieval, saved index loading, extractive grounded answer generation, deterministic grounding evaluation, retrieval evaluation, evaluation metrics, and FastAPI serving.

The tiny local demo corpus can be regenerated under ignored `data/raw/` folders with the demo corpus generator. No new numeric results are claimed until the local commands are run.

## Local Demo Corpus Smoke-Test Result

These results come from a tiny generated local demo corpus under `data/raw/`. The corpus contains 3 documents (`rag_intro.md`, `vector_search.txt`, and `pdf_ingestion_demo.pdf`), 3 chunks, and 3 evaluation queries. Generated documents, the retrieval index, and metrics files remain ignored by Git.

The PDF ingestion demo was included as `pdf_ingestion_demo.pdf` and was retrieved correctly for the PDF query. This is a smoke test for the local workflow, not a broad benchmark.

| Backend | Documents | Chunks | Queries | evaluated_at_k | recall_at_k | precision_at_k | mean_reciprocal_rank | Metrics file |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| tfidf | 3 | 3 | 3 | 3 | 1.0 | 0.3333 | 1.0 | `reports/metrics/demo_retrieval_eval_tfidf.json` |

Broader evaluation requires a larger document set and more diverse relevance labels.

## Local Grounding Smoke-Test Result

This grounding result comes from one local answer JSON at `reports/artifacts/answer_result.json`, with metrics written to `reports/metrics/grounding_eval.json`. Both generated files remain local and ignored by Git. This is a tiny local smoke test, not human evaluation or a broad benchmark.

The evaluator used cited source previews because full cited source text was not available in this generated answer JSON. Both answer sentences were matched to cited source previews, and no unsupported sentence was detected.

| answer_count | evaluated_answer_count | insufficient_context_count | sentence_support_rate | citation_coverage | unsupported_sentence_count | full_text_available_count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 0 | 1.0 | 1.0 | 0 | 0 |

Generated indexes, derived reports, metrics output, and raw private documents should remain ignored by Git.
