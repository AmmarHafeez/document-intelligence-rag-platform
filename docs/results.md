# Results

The repository includes unit-tested local components for text, Markdown, and text-based PDF ingestion; chunking; keyword retrieval; TF-IDF retrieval; retrieval evaluation; extractive grounded answering; grounding evaluation; and FastAPI serving.

All results below are tiny local smoke-test results. They are useful for checking that the workflow is wired together, but they are not broad benchmarks.

Generated documents, indexes, answer JSON, metrics, and report artifacts remain under ignored folders such as `data/raw/`, `indexes/`, `reports/artifacts/`, and `reports/metrics/`.

## Generated Demo Corpus

The demo corpus can be regenerated locally under `data/raw/`.

| Item | Value |
| --- | --- |
| Documents | 3: `rag_intro.md`, `vector_search.txt`, `pdf_ingestion_demo.pdf` |
| Chunks | 3 |
| Evaluation queries | 3 |
| PDF coverage | The PDF ingestion demo was included and retrieved correctly for the PDF query |

## Retrieval Evaluation

| Backend | Documents | Chunks | Queries | evaluated_at_k | recall@3 | precision@3 | MRR | Metrics file |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| TF-IDF | 3 | 3 | 3 | 3 | 1.0 | 0.3333 | 1.0 | `reports/metrics/demo_retrieval_eval_tfidf.json` |

Broader retrieval evaluation requires a larger document set and more diverse relevance labels.

## Grounded Answer Example

The answer workflow generated a local answer JSON at `reports/artifacts/answer_result.json`. Answers are extractive and cite retrieved chunks/documents rather than generating unsupported text.

## Grounding Evaluation

The grounding evaluator checked the local answer JSON and wrote metrics to `reports/metrics/grounding_eval.json`. It used cited source previews because full cited source text was not available in that generated answer JSON.

| answer_count | evaluated_answer_count | insufficient_context_count | sentence_support_rate | citation_coverage | unsupported_sentence_count | full_text_available_count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 0 | 1.0 | 1.0 | 0 | 0 |

Both answer sentences were matched to cited source previews, and no unsupported sentence was detected. This is a deterministic grounding smoke test, not human evaluation.
