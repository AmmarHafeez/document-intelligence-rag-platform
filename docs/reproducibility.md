# Reproducibility

These steps assume PowerShell from the repository root.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## Run Tests

```powershell
python -m pytest
```

## Add Local Demo Documents

```powershell
New-Item -ItemType Directory -Force data/raw/documents
Set-Content data/raw/documents/guide.txt "Retrieval quality depends on matching user questions to relevant chunks."
Set-Content data/raw/documents/notes.md "# Notes`nChunk overlap helps preserve context across chunk boundaries."
```

`data/raw/` is ignored by Git so private local documents remain outside version control.

## Add Local Evaluation Queries

Evaluation labels live outside Git under `data/raw/evaluation/`.

```powershell
New-Item -ItemType Directory -Force data/raw/evaluation
@'
{
  "queries": [
    {
      "query_id": "q1",
      "query": "chunk overlap",
      "relevant_document_ids": ["notes"]
    }
  ]
}
'@ | Set-Content data/raw/evaluation/queries.json
```

Document IDs are derived from local file names, so `data/raw/documents/notes.md` has document ID `notes`. Labels can use `relevant_document_ids`, `relevant_chunk_ids`, or both. When chunk labels are present, evaluation uses chunk IDs for that query.

## Build TF-IDF Index

The command reads local documents, chunks them, builds a TF-IDF retrieval index, and saves it under `indexes/`.

```powershell
python -m document_intelligence_rag.retrieval.build_index `
  --documents-dir data/raw/documents `
  --index-path indexes/tfidf_index.joblib `
  --chunk-size 800 `
  --chunk-overlap 120 `
  --backend tfidf
```

## Query Saved Index

```powershell
python -m document_intelligence_rag.retrieval.query_index `
  --index-path indexes/tfidf_index.joblib `
  --query "chunk overlap" `
  --top-k 3
```

Optionally write query output to an ignored artifacts folder:

```powershell
python -m document_intelligence_rag.retrieval.query_index `
  --index-path indexes/tfidf_index.joblib `
  --query "chunk overlap" `
  --top-k 3 `
  --output reports/artifacts/retrieval_results.json
```

## Ask A Grounded Question

The answer workflow loads a saved retrieval index, retrieves chunks, and builds an extractive answer from retrieved sentences.

```powershell
python -m document_intelligence_rag.answering.answer_query `
  --index-path indexes/tfidf_index.joblib `
  --query "What is retrieval augmented generation?" `
  --top-k 3 `
  --output reports/artifacts/answer_result.json
```

Answer JSON is written to `reports/artifacts/`, which is ignored by Git. The answer text is grounded in retrieved chunks and includes cited chunk and document IDs.

## Run Grounding Evaluation

The grounding evaluator checks whether answer sentences are supported by cited source text. If full cited text is unavailable, it falls back to source previews and records that in the report.

```powershell
python -m document_intelligence_rag.evaluation.evaluate_grounding `
  --answers reports/artifacts/answer_result.json `
  --output reports/metrics/grounding_eval.json
```

Grounding metrics are written to `reports/metrics/`, which is ignored by Git. These checks are deterministic support checks, not human evaluation.

## Run Keyword Evaluation

```powershell
python -m document_intelligence_rag.evaluation.evaluate_retrieval `
  --documents-dir data/raw/documents `
  --queries data/raw/evaluation/queries.json `
  --output reports/metrics/retrieval_eval_keyword.json `
  --backend keyword `
  --chunk-size 800 `
  --chunk-overlap 120 `
  --top-k 3
```

## Run TF-IDF Evaluation

```powershell
python -m document_intelligence_rag.evaluation.evaluate_retrieval `
  --documents-dir data/raw/documents `
  --queries data/raw/evaluation/queries.json `
  --output reports/metrics/retrieval_eval_tfidf.json `
  --backend tfidf `
  --chunk-size 800 `
  --chunk-overlap 120 `
  --top-k 3
```

Evaluation reports are written to `reports/metrics/`, which is ignored by Git.

## Current Local Evaluation Results

The current local smoke-test run used `data/raw/evaluation/queries.json` with 3 queries, `evaluated_at_k: 3`, 2 documents, and 2 chunks. Both keyword and TF-IDF retrieval ranked the relevant chunk first for the simple demo queries.

| Backend | recall_at_k | precision_at_k | mean_reciprocal_rank | Metrics file |
| --- | ---: | ---: | ---: | --- |
| keyword | 1.0 | 0.3333 | 1.0 | `reports/metrics/retrieval_eval_keyword.json` |
| tfidf | 1.0 | 0.3333 | 1.0 | `reports/metrics/retrieval_eval_tfidf.json` |

These are tiny local smoke-test results, not a real benchmark. Generated documents, evaluation labels, and metrics JSON remain local and ignored by Git.

## Current Local Grounding Result

The current local grounding smoke test used `reports/artifacts/answer_result.json` and wrote metrics to `reports/metrics/grounding_eval.json`. Both generated files remain local and ignored by Git.

| answer_count | evaluated_answer_count | insufficient_context_count | sentence_support_rate | citation_coverage | unsupported_sentence_count | full_text_available_count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 0 | 1.0 | 1.0 | 0 | 0 |

The evaluator used cited source previews because full cited source text was not available in this generated answer JSON. Both answer sentences were matched to cited source previews, and no unsupported sentence was detected. This is a tiny local smoke-test result, not human evaluation or a broad benchmark.

## Start API

The default API configuration loads `indexes/tfidf_index.joblib`.

```powershell
uvicorn document_intelligence_rag.api:app --host 127.0.0.1 --port 8000
```

## Call Health

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

## Call Retrieval

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/retrieve `
  -ContentType "application/json" `
  -Body '{"query":"chunk overlap","top_k":3}'
```

## Call Answer

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/answer `
  -ContentType "application/json" `
  -Body '{"query":"What is retrieval augmented generation?","top_k":3}'
```
