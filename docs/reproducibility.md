# Reproducibility

These steps assume PowerShell from the repository root.

## 1. Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## 2. Generate Demo Corpus

```powershell
python -m document_intelligence_rag.ingestion.create_demo_corpus `
  --documents-dir data/raw/documents `
  --queries-path data/raw/evaluation/queries.json `
  --include-pdf
```

The generator creates `rag_intro.md`, `vector_search.txt`, optional `pdf_ingestion_demo.pdf`, and `data/raw/evaluation/queries.json`. It does not overwrite existing files unless `--overwrite` is passed. `data/raw/` is ignored by Git.

## 3. Build TF-IDF Index

```powershell
python -m document_intelligence_rag.retrieval.build_index `
  --documents-dir data/raw/documents `
  --index-path indexes/tfidf_index.joblib `
  --chunk-size 800 `
  --chunk-overlap 120 `
  --backend tfidf
```

The saved index is written under `indexes/`, which is ignored by Git.

## 4. Query Index

```powershell
python -m document_intelligence_rag.retrieval.query_index `
  --index-path indexes/tfidf_index.joblib `
  --query "What is retrieval augmented generation?" `
  --top-k 3
```

Optional JSON output can be written to `reports/artifacts/`, which is ignored by Git.

## 5. Run Retrieval Evaluation

```powershell
python -m document_intelligence_rag.evaluation.evaluate_retrieval `
  --documents-dir data/raw/documents `
  --queries data/raw/evaluation/queries.json `
  --output reports/metrics/demo_retrieval_eval_tfidf.json `
  --backend tfidf `
  --chunk-size 800 `
  --chunk-overlap 120 `
  --top-k 3
```

Evaluation reports are written to `reports/metrics/`, which is ignored by Git.

## 6. Generate Grounded Answer

```powershell
python -m document_intelligence_rag.answering.answer_query `
  --index-path indexes/tfidf_index.joblib `
  --query "What is retrieval augmented generation?" `
  --top-k 3 `
  --output reports/artifacts/answer_result.json
```

Answers are extractive: the answer text is assembled from retrieved context and returned with cited chunk and document IDs.

## 7. Run Grounding Evaluation

```powershell
python -m document_intelligence_rag.evaluation.evaluate_grounding `
  --answers reports/artifacts/answer_result.json `
  --output reports/metrics/grounding_eval.json
```

The grounding evaluator checks whether answer sentences are supported by cited source text or previews. These checks are deterministic support checks, not human evaluation.

## 8. Start API

```powershell
uvicorn document_intelligence_rag.api:app --host 127.0.0.1 --port 8000
```

The default configuration loads `indexes/tfidf_index.joblib`.

## 9. Call API

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/retrieve `
  -ContentType "application/json" `
  -Body '{"query":"What is retrieval augmented generation?","top_k":3}'
```

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/answer `
  -ContentType "application/json" `
  -Body '{"query":"What is retrieval augmented generation?","top_k":3}'
```

## Local Smoke-Test Results

The generated demo corpus smoke test used 3 documents, 3 chunks, and 3 queries. The TF-IDF run produced recall@3 `1.0`, precision@3 `0.3333`, and MRR `1.0`. The grounding smoke test produced sentence support rate `1.0` and citation coverage `1.0`.

These are tiny local smoke-test results, not broad benchmarks. Generated documents, indexes, answer JSON, and metrics remain ignored by Git.
