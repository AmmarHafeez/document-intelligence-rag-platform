# document-intelligence-rag-platform

A local-first Python baseline for document ingestion, chunking, retrieval, extractive grounded answering, and evaluation.

The project is intentionally small and deterministic. It uses local files, local indexes, and local metrics so retrieval behavior can be inspected before adding heavier retrieval methods or generative model calls.

## Why This Matters

Retrieval systems are hard to improve when ingestion, chunking, ranking, citations, and evaluation are mixed together. This repository keeps each step explicit and testable: documents become chunks, chunks become ranked results, ranked results become extractive answers, and both retrieval and grounding can be evaluated with local JSON outputs.

## Key Capabilities

- Ingest `.txt`, `.md`, and text-based `.pdf` files.
- Generate a tiny local demo corpus and relevance labels under ignored `data/raw/` paths.
- Split documents into overlapping chunks with stable IDs and character offsets.
- Retrieve with a deterministic keyword baseline.
- Build, save, load, and query a local TF-IDF retrieval index.
- Evaluate retrieval with recall@k, precision@k, and mean reciprocal rank.
- Build grounded extractive answers from retrieved chunks with citations.
- Evaluate answer grounding with deterministic sentence-support and citation-coverage checks.
- Serve `/health`, `/retrieve`, and `/answer` with FastAPI.
- Run with Docker Compose and GitHub Actions CI.

## Results Summary

These are tiny local smoke-test results, not broad benchmarks.

| Area | Local Result |
| --- | --- |
| Demo corpus | 3 generated documents, 3 chunks, 3 evaluation queries |
| TF-IDF retrieval | recall@3 `1.0`, precision@3 `0.3333`, MRR `1.0` |
| PDF demo | `pdf_ingestion_demo.pdf` was included and retrieved correctly for the PDF query |
| Grounding smoke test | sentence support rate `1.0`, citation coverage `1.0` |

Generated documents, indexes, answer JSON, and metrics stay under ignored folders such as `data/raw/`, `indexes/`, `reports/artifacts/`, and `reports/metrics/`.

## Common Commands

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
python -m pytest
```

Generate the local demo corpus:

```powershell
python -m document_intelligence_rag.ingestion.create_demo_corpus `
  --documents-dir data/raw/documents `
  --queries-path data/raw/evaluation/queries.json `
  --include-pdf
```

Build and query a TF-IDF index:

```powershell
python -m document_intelligence_rag.retrieval.build_index `
  --documents-dir data/raw/documents `
  --index-path indexes/tfidf_index.joblib `
  --backend tfidf

python -m document_intelligence_rag.retrieval.query_index `
  --index-path indexes/tfidf_index.joblib `
  --query "What is retrieval augmented generation?" `
  --top-k 3
```

Run retrieval evaluation:

```powershell
python -m document_intelligence_rag.evaluation.evaluate_retrieval `
  --documents-dir data/raw/documents `
  --queries data/raw/evaluation/queries.json `
  --output reports/metrics/demo_retrieval_eval_tfidf.json `
  --backend tfidf `
  --top-k 3
```

Generate and evaluate a grounded answer:

```powershell
python -m document_intelligence_rag.answering.answer_query `
  --index-path indexes/tfidf_index.joblib `
  --query "What is retrieval augmented generation?" `
  --top-k 3 `
  --output reports/artifacts/answer_result.json

python -m document_intelligence_rag.evaluation.evaluate_grounding `
  --answers reports/artifacts/answer_result.json `
  --output reports/metrics/grounding_eval.json
```

Start the API and call it:

```powershell
uvicorn document_intelligence_rag.api:app --host 127.0.0.1 --port 8000

Invoke-RestMethod http://127.0.0.1:8000/health

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/retrieve `
  -ContentType "application/json" `
  -Body '{"query":"What is retrieval augmented generation?","top_k":3}'

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/answer `
  -ContentType "application/json" `
  -Body '{"query":"What is retrieval augmented generation?","top_k":3}'
```

Container and CI entry points:

```powershell
docker compose up --build
```

## Artifact Policy

Local documents, generated demo data, indexes, vector stores, embeddings, answer outputs, metrics, and report artifacts are ignored by Git. Keep raw or private documents under `data/raw/` and generated outputs under ignored folders such as `indexes/`, `reports/artifacts/`, and `reports/metrics/`.

## Documentation

- [Architecture](docs/architecture.md)
- [Reproducibility](docs/reproducibility.md)
- [Results](docs/results.md)
