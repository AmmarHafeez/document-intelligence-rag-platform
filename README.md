# document-intelligence-rag-platform

A local-first baseline for document ingestion, text chunking, keyword and TF-IDF vector retrieval, FastAPI serving, and retrieval evaluation.

The first milestone focuses on deterministic retrieval components that can run on small local document folders. It does not call LLMs, paid services, external embedding APIs, or model downloads.

## Why This Matters

Document retrieval systems are easier to improve when the core pipeline is measurable and reproducible. This project starts with a small, testable baseline so later retrieval methods can be compared against clear local behavior.

## Key Capabilities

- Ingest `.txt` and `.md` files from a local directory.
- Split documents into overlapping chunks with stable character offsets.
- Retrieve chunks with a deterministic token-overlap baseline.
- Build, save, load, and query a local TF-IDF retrieval index.
- Evaluate ranked retrieval with recall@k, precision@k, and mean reciprocal rank.
- Run offline retrieval evaluation against local JSON relevance labels.
- Serve retrieval through FastAPI using a saved local index.
- Keep private documents, indexes, embeddings, and reports outside version control.

## Quickstart

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
python -m pytest
```

Add local demo documents under `data/raw/documents/`:

```powershell
New-Item -ItemType Directory -Force data/raw/documents
Set-Content data/raw/documents/example.txt "A short local document about retrieval quality."
```

Run a retrieval query:

```powershell
python -m document_intelligence_rag.retrieval.build_index `
  --documents-dir data/raw/documents `
  --index-path indexes/tfidf_index.joblib `
  --chunk-size 800 `
  --chunk-overlap 120 `
  --backend tfidf

python -m document_intelligence_rag.retrieval.query_index `
  --index-path indexes/tfidf_index.joblib `
  --query "retrieval quality" `
  --top-k 3
```

Create tiny local evaluation labels:

```powershell
New-Item -ItemType Directory -Force data/raw/evaluation
@'
{
  "queries": [
    {
      "query_id": "q1",
      "query": "retrieval quality",
      "relevant_document_ids": ["example"]
    }
  ]
}
'@ | Set-Content data/raw/evaluation/queries.json
```

Document IDs are derived from local file names, so `data/raw/documents/example.txt` has document ID `example`.

Run retrieval evaluation:

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

## Local Smoke-Test Results

A tiny local retrieval evaluation has been run with 2 documents, 2 chunks, and 3 queries at `top_k=3`. Both keyword and TF-IDF retrieval returned `recall_at_k: 1.0`, `precision_at_k: 0.3333`, and `mean_reciprocal_rank: 1.0` for this simple demo. These are smoke-test results only, not a real benchmark; broader evaluation needs more documents and more diverse relevance labels.

The local metrics files are `reports/metrics/retrieval_eval_keyword.json` and `reports/metrics/retrieval_eval_tfidf.json`, and they remain ignored by Git.

Start the API:

```powershell
uvicorn document_intelligence_rag.api:app --reload
```

## Common Commands

```powershell
pip install -e .
python -m pytest
python -m document_intelligence_rag.retrieval.build_index --documents-dir data/raw/documents --index-path indexes/tfidf_index.joblib --backend tfidf
python -m document_intelligence_rag.retrieval.query_index --index-path indexes/tfidf_index.joblib --query "example query" --top-k 3
python -m document_intelligence_rag.evaluation.evaluate_retrieval --documents-dir data/raw/documents --queries data/raw/evaluation/queries.json --output reports/metrics/retrieval_eval_keyword.json --backend keyword --top-k 3
python -m document_intelligence_rag.evaluation.evaluate_retrieval --documents-dir data/raw/documents --queries data/raw/evaluation/queries.json --output reports/metrics/retrieval_eval_tfidf.json --backend tfidf --top-k 3
uvicorn document_intelligence_rag.api:app --host 127.0.0.1 --port 8000
docker compose up --build
```

## Artifact Policy

The repository is configured to ignore local private documents and derived files, including `data/raw/`, `data/processed/`, `data/interim/`, `indexes/`, `vectorstores/`, `embeddings/`, `reports/metrics/`, and `reports/artifacts/`. Keep evaluation outputs and raw private documents outside Git.

## Documentation

- [Architecture](docs/architecture.md)
- [Reproducibility](docs/reproducibility.md)
- [Results](docs/results.md)
