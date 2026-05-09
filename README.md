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
uvicorn document_intelligence_rag.api:app --host 127.0.0.1 --port 8000
docker compose up --build
```

## Artifact Policy

The repository is configured to ignore local private documents and derived files, including `data/raw/`, `data/processed/`, `data/interim/`, `indexes/`, `vectorstores/`, `embeddings/`, and report output folders. Keep benchmark outputs and raw private documents outside Git.

## Documentation

- [Architecture](docs/architecture.md)
- [Reproducibility](docs/reproducibility.md)
- [Results](docs/results.md)
