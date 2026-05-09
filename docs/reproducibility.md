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
