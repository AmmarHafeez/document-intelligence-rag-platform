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

## Run Retrieval

The command builds an in-memory index from the configured documents and prints ranked chunks as JSON.

```powershell
python -m document_intelligence_rag retrieve "chunk overlap"
```

## Start API

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
