# Architecture

```mermaid
flowchart TB
    subgraph LocalInputs["Local Inputs"]
        A["Demo corpus generator"]
        B["Local documents<br/>.txt, .md, text-based .pdf"]
        C["Local relevance labels"]
        ARaw["Ignored: data/raw/"]
        A --> B
        A --> C
        B -. stored under .-> ARaw
        C -. stored under .-> ARaw
    end

    subgraph Processing["Ingestion And Chunking"]
        D["Ingestion"]
        E["Document objects"]
        F["Chunking"]
        G["Text chunks<br/>IDs + offsets"]
        B --> D
        D --> E
        E --> F
        F --> G
    end

    subgraph Retrieval["Retrieval"]
        H["Keyword retriever"]
        I["TF-IDF index build"]
        J["Saved TF-IDF index"]
        K["TF-IDF index load"]
        U["Query CLI"]
        Idx["Ignored: indexes/"]
        G --> H
        G --> I
        I --> J
        J --> K
        J -. stored under .-> Idx
        K --> U
    end

    subgraph Evaluation["Retrieval Evaluation"]
        L["Ranked chunks"]
        M["Retrieval evaluation"]
        N["Metrics report"]
        Metrics["Ignored: reports/metrics/"]
        H --> L
        K --> L
        C --> M
        L --> M
        M --> N
        N -. stored under .-> Metrics
    end

    subgraph Answering["Grounded Answering"]
        O["Grounded extractive answer builder"]
        P["Answer JSON with citations"]
        Q["Grounding evaluation"]
        R["Grounding metrics report"]
        Artifacts["Ignored: reports/artifacts/"]
        L --> O
        O --> P
        P --> Q
        Q --> R
        P -. stored under .-> Artifacts
        R -. stored under .-> Metrics
    end

    subgraph API["FastAPI"]
        Health["GET /health"]
        Retrieve["POST /retrieve"]
        Answer["POST /answer"]
        K --> Retrieve
        O --> Answer
    end
```

## Component Responsibilities

### Demo Corpus Generation

Creates tiny local demo documents and matching retrieval labels under ignored `data/raw/` paths. Existing files are preserved unless overwrite is requested.

### Ingestion

Reads local `.txt`, `.md`, and text-based `.pdf` files. It derives deterministic document IDs from file names, records source paths and metadata such as PDF page count, and raises clear errors for unsupported or unreadable files. Scanned-image PDFs and OCR are not supported.

### Chunking

Splits document text into overlapping chunks. Each chunk includes a stable chunk ID, document ID, source path, text, and character offsets.

### Retrieval

Provides a deterministic keyword baseline and a TF-IDF backend. The TF-IDF backend can build, save, load, and query a local index.

### Retrieval Evaluation

Reads local relevance labels, runs a selected retriever, and writes recall@k, precision@k, mean reciprocal rank, and per-query details to `reports/metrics/`.

### Grounded Answering

Builds extractive answers by selecting sentences from retrieved chunks. Responses include cited chunk IDs, cited document IDs, source previews, and a simple coverage score.

### Grounding Evaluation

Checks answer sentences against cited source text or previews. It reports sentence support, citation coverage, and unsupported sentences using deterministic local checks.

### API

`GET /health` reports service and index status. `POST /retrieve` returns ranked chunks. `POST /answer` returns an extractive answer with citations. The API loads the configured local TF-IDF index when available.
