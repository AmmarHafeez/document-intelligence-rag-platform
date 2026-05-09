from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_DOCUMENTS_DIR = Path("data/raw/documents")
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120
DEFAULT_TOP_K = 5
DEFAULT_RETRIEVER_BACKEND = "tfidf"
DEFAULT_INDEX_PATH = Path("indexes/tfidf_index.joblib")
CONFIG_ENV_VAR = "DOCUMENT_INTELLIGENCE_RAG_CONFIG"


@dataclass(frozen=True, slots=True)
class AppConfig:
    documents_dir: Path = DEFAULT_DOCUMENTS_DIR
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    top_k: int = DEFAULT_TOP_K
    retriever_backend: str = DEFAULT_RETRIEVER_BACKEND
    index_path: Path = DEFAULT_INDEX_PATH


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def _coerce_config(data: dict[str, Any]) -> AppConfig:
    return AppConfig(
        documents_dir=Path(data.get("documents_dir", DEFAULT_DOCUMENTS_DIR)),
        chunk_size=int(data.get("chunk_size", DEFAULT_CHUNK_SIZE)),
        chunk_overlap=int(data.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)),
        top_k=int(data.get("top_k", DEFAULT_TOP_K)),
        retriever_backend=str(data.get("retriever_backend", DEFAULT_RETRIEVER_BACKEND)),
        index_path=Path(data.get("index_path", DEFAULT_INDEX_PATH)),
    )


def load_config(path: str | Path | None = None) -> AppConfig:
    configured_path = path or os.getenv(CONFIG_ENV_VAR)
    config_path = Path(configured_path) if configured_path else default_config_path()

    if not config_path.exists():
        return AppConfig()

    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {config_path}")

    return _coerce_config(data)
