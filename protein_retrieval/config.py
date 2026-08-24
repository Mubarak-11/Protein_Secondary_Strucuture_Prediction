from __future__ import annotations

import os

from dotenv import load_dotenv

DEFAULT_DATABASE_URL = "postgresql://mubarak@localhost:5432/protein_rag"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_RRF_K = 60
DEFAULT_VECTOR_WEIGHT = 1.0
DEFAULT_KEYWORD_WEIGHT = 0.1
MIN_TOP_K = 1
MAX_TOP_K = 20
LOCAL_FILES_ONLY_ENV_VAR = "PROTEIN_RETRIEVAL_LOCAL_FILES_ONLY"


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "on"}

def load_config() -> None:
    load_dotenv()


def get_database_url() -> str:
    return os.getenv("DATABASE_URL") or DEFAULT_DATABASE_URL


def get_embedding_model_name() -> str:
    return os.getenv("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL


def get_local_files_only() -> bool:
    return _env_flag(LOCAL_FILES_ONLY_ENV_VAR, default=True)
