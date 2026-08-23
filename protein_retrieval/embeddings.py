from __future__ import annotations

import logging
from typing import Any

import psycopg
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

SELECT_UNEMBEDDED_SQL = """
SELECT accession, search_text
FROM proteins
WHERE embedding IS NULL
ORDER BY accession
LIMIT %(limit)s;
"""

UPDATE_EMBEDDING_SQL = """
UPDATE proteins
SET
    embedding = %(embedding)s,
    embedding_model = %(embedding_model)s,
    embedded_at = now()
WHERE accession = %(accession)s;
"""


def load_model(model_name: str) -> SentenceTransformer:
    logger.info("Loading embedding model: %s", model_name)
    return SentenceTransformer(model_name)


def embed_query(model: Any, query: str) -> list[float]:
    embedding = model.encode(query, normalize_embeddings=True)
    return embedding.tolist()


def embed_texts(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return embeddings.tolist()


def fetch_unembedded_rows(conn: psycopg.Connection, limit: int) -> list[dict[str, Any]]:
    rows = conn.execute(SELECT_UNEMBEDDED_SQL, {"limit": limit}).fetchall()
    return [{"accession": row[0], "search_text": row[1]} for row in rows]


def update_embedding(
    conn: psycopg.Connection,
    accession: str,
    embedding: list[float],
    embedding_model: str,
) -> None:
    conn.execute(
        UPDATE_EMBEDDING_SQL,
        {
            "accession": accession,
            "embedding": embedding,
            "embedding_model": embedding_model,
        },
    )

