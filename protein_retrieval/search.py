from __future__ import annotations

from dataclasses import asdict, dataclass

import psycopg
from pgvector import Vector

from protein_retrieval.config import DEFAULT_KEYWORD_WEIGHT, DEFAULT_RRF_K, DEFAULT_VECTOR_WEIGHT

VECTOR_SEARCH_SQL = """
SELECT
    accession,
    protein_name,
    gene_names,
    organism,
    reviewed,
    source_url,
    1 - (embedding <=> %(query_embedding)s) AS similarity
FROM proteins
WHERE embedding IS NOT NULL
    AND (%(embedding_model)s::text IS NULL OR embedding_model = %(embedding_model)s)
ORDER BY embedding <=> %(query_embedding)s
LIMIT %(top_k)s;
"""

KEYWORD_SEARCH_SQL = """
SELECT
    accession,
    protein_name,
    gene_names,
    organism,
    reviewed,
    source_url,
    ts_rank_cd(search_tsv, websearch_to_tsquery('english', %(query)s)) AS lexical_score
FROM proteins
WHERE search_tsv @@ websearch_to_tsquery('english', %(query)s)
ORDER BY lexical_score DESC
LIMIT %(top_k)s;
"""


@dataclass
class HybridResult:
    accession: str
    protein_name: str | None
    gene_names: list[str]
    organism: str | None
    reviewed: bool
    source_url: str | None
    hybrid_score: float
    vector_rank: int | None = None
    vector_similarity: float | None = None
    keyword_rank: int | None = None
    lexical_score: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def vector_search(
    conn: psycopg.Connection,
    query_embedding: list[float],
    top_k: int,
    embedding_model: str | None = None,
) -> list[tuple]:
    return conn.execute(
        VECTOR_SEARCH_SQL,
        {
            "query_embedding": Vector(query_embedding),
            "top_k": top_k,
            "embedding_model": embedding_model,
        },
    ).fetchall()


def keyword_search(conn: psycopg.Connection, query: str, top_k: int) -> list[tuple]:
    return conn.execute(
        KEYWORD_SEARCH_SQL,
        {"query": query, "top_k": top_k},
    ).fetchall()


def rrf_score(rank: int, k: int = DEFAULT_RRF_K) -> float:
    return 1 / (rank + k)


def add_vector_results(
    combined: dict[str, HybridResult],
    rows: list[tuple],
    k: int,
    weight: float = DEFAULT_VECTOR_WEIGHT,
) -> None:
    for rank, row in enumerate(rows, start=1):
        accession, protein_name, gene_names, organism, reviewed, source_url, similarity = row
        combined[accession] = HybridResult(
            accession=accession,
            protein_name=protein_name,
            gene_names=gene_names or [],
            organism=organism,
            reviewed=reviewed,
            source_url=source_url,
            hybrid_score=weight * rrf_score(rank, k),
            vector_rank=rank,
            vector_similarity=similarity,
        )


def add_keyword_results(
    combined: dict[str, HybridResult],
    rows: list[tuple],
    k: int,
    weight: float = DEFAULT_KEYWORD_WEIGHT,
) -> None:
    for rank, row in enumerate(rows, start=1):
        accession, protein_name, gene_names, organism, reviewed, source_url, lexical_score = row
        score = weight * rrf_score(rank, k)
        if accession in combined:
            result = combined[accession]
            result.hybrid_score += score
            result.keyword_rank = rank
            result.lexical_score = lexical_score
        else:
            combined[accession] = HybridResult(
                accession=accession,
                protein_name=protein_name,
                gene_names=gene_names or [],
                organism=organism,
                reviewed=reviewed,
                source_url=source_url,
                hybrid_score=score,
                keyword_rank=rank,
                lexical_score=lexical_score,
            )


def hybrid_search(
    conn: psycopg.Connection,
    query: str,
    query_embedding: list[float],
    top_k: int,
    embedding_model: str | None = None,
    candidate_k: int = 20,
    rrf_k: int = DEFAULT_RRF_K,
    vector_weight: float = DEFAULT_VECTOR_WEIGHT,
    keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
) -> list[HybridResult]:
    vector_rows = vector_search(
        conn,
        query_embedding,
        top_k=candidate_k,
        embedding_model=embedding_model,
    )
    keyword_rows = keyword_search(conn, query, top_k=candidate_k)
    combined: dict[str, HybridResult] = {}
    add_vector_results(combined, vector_rows, rrf_k, vector_weight)
    add_keyword_results(combined, keyword_rows, rrf_k, keyword_weight)
    return sorted(
        combined.values(),
        key=lambda result: result.hybrid_score,
        reverse=True,
    )[:top_k]
