""" Higher level retreival service for MCP tools, ADK tools, etc. """

from __future__ import annotations

from typing import Any
from threading import Lock

from protein_retrieval.config import (
    DEFAULT_KEYWORD_WEIGHT,
    DEFAULT_RRF_K,
    DEFAULT_VECTOR_WEIGHT,
    MAX_TOP_K,
    MIN_TOP_K,
    get_database_url,
    get_embedding_model_name,
    load_config
)

from protein_retrieval.db import connect
from protein_retrieval.search import hybrid_search, keyword_search, vector_search


_model: Any | None = None
_model_name: str | None = None
_model_lock = Lock()

def _normalize_query(query: str) -> str:
    """Trim and validate user search text."""
    normalized = query.strip()

    if not normalized:
        raise ValueError("query must not be empty")

    return normalized


def _clamp_top_k(top_k: int) -> int:
    """Keep retrieval result counts inside the supported range."""
    return max(MIN_TOP_K, min(top_k, MAX_TOP_K))


def _normalize_candidate_k(candidate_k: int, top_k: int) -> int:
    """Hybrid search needs at least top_k candidates to return top_k results."""
    candidate_k = _clamp_top_k(candidate_k)
    return max(candidate_k, top_k)


def _validate_hybrid_params(rrf_k: int, vector_weight: float, keyword_weight: float) -> None:
    """Reject hybrid settings that make rank fusion invalid or misleading."""
    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than 0")
    if vector_weight < 0:
        raise ValueError("vector_weight must be greater than or equal to 0")
    if keyword_weight < 0:
        raise ValueError("keyword_weight must be greater than or equal to 0")
    if vector_weight == 0 and keyword_weight == 0:
        raise ValueError("at least one retrieval weight must be greater than 0")


def get_embedding_model(model_name: str | None = None) -> Any:
    """ Lazy_load and reuse the embedding model for long-running process. """

    from protein_retrieval.embeddings import load_model

    global _model, _model_name
    load_config()

    resolved_model_name = model_name or get_embedding_model_name()

    with _model_lock:
        if _model is None or _model_name != resolved_model_name:
            _model = load_model(resolved_model_name)
            _model_name = resolved_model_name

        model = _model

    return model


def warm_embedding_model(
    model_name: str | None = None,
    warmup_text: str = "protein retrieval warmup",
) -> dict[str, Any]:
    """Load the embedding model and run one small query embedding."""

    from protein_retrieval.embeddings import embed_query

    load_config()

    resolved_model_name = model_name or get_embedding_model_name()
    model = get_embedding_model(resolved_model_name)
    embedding = embed_query(model, warmup_text)

    return {
        "embedding_model": resolved_model_name,
        "warmup_text": warmup_text,
        "embedding_dimensions": len(embedding),
    }


def _vector_rows_to_dicts(rows: list[tuple]) -> list[dict[str, Any]]:
    """ Format tuple rows from vector_search into JSON-friendly dicts. """

    results = []

    for rank, row in enumerate(rows, start=1):
        accession, protein_name, gene_names, organism, reviewed, source_url, similarity = row

        results.append(
            {
                "rank": rank,
                "accession": accession,
                "protein_name": protein_name,
                "gene_names": gene_names or [],
                "organism": organism,
                "reviewed": reviewed,
                "source_url": source_url,
                "vector_similarity": float(similarity),
            }
        )

    return results


def _keyword_rows_to_dicts(rows: list[tuple]) -> list[dict[str, Any]]:
    """Format tuple rows from keyword_search into JSON-friendly dicts."""
    results = []

    for rank, row in enumerate(rows, start=1):
        accession, protein_name, gene_names, organism, reviewed, source_url, lexical_score = row

        results.append(
            {
                "rank": rank,
                "accession": accession,
                "protein_name": protein_name,
                "gene_names": gene_names or [],
                "organism": organism,
                "reviewed": reviewed,
                "source_url": source_url,
                "lexical_score": float(lexical_score),
            }
        )

    return results

def semantic_search_proteins(query: str, top_k: int = 5, model_name: str | None = None) ->dict[str, Any]:
    """ Run semantic vector search over embedded Uniprote metadata"""

    from protein_retrieval.embeddings import embed_query

    load_config()

    query = _normalize_query(query)
    top_k = _clamp_top_k(top_k)
    resolved_model_name = model_name or get_embedding_model_name()

    model = get_embedding_model(resolved_model_name)
    query_embedding = embed_query(model, query)

    with connect(get_database_url()) as conn:
        rows = vector_search(
            conn = conn,
            query_embedding = query_embedding,
            top_k=top_k,
            embedding_model=resolved_model_name,
        )

    return {
        "query": query,
        "method": "semantic",
        "top_k": top_k,
        "embedding_model": resolved_model_name,
        "results": _vector_rows_to_dicts(rows)
    }


def keyword_search_proteins(query: str, top_k: int = 5) -> dict[str, Any]:
    """ Run PostgreSQL full-text lexical search over protein metadata"""

    load_config()

    query = _normalize_query(query)
    top_k = _clamp_top_k(top_k)

    with connect(get_database_url()) as conn:
        rows = keyword_search(
            conn = conn, 
            query = query,
            top_k=top_k
        )

    return{
        "query": query,
        "method": "keyword",
        "top_k": top_k,
        "results": _keyword_rows_to_dicts(rows)

    }


def hybrid_search_proteins(query: str, 
                           top_k: int = 5, 
                           candidate_k: int = 20, 
                           rrf_k: int = DEFAULT_RRF_K, 
                           vector_weight: float = DEFAULT_VECTOR_WEIGHT,
                           keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
                           model_name: str | None = None) -> dict[str, Any]:

    """ Run weighted Hybrid retreival using semantic + lexical search"""

    load_config()
    from protein_retrieval.embeddings import embed_query

    query = _normalize_query(query)
    top_k = _clamp_top_k(top_k)
    candidate_k = _normalize_candidate_k(candidate_k, top_k)
    _validate_hybrid_params(rrf_k, vector_weight, keyword_weight)
    resolved_model_name = model_name or get_embedding_model_name()

    model = get_embedding_model(resolved_model_name)
    query_embedding = embed_query(model, query)

    with connect(get_database_url()) as conn:
        results = hybrid_search(
            conn = conn,
            query= query,
            query_embedding=query_embedding,
            top_k= top_k,
            embedding_model=resolved_model_name,
            candidate_k=candidate_k,
            rrf_k=rrf_k,
            vector_weight=vector_weight,
            keyword_weight=keyword_weight
        )

    return {
        "query": query,
        "method": "hybrid",
        "top_k": top_k,
        "candidate_k": candidate_k,
        "rrf_k": rrf_k,
        "vector_weight": vector_weight,
        "keyword_weight": keyword_weight,
        "embedding_model": resolved_model_name,
        "results": [result.to_dict() for result in results],
    }
