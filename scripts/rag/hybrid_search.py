"""Hybrid protein retrieval using vector search + keyword search + weighted RRF."""

from __future__ import annotations

import argparse
import os

from protein_retrieval.config import (
    DEFAULT_KEYWORD_WEIGHT,
    DEFAULT_RRF_K,
    DEFAULT_VECTOR_WEIGHT,
    get_database_url,
    get_embedding_model_name,
    load_config,
)
from protein_retrieval.db import connect
from protein_retrieval.embeddings import embed_query, load_model
from protein_retrieval.search import HybridResult, hybrid_search


def print_results(results: list[HybridResult]) -> None:
    if not results:
        print("No hybrid results found.")
        return

    for rank, result in enumerate(results, start=1):
        channels = []
        if result.vector_rank is not None:
            channels.append("vector")
        if result.keyword_rank is not None:
            channels.append("keyword")

        print(f"\n#{rank} {result.accession} | hybrid_score={result.hybrid_score:.5f}")
        print(f"Protein: {result.protein_name}")
        print(f"Genes: {', '.join(result.gene_names)}")
        print(f"Organism: {result.organism}")
        print(f"Reviewed: {result.reviewed}")
        print(f"Channels: {', '.join(channels)}")
        print(f"Vector rank: {result.vector_rank}, similarity: {result.vector_similarity}")
        print(f"Keyword rank: {result.keyword_rank}, lexical score: {result.lexical_score}")
        print(f"URL: {result.source_url}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)
    parser.add_argument("--vector-weight", type=float, default=DEFAULT_VECTOR_WEIGHT)
    parser.add_argument("--keyword-weight", type=float, default=DEFAULT_KEYWORD_WEIGHT)
    parser.add_argument("--model", default=os.getenv("EMBEDDING_MODEL"))
    args = parser.parse_args()

    load_config()
    model_name = args.model or get_embedding_model_name()
    model = load_model(model_name)
    query_embedding = embed_query(model, args.query)

    with connect(get_database_url()) as conn:
        results = hybrid_search(
            conn=conn,
            query=args.query,
            query_embedding=query_embedding,
            top_k=args.top_k,
            embedding_model=model_name,
            candidate_k=args.candidate_k,
            rrf_k=args.rrf_k,
            vector_weight=args.vector_weight,
            keyword_weight=args.keyword_weight,
        )

    print_results(results)


if __name__ == "__main__":
    main()
