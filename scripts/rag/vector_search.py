"""Run semantic vector search over embedded protein metadata."""

from __future__ import annotations

import argparse
import os

from protein_retrieval.config import get_database_url, get_embedding_model_name, load_config
from protein_retrieval.db import connect
from protein_retrieval.embeddings import embed_query, load_model
from protein_retrieval.search import vector_search


def print_results(rows: list[tuple]) -> None:
    if not rows:
        print("No result found")
        return

    for rank, row in enumerate(rows, start=1):
        accession, protein_name, gene_names, organism, reviewed, source_url, similarity = row
        print(f"\n#{rank} {accession} | similarity={similarity:.4f}")
        print(f"Protein: {protein_name}")
        print(f"Genes: {', '.join(gene_names or [])}")
        print(f"Organism: {organism}")
        print(f"Reviewed: {reviewed}")
        print(f"URL: {source_url}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model", default=os.getenv("EMBEDDING_MODEL"))
    args = parser.parse_args()

    load_config()
    model_name = args.model or get_embedding_model_name()
    model = load_model(model_name)
    query_embedding = embed_query(model, args.query)

    with connect(get_database_url()) as conn:
        rows = vector_search(
            conn,
            query_embedding,
            top_k=args.top_k,
            embedding_model=model_name,
        )

    print_results(rows)


if __name__ == "__main__":
    main()
