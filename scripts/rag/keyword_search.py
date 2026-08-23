"""Run PostgreSQL full-text keyword search over protein metadata."""

from __future__ import annotations

import argparse

from protein_retrieval.config import get_database_url, load_config
from protein_retrieval.db import connect
from protein_retrieval.search import keyword_search


def print_results(rows: list[tuple]) -> None:
    if not rows:
        print("No result found")
        return

    for rank, row in enumerate(rows, start=1):
        accession, protein_name, gene_names, organism, reviewed, source_url, lexical_score = row
        print(f"\n#{rank} {accession} | lexical_score={lexical_score:.4f}")
        print(f"Protein: {protein_name}")
        print(f"Genes: {', '.join(gene_names or [])}")
        print(f"Organism: {organism}")
        print(f"Reviewed: {reviewed}")
        print(f"URL: {source_url}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    load_config()
    with connect(get_database_url()) as conn:
        rows = keyword_search(conn, args.query, top_k=args.top_k)

    print_results(rows)


if __name__ == "__main__":
    main()
