"""Evaluate vector, keyword, and hybrid protein retrieval."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

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
from protein_retrieval.search import hybrid_search, keyword_search, vector_search

DEFAULT_BENCHMARK_PATH = Path("benchmarks/retrieval_queries.jsonl")


def read_benchmark(path: Path) -> list[dict]:
    examples = []
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def precision_at_k(results: list[str], relevant: set[str], k: int) -> float:
    top_k = results[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for accession in top_k if accession in relevant)
    return hits / k


def recall_at_k(results: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = results[:k]
    hits = sum(1 for accession in top_k if accession in relevant)
    return hits / len(relevant)


def reciprocal_rank(results: list[str], relevant: set[str]) -> float:
    for rank, accession in enumerate(results, start=1):
        if accession in relevant:
            return 1 / rank
    return 0.0


def summarize(metric_rows: list[dict]) -> dict:
    if not metric_rows:
        return {"precision_at_5": 0.0, "recall_at_10": 0.0, "mrr": 0.0}
    return {
        "precision_at_5": sum(row["precision_at_5"] for row in metric_rows) / len(metric_rows),
        "recall_at_10": sum(row["recall_at_10"] for row in metric_rows) / len(metric_rows),
        "mrr": sum(row["mrr"] for row in metric_rows) / len(metric_rows),
    }


def accessions_from_rows(rows: list[tuple]) -> list[str]:
    return [row[0] for row in rows]


def evaluate_accessions(results: list[str], relevant: set[str]) -> dict:
    return {
        "precision_at_5": precision_at_k(results, relevant, k=5),
        "recall_at_10": recall_at_k(results, relevant, k=10),
        "mrr": reciprocal_rank(results, relevant),
    }


def print_summary(method_name: str, metric_rows: list[dict]) -> None:
    summary = summarize(metric_rows)
    print(f"\n{method_name}")
    print(f"  Precision@5: {summary['precision_at_5']:.3f}")
    print(f"  Recall@10:   {summary['recall_at_10']:.3f}")
    print(f"  MRR:         {summary['mrr']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)
    parser.add_argument("--vector-weight", type=float, default=DEFAULT_VECTOR_WEIGHT)
    parser.add_argument("--keyword-weight", type=float, default=DEFAULT_KEYWORD_WEIGHT)
    parser.add_argument("--model", default=os.getenv("EMBEDDING_MODEL"))
    args = parser.parse_args()

    load_config()
    benchmark = read_benchmark(args.benchmark)
    model_name = args.model or get_embedding_model_name()
    model = load_model(model_name)
    print(
        f"Evaluating hybrid with vector_weight={args.vector_weight}, "
        f"keyword_weight={args.keyword_weight}, rrf_k={args.rrf_k}"
    )

    metrics_by_method = {"vector": [], "keyword": [], "hybrid": []}

    with connect(get_database_url()) as conn:
        for example in benchmark:
            query = example["query"]
            relevant = set(example["relevant_accessions"])
            query_embedding = embed_query(model, query)

            vector_rows = vector_search(
                conn,
                query_embedding,
                top_k=args.candidate_k,
                embedding_model=model_name,
            )
            keyword_rows = keyword_search(conn, query, top_k=args.candidate_k)
            hybrid_rows = hybrid_search(
                conn=conn,
                query=query,
                query_embedding=query_embedding,
                top_k=args.candidate_k,
                embedding_model=model_name,
                candidate_k=args.candidate_k,
                rrf_k=args.rrf_k,
                vector_weight=args.vector_weight,
                keyword_weight=args.keyword_weight,
            )

            vector_accessions = accessions_from_rows(vector_rows)[: args.top_k]
            keyword_accessions = accessions_from_rows(keyword_rows)[: args.top_k]
            hybrid_accessions = [result.accession for result in hybrid_rows[: args.top_k]]

            metrics_by_method["vector"].append(evaluate_accessions(vector_accessions, relevant))
            metrics_by_method["keyword"].append(evaluate_accessions(keyword_accessions, relevant))
            metrics_by_method["hybrid"].append(evaluate_accessions(hybrid_accessions, relevant))

            print(f"\nQuery: {query}")
            print(f"Relevant:     {', '.join(sorted(relevant))}")
            print(f"Vector top 5: {', '.join(vector_accessions[:5])}")
            print(f"Keyword top 5: {', '.join(keyword_accessions[:5]) or '(none)'}")
            print(f"Hybrid top 5: {', '.join(hybrid_accessions[:5])}")

    print("\n=== Average Metrics ===")
    print_summary("vector", metrics_by_method["vector"])
    print_summary("keyword", metrics_by_method["keyword"])
    print_summary("hybrid", metrics_by_method["hybrid"])


if __name__ == "__main__":
    main()
