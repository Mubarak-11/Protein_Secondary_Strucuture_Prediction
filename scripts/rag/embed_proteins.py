"""Generate BGE embeddings for protein metadata rows."""

from __future__ import annotations

import argparse
import os

from protein_retrieval.config import get_database_url, get_embedding_model_name, load_config
from protein_retrieval.db import connect
from protein_retrieval.embeddings import embed_texts, fetch_unembedded_rows, load_model, update_embedding


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--model", default=os.getenv("EMBEDDING_MODEL"))
    args = parser.parse_args()

    load_config()
    model_name = args.model or get_embedding_model_name()
    model = load_model(model_name)

    with connect(get_database_url()) as conn:
        rows = fetch_unembedded_rows(conn, limit=args.limit)
        if not rows:
            print("No unembedded protein rows found.")
            return

        print(f"Embedding {len(rows)} protein rows")
        embeddings = embed_texts(model, [row["search_text"] for row in rows])

        for row, embedding in zip(rows, embeddings, strict=True):
            update_embedding(conn, row["accession"], embedding, model_name)
            print(f" [ok] embedded {row['accession']}")

        conn.commit()

    print(f"Updated Embedding for {len(rows)} proteins")


if __name__ == "__main__":
    main()
