"""Warm the retrieval embedding model for demo runs."""

from __future__ import annotations

import argparse
import os

from protein_retrieval.config import get_embedding_model_name, load_config
from protein_retrieval.service import warm_embedding_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.getenv("EMBEDDING_MODEL"))
    parser.add_argument("--text", default="protein retrieval warmup")
    args = parser.parse_args()

    load_config()
    model_name = args.model or get_embedding_model_name()
    result = warm_embedding_model(model_name=model_name, warmup_text=args.text)

    print(f"Warmed embedding model: {result['embedding_model']}")
    print(f"Embedding dimensions: {result['embedding_dimensions']}")


if __name__ == "__main__":
    main()
