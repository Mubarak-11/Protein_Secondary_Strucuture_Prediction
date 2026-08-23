"""Normalize raw UniProt JSONL records and insert metadata into Postgres."""

from __future__ import annotations

import argparse
from pathlib import Path

from protein_retrieval.config import get_database_url, load_config
from protein_retrieval.db import connect
from protein_retrieval.uniprot import insert_record, normalize_record, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, default=Path("dataset/uniprot_seed_jsonl"))
    args = parser.parse_args()

    load_config()
    raw_records = read_jsonl(args.jsonl)
    inserted_count = 0

    with connect(get_database_url()) as conn:
        for raw_record in raw_records:
            normalized = normalize_record(raw_record)
            insert_record(conn, normalized)
            inserted_count += 1
            print(
                f" [ok] {normalized['accession']}: "
                f" {normalized.get('protein_name') or 'unknown protein'}"
            )

        conn.commit()

    print(f" Inserted/Updated {inserted_count} protein metadata rows")


if __name__ == "__main__":
    main()
