"""Fetch raw UniProt records into a local JSONL seed file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests

from protein_retrieval.uniprot import fetch_uniprot_record, read_accessions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accessions", type=Path, default=Path("dataset/uniprot_seed_accessions.txt"))
    parser.add_argument("--out", type=Path, default=Path("dataset/uniprot_seed_jsonl"))
    args = parser.parse_args()

    accessions = read_accessions(args.accessions)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fetched_count = 0

    with args.out.open("w", encoding="utf-8") as output_file:
        for accession in accessions:
            try:
                record = fetch_uniprot_record(accession)
            except requests.RequestException as exc:
                print(f"[skip] {accession}: {exc}")
                continue

            output_file.write(json.dumps(record) + "\n")
            fetched_count += 1
            protein_name = (
                record.get("proteinDescription", {})
                .get("recommendedName", {})
                .get("fullName", {})
                .get("value", "unknown_protein")
            )
            print(f"[ok] {accession}: {protein_name}")

    print(f"Wrote {fetched_count}/{len(accessions)} records to {args.out}")


if __name__ == "__main__":
    main()
