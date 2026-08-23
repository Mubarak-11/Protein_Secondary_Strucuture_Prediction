from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
import requests
from psycopg.types.json import Jsonb

UNIPROT_BASE = "https://rest.uniprot.org/uniprotkb"


def read_accessions(path: Path) -> list[str]:
    accessions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        accession = line.strip()
        if accession and not accession.startswith("#"):
            accessions.append(accession)
    return accessions


def fetch_uniprot_record(accession: str) -> dict:
    response = requests.get(f"{UNIPROT_BASE}/{accession}.json", timeout=25)
    response.raise_for_status()
    return response.json()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def first_text(value: Any) -> str | None:
    if isinstance(value, dict):
        return value.get("value")
    return None


def extract_gene_names(record: dict[str, Any]) -> list[str]:
    gene_names = []
    for gene in record.get("genes", []):
        gene_name = first_text(gene.get("geneName"))
        if gene_name:
            gene_names.append(gene_name)
        for synonym in gene.get("synonyms", []):
            synonym_name = first_text(synonym)
            if synonym_name:
                gene_names.append(synonym_name)
    return sorted(set(gene_names))


def extract_function_text(record: dict[str, Any]) -> str:
    function_parts = []
    for comment in record.get("comments", []):
        if comment.get("commentType") != "FUNCTION":
            continue
        for text in comment.get("texts", []):
            value = text.get("value")
            if value:
                function_parts.append(value)
    return " ".join(function_parts)


def extract_keywords(record: dict[str, Any]) -> list[str]:
    return sorted({keyword["name"] for keyword in record.get("keywords", []) if keyword.get("name")})


def extract_go_terms(record: dict[str, Any]) -> list[str]:
    return sorted({
        ref["id"]
        for ref in record.get("uniProtKBCrossReferences", [])
        if ref.get("database") == "GO" and ref.get("id")
    })


def extract_ec_numbers(record: dict[str, Any]) -> list[str]:
    recommended_name = record.get("proteinDescription", {}).get("recommendedName", {})
    return sorted({ec["value"] for ec in recommended_name.get("ecNumbers", []) if ec.get("value")})


def extract_evidence_tags(record: dict[str, Any]) -> list[str]:
    evidence_tags = []

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            for evidence in value.get("evidences", []):
                if isinstance(evidence, dict) and evidence.get("evidenceCode"):
                    evidence_tags.append(evidence["evidenceCode"])
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(record)
    return sorted(set(evidence_tags))


def build_search_text(normalized: dict[str, Any]) -> str:
    parts = [
        normalized.get("accession"),
        normalized.get("uniprot_id"),
        normalized.get("protein_name"),
        normalized.get("recommended_name"),
        " ".join(normalized.get("gene_names", [])),
        normalized.get("organism"),
        " ".join(normalized.get("keywords", [])),
        " ".join(normalized.get("go_terms", [])),
        " ".join(normalized.get("ec_numbers", [])),
        normalized.get("function_text"),
    ]
    return "\n".join(part for part in parts if part)


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    protein_description = record.get("proteinDescription", {})
    recommended_name = protein_description.get("recommendedName", {})
    accession = record["primaryAccession"]
    protein_name = first_text(recommended_name.get("fullName"))
    organism = record.get("organism", {})
    sequence = record.get("sequence", {})

    normalized = {
        "accession": accession,
        "uniprot_id": record.get("uniProtkbId"),
        "reviewed": "swiss-prot" in record.get("entryType", "").lower(),
        "protein_name": protein_name,
        "recommended_name": protein_name,
        "gene_names": extract_gene_names(record),
        "organism": organism.get("scientificName"),
        "organism_taxon_id": organism.get("taxonId"),
        "sequence": sequence.get("value"),
        "sequence_length": sequence.get("length"),
        "function_text": extract_function_text(record),
        "keywords": extract_keywords(record),
        "go_terms": extract_go_terms(record),
        "ec_numbers": extract_ec_numbers(record),
        "evidence_tags": extract_evidence_tags(record),
        "source_database": "UniProtKB",
        "source_version": str(record.get("entryAudit", {}).get("sequenceVersion", "")),
        "source_url": f"https://www.uniprot.org/uniprotkb/{accession}/entry",
        "retrieved_at": datetime.now(timezone.utc),
        "raw_payload": record,
    }
    normalized["search_text"] = build_search_text(normalized)
    normalized["normalized_payload"] = {
        key: value.isoformat() if isinstance(value, datetime) else value
        for key, value in normalized.items()
        if key not in {"raw_payload", "normalized_payload"}
    }
    return normalized


UPSERT_SQL = """
INSERT INTO proteins(
    accession, uniprot_id, reviewed, protein_name, recommended_name, gene_names,
    organism, organism_taxon_id, sequence, sequence_length, function_text,
    keywords, go_terms, ec_numbers, evidence_tags, source_database, source_version,
    source_url, retrieved_at, raw_payload, normalized_payload, search_text
)
VALUES (
    %(accession)s, %(uniprot_id)s, %(reviewed)s, %(protein_name)s, %(recommended_name)s,
    %(gene_names)s, %(organism)s, %(organism_taxon_id)s, %(sequence)s,
    %(sequence_length)s, %(function_text)s, %(keywords)s, %(go_terms)s,
    %(ec_numbers)s, %(evidence_tags)s, %(source_database)s, %(source_version)s,
    %(source_url)s, %(retrieved_at)s, %(raw_payload)s, %(normalized_payload)s,
    %(search_text)s
)
ON CONFLICT(accession) DO UPDATE SET
    uniprot_id = EXCLUDED.uniprot_id,
    reviewed = EXCLUDED.reviewed,
    protein_name = EXCLUDED.protein_name,
    recommended_name = EXCLUDED.recommended_name,
    gene_names = EXCLUDED.gene_names,
    organism = EXCLUDED.organism,
    organism_taxon_id = EXCLUDED.organism_taxon_id,
    sequence = EXCLUDED.sequence,
    sequence_length = EXCLUDED.sequence_length,
    function_text = EXCLUDED.function_text,
    keywords = EXCLUDED.keywords,
    go_terms = EXCLUDED.go_terms,
    ec_numbers = EXCLUDED.ec_numbers,
    evidence_tags = EXCLUDED.evidence_tags,
    source_database = EXCLUDED.source_database,
    source_version = EXCLUDED.source_version,
    source_url = EXCLUDED.source_url,
    retrieved_at = EXCLUDED.retrieved_at,
    raw_payload = EXCLUDED.raw_payload,
    normalized_payload = EXCLUDED.normalized_payload,
    search_text = EXCLUDED.search_text;
"""


def insert_record(conn: psycopg.Connection, normalized: dict[str, Any]) -> None:
    params = {
        **normalized,
        "evidence_tags": Jsonb(normalized["evidence_tags"]),
        "raw_payload": Jsonb(normalized["raw_payload"]),
        "normalized_payload": Jsonb(normalized["normalized_payload"]),
    }
    conn.execute(UPSERT_SQL, params)

