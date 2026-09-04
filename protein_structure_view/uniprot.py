"""UniProt helpers used only for structure-view resolution."""

from __future__ import annotations

from typing import Any

import requests

UNIPROT_BASE = "https://rest.uniprot.org/uniprotkb"


def fetch_uniprot_structure_entry(accession: str) -> dict[str, Any]:
    """Fetch raw UniProt JSON so PDB cross-references remain available."""

    normalized = accession.strip()
    if not normalized:
        raise ValueError("accession must not be empty")

    response = requests.get(f"{UNIPROT_BASE}/{normalized}.json", timeout=15)
    response.raise_for_status()
    return response.json()
