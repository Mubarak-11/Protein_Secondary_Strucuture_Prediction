"""Resolve PDB structure candidates from UniProt-derived cross-references."""

from __future__ import annotations

from typing import Any

from .models import StructureCandidate


PREFERRED_STRUCTURE_BY_ACCESSION = {
    "P68871": "2HHB",  # Human hemoglobin subunit beta
    "P69905": "2HHB",  # Human hemoglobin subunit alpha
}


def parse_pdb_crossrefs(entry: dict[str, Any]) -> list[StructureCandidate]:
    """Extract PDB candidates from a UniProt entry or agent-normalized entry."""

    candidates = entry.get("pdb_crossrefs")
    if candidates is None:
        candidates = [
            _candidate_from_uniprot_ref(ref)
            for ref in entry.get("uniProtKBCrossReferences", [])
            if ref.get("database") == "PDB"
        ]
    else:
        candidates = [_candidate_from_normalized_ref(ref) for ref in candidates]

    return [candidate for candidate in candidates if candidate.pdb_id]


def choose_best_structure(
    candidates: list[StructureCandidate],
    *,
    uniprot_id: str | None = None,
) -> StructureCandidate | None:
    """Choose a representative structure, falling back to method/resolution ranking."""

    if not candidates:
        return None

    preferred_pdb_id = PREFERRED_STRUCTURE_BY_ACCESSION.get(str(uniprot_id or "").upper())
    if preferred_pdb_id:
        for candidate in candidates:
            if candidate.pdb_id == preferred_pdb_id:
                return candidate

    def sort_key(candidate: StructureCandidate) -> tuple[int, float, str]:
        method = candidate.method.lower()
        method_rank = 0 if "x-ray" in method or "electron" in method else 1
        resolution = candidate.resolution if candidate.resolution is not None else 99.0
        return (method_rank, resolution, candidate.pdb_id)

    return sorted(candidates, key=sort_key)[0]


def _candidate_from_uniprot_ref(ref: dict[str, Any]) -> StructureCandidate:
    properties = {
        prop.get("key"): prop.get("value")
        for prop in ref.get("properties", [])
        if prop.get("key") and prop.get("value")
    }
    pdb_id = str(ref.get("id", "")).upper()
    return StructureCandidate(
        pdb_id=pdb_id,
        method=properties.get("Method", ""),
        resolution=_parse_resolution(properties.get("Resolution")),
        chains=_parse_chains(properties.get("Chains")),
        source_url=f"https://www.rcsb.org/structure/{pdb_id}" if pdb_id else "",
    )


def _candidate_from_normalized_ref(ref: dict[str, Any]) -> StructureCandidate:
    pdb_id = str(ref.get("pdb_id") or ref.get("id") or "").upper()
    return StructureCandidate(
        pdb_id=pdb_id,
        method=ref.get("method", ""),
        resolution=_parse_resolution(ref.get("resolution")),
        chains=list(ref.get("chains") or []),
        source_url=ref.get("source_url") or (
            f"https://www.rcsb.org/structure/{pdb_id}" if pdb_id else ""
        ),
    )


def _parse_resolution(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    text = str(value).replace("A", "").replace("angstrom", "").strip()
    try:
        return float(text)
    except ValueError:
        return None


def _parse_chains(value: Any) -> list[str]:
    if not value:
        return []
    text = str(value)
    chain_part = text.split("=")[0]
    chains = []
    for chunk in chain_part.replace("/", ",").replace(";", ",").split(","):
        chain = chunk.strip()
        if chain:
            chains.append(chain)
    return sorted(set(chains))
