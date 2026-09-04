"""Build Structure Studio payloads without depending on viewer internals."""

from __future__ import annotations

from typing import Any

from .models import FocusResidue, StructureCandidate


DEFAULT_VIEW_MODE = "Function"


def build_viewer_payload(
    *,
    protein_name: str,
    uniprot_id: str,
    structure: StructureCandidate,
    summary: str = "",
    focus_residues: list[FocusResidue] | None = None,
    view_mode: str = DEFAULT_VIEW_MODE,
) -> dict[str, Any]:
    """Return the versioned payload consumed by Protein Structure Studio."""

    payload: dict[str, Any] = {
        "protein_name": protein_name,
        "uniprot_id": uniprot_id,
        "pdb_id": structure.pdb_id,
        "chains": structure.chains,
        "focus_residues": [
            residue.to_payload()
            for residue in focus_residues or []
        ],
        "view_mode": view_mode,
        "summary": summary,
        "source": {
            "database": "RCSB PDB",
            "url": structure.source_url,
            "experimental_method": structure.method,
            "resolution": structure.resolution,
        },
    }
    return payload
