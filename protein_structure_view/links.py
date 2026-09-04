"""Create shareable links for Protein Structure Studio."""

from __future__ import annotations

import base64
import json
import os
from typing import Any
from urllib.parse import urlencode

from .models import FocusResidue, StructureCandidate, StructureView
from .payload import DEFAULT_VIEW_MODE, build_viewer_payload
from .pdb_mapping import choose_best_structure, parse_pdb_crossrefs

DEFAULT_STUDIO_URL = "http://127.0.0.1:8765/protein-sculpture-studio.html"
STUDIO_URL_ENV_VAR = "PROTEIN_STRUCTURE_STUDIO_URL"


def encode_payload(payload: dict[str, Any]) -> str:
    """Encode payload as URL-safe base64 JSON for Structure Studio."""

    raw_json = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw_json).decode("ascii").rstrip("=")


def decode_payload(encoded: str) -> dict[str, Any]:
    """Decode a Structure Studio payload token."""

    padded = encoded + ("=" * (-len(encoded) % 4))
    return json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))


def build_studio_url(payload: dict[str, Any], base_url: str | None = None) -> str:
    base_url = base_url or os.getenv(STUDIO_URL_ENV_VAR) or DEFAULT_STUDIO_URL
    return f"{base_url}?{urlencode({'payload': encode_payload(payload)})}"


def create_structure_view(
    *,
    protein_name: str,
    uniprot_id: str,
    uniprot_entry: dict[str, Any] | None = None,
    pdb_id: str | None = None,
    summary: str = "",
    focus_residues: list[FocusResidue] | None = None,
    view_mode: str = DEFAULT_VIEW_MODE,
    base_url: str | None = None,
) -> StructureView:
    """Build a Structure Studio link from a UniProt entry or explicit PDB ID."""

    uncertainty: list[str] = []
    candidates = parse_pdb_crossrefs(uniprot_entry or {})

    selected = _explicit_candidate(pdb_id, candidates) if pdb_id else choose_best_structure(candidates)
    if selected is None:
        raise ValueError("No PDB structure candidate was available for this protein.")

    if not selected.chains:
        uncertainty.append("No preferred chain mapping was available from UniProt.")

    payload = build_viewer_payload(
        protein_name=protein_name,
        uniprot_id=uniprot_id,
        structure=selected,
        summary=summary,
        focus_residues=focus_residues,
        view_mode=view_mode,
    )
    return StructureView(
        viewer_url=build_studio_url(payload, base_url=base_url),
        payload=payload,
        selected_pdb_id=selected.pdb_id,
        selected_structure=selected,
        uncertainty=uncertainty,
    )


def _explicit_candidate(
    pdb_id: str | None,
    candidates: list[StructureCandidate],
) -> StructureCandidate | None:
    normalized = str(pdb_id or "").upper().strip()
    for candidate in candidates:
        if candidate.pdb_id == normalized:
            return candidate
    if not normalized:
        return None
    return StructureCandidate(
        pdb_id=normalized,
        source_url=f"https://www.rcsb.org/structure/{normalized}",
    )
