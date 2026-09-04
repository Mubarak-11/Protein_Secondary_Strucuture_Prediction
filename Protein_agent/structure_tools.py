"""Structure tools for the protein agent: build Structure Studio viewer links."""

from __future__ import annotations
from typing import Any

from protein_structure_view import FocusResidue, create_structure_view
from protein_structure_view.uniprot import fetch_uniprot_structure_entry


def create_structure_view_link(
        accession: str,
        protein_name: str = "",
        summary: str = "",
        pdb_id: str | None = None,
        uniprot_entry: dict[str, Any] | None = None,
        focus_residues: list[dict[str, Any]] | None = None,
        view_mode: str = "Function",
) -> dict[str, Any]:
    """ Create a Protein structure studio link for an accession or PDB ID.

    Args:
        accession: UniProt accession, e.g. 'P0DP24'. If pdb_id is not given,
            the tool fetches the UniProt entry to resolve PDB candidates.
        protein_name: Display name of the protein, e.g. 'Calmodulin-2'.
        summary: Short biological summary shown in the viewer.
        pdb_id: Optional explicit PDB ID, e.g. '5NIN'. When given, no
            UniProt fetch is needed. If the user asks to use a specific PDB
            ID, pass that exact ID here instead of relying on automatic
            structure selection.
        uniprot_entry: Optional raw UniProt JSON entry; fetched automatically
            when omitted.
        focus_residues: Residues to highlight in the viewer. Pass ONE dict
            per residue anchor: {"chain": "A", "residue_number": 21,
            "label": "EF-hand 1"}. residue_number MUST be a single integer
            (e.g. 21). NEVER pass a range string like "21-32" or "21..32" —
            they are rejected with an error. For a region (e.g. a binding
            loop), pass 1-3 REPRESENTATIVE anchor residues with DISTINCT
            labels (e.g. residue 21 labelled "EF-hand 1 start") — do not pass
            every residue of the region, which floods the viewer with
            duplicate layers.
        view_mode: Viewer coloring mode, e.g. 'Function'.
    """

    try:
        structure_entry = uniprot_entry
        if structure_entry is None and pdb_id is None:
            structure_entry = fetch_uniprot_structure_entry(accession)

        residues = [
            FocusResidue(
                chain=str(item.get("chain", "")),
                residue_number=int(item["residue_number"]),
                label=str(item.get("label", "")),
            )
            for item in focus_residues or []
        ]

        view = create_structure_view(
            protein_name=protein_name or accession,
            uniprot_id=accession,
            uniprot_entry=structure_entry,
            pdb_id=pdb_id,
            summary=summary,
            focus_residues=residues,
            view_mode=view_mode,
        )

        return {"ok": True, **view.to_dict()}

    except Exception as exc:
        return {
            "ok": False,
            "error": f"Structure view link could not be created: {exc}",
        }
