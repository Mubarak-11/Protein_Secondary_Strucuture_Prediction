# Protein Structure Studio Handoff

The assistant does not render protein structures directly. It builds a structured
payload URL for Protein Structure Studio, which owns the NGL Viewer rendering,
styling, residue highlighting, camera behavior, and visual storytelling.

## Boundary

Assistant responsibilities:

- choose or verify the protein accession,
- resolve a PDB structure candidate,
- select chains or focus residues when they are known,
- summarize retrieved facts, model predictions, and uncertainty,
- generate the Structure Studio link.

Structure Studio responsibilities:

- load the PDB coordinates,
- render all 3D views,
- apply visual modes and camera framing,
- highlight residues, chains, ligands, contacts, and annotations,
- present the visual story.

## Link Format

The local demo integration uses a URL-safe base64 JSON payload:

```text
http://127.0.0.1:8765/protein-sculpture-studio.html?payload=<base64url-json>
```

Override the base URL with:

```text
PROTEIN_STRUCTURE_STUDIO_URL=https://your-structure-studio.example/view.html
```

The default intentionally points to the local Structure Studio server so the
protein agent and visualization repo stay separate during demo recording.

## Python API

```python
from protein_structure_view import FocusResidue, create_structure_view

view = create_structure_view(
    protein_name="Green Fluorescent Protein",
    uniprot_id="P42212",
    uniprot_entry=uniprot_entry,
    summary="GFP forms a beta barrel around a buried chromophore.",
    focus_residues=[
        FocusResidue(chain="A", residue_number=65, label="Chromophore region")
    ],
)

print(view.viewer_url)
```

The `uniprot_entry` can be a raw UniProt JSON object containing
`uniProtKBCrossReferences`, or a normalized object containing `pdb_crossrefs`.

## Payload Shape

```json
{
  "protein_name": "Green Fluorescent Protein",
  "uniprot_id": "P42212",
  "pdb_id": "1GFL",
  "chains": ["A"],
  "focus_residues": [
    {
      "chain": "A",
      "residue_number": 65,
      "label": "Chromophore region"
    }
  ],
  "view_mode": "Function",
  "summary": "GFP forms a beta barrel around a buried chromophore.",
  "source": {
    "database": "RCSB PDB",
    "url": "https://www.rcsb.org/structure/1GFL",
    "experimental_method": "X-ray",
    "resolution": 1.9
  }
}
```

## Agent Integration

The ADK agent exposes one thin tool that calls `create_structure_view`. The tool
is used near the end of combined protein workflows, after retrieval, UniProt
verification, and Q3/Q8 prediction when those steps are relevant.
