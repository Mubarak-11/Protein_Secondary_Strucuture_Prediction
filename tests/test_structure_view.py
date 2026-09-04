"""Tests for Protein Structure Studio handoff payloads."""

from __future__ import annotations

import unittest

from protein_structure_view.links import (
    build_studio_url,
    create_structure_view,
    decode_payload,
    encode_payload,
)
from protein_structure_view.models import FocusResidue
from protein_structure_view.pdb_mapping import choose_best_structure, parse_pdb_crossrefs


class StructureViewerPayloadTests(unittest.TestCase):
    def test_payload_encoding_round_trips_url_safe_json(self) -> None:
        payload = {
            "protein_name": "Green Fluorescent Protein",
            "uniprot_id": "P42212",
            "pdb_id": "1GFL",
            "chains": ["A"],
        }

        encoded = encode_payload(payload)

        self.assertNotIn("=", encoded)
        self.assertEqual(payload, decode_payload(encoded))

    def test_build_studio_url_uses_payload_query_parameter(self) -> None:
        payload = {"protein_name": "Hemoglobin", "pdb_id": "2HHB"}

        url = build_studio_url(payload, base_url="https://studio.example/view.html")

        self.assertTrue(url.startswith("https://studio.example/view.html?payload="))
        encoded_payload = url.split("payload=", maxsplit=1)[1]
        self.assertEqual(payload, decode_payload(encoded_payload))

    def test_create_structure_view_builds_payload_from_uniprot_pdb_crossref(self) -> None:
        uniprot_entry = {
            "uniProtKBCrossReferences": [
                {
                    "database": "PDB",
                    "id": "1GFL",
                    "properties": [
                        {"key": "Method", "value": "X-ray"},
                        {"key": "Resolution", "value": "1.90 A"},
                        {"key": "Chains", "value": "A=1-238"},
                    ],
                }
            ]
        }

        view = create_structure_view(
            protein_name="Green Fluorescent Protein",
            uniprot_id="P42212",
            uniprot_entry=uniprot_entry,
            summary="GFP forms a beta barrel around a buried chromophore.",
            focus_residues=[
                FocusResidue(chain="A", residue_number=65, label="Chromophore region")
            ],
            base_url="http://127.0.0.1:8765/protein-sculpture-studio.html",
        )

        self.assertEqual("1GFL", view.selected_pdb_id)
        self.assertEqual(["A"], view.payload["chains"])
        self.assertEqual("P42212", view.payload["uniprot_id"])
        self.assertEqual("1GFL", decode_payload(view.viewer_url.split("payload=", 1)[1])["pdb_id"])
        self.assertEqual(
            [{"chain": "A", "residue_number": 65, "label": "Chromophore region"}],
            view.payload["focus_residues"],
        )

    def test_create_structure_view_allows_explicit_pdb_id_without_uniprot_crossrefs(self) -> None:
        view = create_structure_view(
            protein_name="Example protein",
            uniprot_id="P00000",
            pdb_id="1abc",
            base_url="https://studio.example/view.html",
        )

        self.assertEqual("1ABC", view.selected_pdb_id)
        self.assertEqual("https://www.rcsb.org/structure/1ABC", view.selected_structure.source_url)
        self.assertIn("No preferred chain mapping", view.uncertainty[0])

    def test_create_structure_view_reports_missing_pdb_candidates(self) -> None:
        with self.assertRaisesRegex(ValueError, "No PDB structure candidate"):
            create_structure_view(
                protein_name="No known structure protein",
                uniprot_id="P00000",
                uniprot_entry={"uniProtKBCrossReferences": []},
            )


class StructureCandidateMappingTests(unittest.TestCase):
    def test_parse_pdb_crossrefs_extracts_method_resolution_and_chains(self) -> None:
        entry = {
            "uniProtKBCrossReferences": [
                {"database": "GO", "id": "GO:0000000"},
                {
                    "database": "PDB",
                    "id": "2HHB",
                    "properties": [
                        {"key": "Method", "value": "X-ray"},
                        {"key": "Resolution", "value": "1.74 A"},
                        {"key": "Chains", "value": "A/B/C/D=1-146"},
                    ],
                },
            ]
        }

        candidates = parse_pdb_crossrefs(entry)

        self.assertEqual(1, len(candidates))
        self.assertEqual("2HHB", candidates[0].pdb_id)
        self.assertEqual("X-ray", candidates[0].method)
        self.assertEqual(1.74, candidates[0].resolution)
        self.assertEqual(["A", "B", "C", "D"], candidates[0].chains)

    def test_choose_best_structure_prefers_lower_resolution_experimental_candidate(self) -> None:
        candidates = parse_pdb_crossrefs(
            {
                "pdb_crossrefs": [
                    {"pdb_id": "9ZZZ", "method": "NMR", "resolution": None, "chains": ["A"]},
                    {"pdb_id": "2BBB", "method": "X-ray", "resolution": 2.1, "chains": ["A"]},
                    {"pdb_id": "1AAA", "method": "X-ray", "resolution": 1.5, "chains": ["A"]},
                ]
            }
        )

        best = choose_best_structure(candidates)

        self.assertIsNotNone(best)
        self.assertEqual("1AAA", best.pdb_id)


if __name__ == "__main__":
    unittest.main()
