"""Tests for the thin agent wrapper around Structure Studio links."""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from Protein_agent.structure_tools import create_structure_view_link
from protein_structure_view.uniprot import fetch_uniprot_structure_entry


class StructureAgentToolTests(unittest.TestCase):
    @patch("Protein_agent.structure_tools.fetch_uniprot_structure_entry")
    def test_structure_tool_fetches_raw_uniprot_crossrefs_when_only_accession_is_given(
        self,
        fetch_uniprot_structure_entry,
    ) -> None:
        fetch_uniprot_structure_entry.return_value = {
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

        result = create_structure_view_link(
            accession="P42212",
            protein_name="Green Fluorescent Protein",
            summary="GFP forms a beta barrel around a buried chromophore.",
            focus_residues=[
                {"chain": "A", "residue_number": 65, "label": "Chromophore region"}
            ],
        )

        self.assertTrue(result["ok"])
        self.assertEqual("1GFL", result["selected_pdb_id"])
        self.assertIn("payload=", result["viewer_url"])
        fetch_uniprot_structure_entry.assert_called_once_with("P42212")

    @patch("Protein_agent.structure_tools.fetch_uniprot_structure_entry")
    def test_structure_tool_skips_uniprot_fetch_when_explicit_pdb_id_is_given(
        self,
        fetch_uniprot_structure_entry,
    ) -> None:
        result = create_structure_view_link(
            accession="P42212",
            protein_name="Green Fluorescent Protein",
            pdb_id="1GFL",
        )

        self.assertTrue(result["ok"])
        self.assertEqual("1GFL", result["selected_pdb_id"])
        fetch_uniprot_structure_entry.assert_not_called()

    @patch("protein_structure_view.uniprot.requests.get")
    def test_fetch_uniprot_structure_entry_keeps_raw_crossrefs(self, requests_get) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "primaryAccession": "P42212",
            "uniProtKBCrossReferences": [{"database": "PDB", "id": "1GFL"}],
        }
        requests_get.return_value = response

        result = fetch_uniprot_structure_entry("P42212")

        self.assertEqual("P42212", result["primaryAccession"])
        self.assertEqual("1GFL", result["uniProtKBCrossReferences"][0]["id"])
        requests_get.assert_called_once()


if __name__ == "__main__":
    unittest.main()
