"""Tests for graceful UniProt tool failures."""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import requests

from Protein_agent.uniprot_tools import get_uniprot_entry, search_uniprot


class UniProtToolFailureTests(unittest.TestCase):
    @patch("Protein_agent.uniprot_tools.requests.get")
    def test_get_uniprot_entry_returns_error_for_invalid_accession(self, requests_get) -> None:
        response = Mock()
        response.status_code = 400
        response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "400 Client Error"
        )
        requests_get.return_value = response

        result = get_uniprot_entry("NOT_A_REAL_ACCESSION")

        self.assertFalse(result["ok"])
        self.assertEqual("NOT_A_REAL_ACCESSION", result["accession"])
        self.assertIn("HTTP 400", result["error"])
        self.assertIn("could not be verified", result["error"])

    @patch("Protein_agent.uniprot_tools.requests.get")
    def test_get_uniprot_entry_returns_error_for_network_failure(self, requests_get) -> None:
        requests_get.side_effect = requests.exceptions.Timeout("request timed out")

        result = get_uniprot_entry("P38398")

        self.assertFalse(result["ok"])
        self.assertEqual("P38398", result["accession"])
        self.assertIn("UniProt lookup failed", result["error"])
        self.assertIn("request timed out", result["error"])

    @patch("Protein_agent.uniprot_tools.requests.get")
    def test_search_uniprot_returns_empty_results_for_http_failure(self, requests_get) -> None:
        response = Mock()
        response.status_code = 503
        response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "503 Server Error"
        )
        requests_get.return_value = response

        result = search_uniprot("TP53 human")

        self.assertFalse(result["ok"])
        self.assertEqual("TP53 human", result["query"])
        self.assertEqual([], result["results"])
        self.assertEqual(0, result["total"])
        self.assertIn("HTTP 503", result["error"])

    @patch("Protein_agent.uniprot_tools.requests.get")
    def test_search_uniprot_success_has_ok_flag(self, requests_get) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "total": 1,
            "results": [
                {
                    "primaryAccession": "P04637",
                    "uniProtkbId": "P53_HUMAN",
                    "entryType": "UniProtKB reviewed (Swiss-Prot)",
                    "proteinDescription": {
                        "recommendedName": {
                            "fullName": {"value": "Cellular tumor antigen p53"}
                        }
                    },
                    "genes": [{"geneName": {"value": "TP53"}}],
                    "organism": {"scientificName": "Homo sapiens"},
                    "sequence": {"length": 393},
                }
            ],
        }
        requests_get.return_value = response

        result = search_uniprot("TP53 human")

        self.assertTrue(result["ok"])
        self.assertEqual(1, result["total"])
        self.assertEqual("P04637", result["results"][0]["accession"])
        self.assertTrue(result["results"][0]["reviewed"])


if __name__ == "__main__":
    unittest.main()
