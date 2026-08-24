""" Reliability contract tests for Protein research Agent. """

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from Protein_agent.reliability import(
    ANSWER_CONTRACT_REQUIREMENTS,
    RELIABILITY_SCENARIOS,
    scenario_names
)

from protein_retrieval.config import MAX_TOP_K, MIN_TOP_K
from protein_retrieval_mcp_server.server import (
    _clamp_top_k,
    _error,
    _success,
    hybrid_search_proteins,
    keyword_search_proteins,
    semantic_search_proteins,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = PROJECT_ROOT /"Protein_agent"/ "agent-prompt.md"


class ReliabilityContractTests(unittest.TestCase):
    def test_expected_reliability_scenarios_are_registered(self) -> None:
        expected_names = {
            "ambiguous_query",
            "invalid_accession",
            "no_result_query",
            "wrong_organism_temptation",
            "long_sequence_prediction_limit",
            "tool_api_failure",
        }

        self.assertEqual(expected_names, scenario_names())

    def test_scenarios_have_prompts_and_expected_behaviors(self) -> None:
        for scenario in RELIABILITY_SCENARIOS:
            with self.subTest(scenario=scenario.name):
                self.assertTrue(scenario.user_prompt.strip())
                self.assertGreaterEqual(len(scenario.expected_behavior), 3)
                self.assertTrue(all(item.strip() for item in scenario.expected_behavior))

    def test_prompt_contains_answer_contract_requirements(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8").lower()

        for requirement in ANSWER_CONTRACT_REQUIREMENTS:
            with self.subTest(requirement=requirement):
                self.assertIn(requirement, prompt)

    def test_prompt_covers_week_three_failure_modes(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8").lower()
        required_phrases = (
            "wrong organism",
            "no reliable result",
            "accession lookup fails",
            "tool or api fails",
            "longer than 512 residues",
        )

        for phrase in required_phrases:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, prompt)


class McpEnvelopeTests(unittest.TestCase):
    @patch("protein_retrieval_mcp_server.server.uuid.uuid4")
    @patch("protein_retrieval_mcp_server.server.logging.exception")
    def test_retrieval_mcp_error_envelope_is_stable(self, logging_exception, uuid4) -> None:
        uuid4.return_value = "fixed-error-id"

        result = _error(RuntimeError("database unavailable"))

        self.assertEqual(
            {
                "ok": False,
                "error": {
                    "code": "protein_retrieval_tool_error",
                    "message": "Protein retrieval tool failed.",
                    "error_id": "fixed-error-id",
            },
        },
        result,
        )
        logging_exception.assert_called_once()

    def test_retrieval_mcp_success_envelope_is_stable(self) -> None:
        result = _success({"query": "kinase", "results": []})

        self.assertEqual(
            {"ok": True, "data": {"query": "kinase", "results": []}},
            result,
        )

    def test_retrieval_mcp_top_k_is_clamped_to_supported_range(self) -> None:
        self.assertEqual(MIN_TOP_K, _clamp_top_k(MIN_TOP_K - 1))
        self.assertEqual(5, _clamp_top_k(5))
        self.assertEqual(MAX_TOP_K, _clamp_top_k(MAX_TOP_K + 1))

    @patch("protein_retrieval.service.keyword_search_proteins")
    def test_keyword_search_tool_wraps_service_result_without_db(self, run_keyword_search) -> None:
        service_result = {"query": "kinase", "method": "keyword", "top_k": 3, "results": []}
        run_keyword_search.return_value = service_result

        result = keyword_search_proteins("kinase", top_k=3)

        self.assertEqual({"ok": True, "data": service_result}, result)
        run_keyword_search.assert_called_once_with(query="kinase", top_k=3)

    @patch("protein_retrieval.service.semantic_search_proteins")
    def test_semantic_search_tool_clamps_top_k_before_service_call(self, run_semantic_search) -> None:
        service_result = {
            "query": "DNA repair",
            "method": "semantic",
            "top_k": MAX_TOP_K,
            "results": [],
        }
        run_semantic_search.return_value = service_result

        result = semantic_search_proteins("DNA repair", top_k=MAX_TOP_K + 50)

        self.assertEqual({"ok": True, "data": service_result}, result)
        run_semantic_search.assert_called_once_with(query="DNA repair", top_k=MAX_TOP_K)

    @patch("protein_retrieval_mcp_server.server.uuid.uuid4")
    @patch("protein_retrieval_mcp_server.server.logging.exception")
    @patch("protein_retrieval.service.hybrid_search_proteins")
    def test_hybrid_search_tool_returns_error_envelope_on_service_failure(
        self,
        run_hybrid_search,
        logging_exception,
        uuid4,
    ) -> None:
        run_hybrid_search.side_effect = ValueError("query must not be empty")
        uuid4.return_value = "validation-error-id"

        result = hybrid_search_proteins("   ", top_k=5)

        self.assertEqual(
            {
                "ok": False,
                "error": {
                    "code": "protein_retrieval_tool_error",
                    "message": "Protein retrieval tool failed.",
                    "error_id": "validation-error-id",
                },
            },
            result,
        )
        run_hybrid_search.assert_called_once_with(query="   ", top_k=5, candidate_k=20)
        logging_exception.assert_called_once()


if __name__ == "__main__":
    unittest.main()
