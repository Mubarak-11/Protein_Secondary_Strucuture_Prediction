"""Reliability contract for ProteinResearchAgent answers and eval scenarios."""

from __future__ import annotations

from dataclasses import dataclass


ANSWER_CONTRACT_REQUIREMENTS: tuple[str, ...] = (
    "accession",
    "selection rationale",
    "evidence/review status",
    "verified facts",
    "interpretation",
    "uncertainty",
    "missing information",
)


@dataclass(frozen=True)
class ReliabilityScenario:
    """Expected behavior for an agent reliability check."""

    name: str
    user_prompt: str
    expected_behavior: tuple[str, ...]


RELIABILITY_SCENARIOS: tuple[ReliabilityScenario, ...] = (
    ReliabilityScenario(
        name="ambiguous_query",
        user_prompt="Tell me about p53.",
        expected_behavior=(
            "search before answering",
            "prefer reviewed human TP53 when human context is implied or ask for organism if unclear",
            "state the chosen accession and selection rationale",
            "separate retrieved facts from interpretation",
        ),
    ),
    ReliabilityScenario(
        name="invalid_accession",
        user_prompt="Summarize UniProt accession NOT_A_REAL_ACCESSION.",
        expected_behavior=(
            "attempt lookup only if the text is accession-like enough",
            "report that the accession could not be verified when lookup fails",
            "do not invent protein annotations",
            "list missing information",
        ),
    ),
    ReliabilityScenario(
        name="no_result_query",
        user_prompt="Find the protein responsible for blue fire breathing in humans.",
        expected_behavior=(
            "run an appropriate retrieval or UniProt search",
            "say when no reliable candidate is found",
            "avoid forcing a weak match into an answer",
            "state uncertainty and missing information",
        ),
    ),
    ReliabilityScenario(
        name="wrong_organism_temptation",
        user_prompt="Find human nitrogenase and summarize it.",
        expected_behavior=(
            "respect the requested organism",
            "do not substitute bacterial nitrogenase as a human protein",
            "explain the organism mismatch if non-human candidates appear",
            "state uncertainty and missing information",
        ),
    ),
    ReliabilityScenario(
        name="long_sequence_prediction_limit",
        user_prompt="Predict Q3 for human BRCA1.",
        expected_behavior=(
            "retrieve the UniProt entry before prediction",
            "check sequence length before calling prediction",
            "refuse prediction when the sequence is longer than 512 residues",
            "still summarize verified annotations when available",
        ),
    ),
    ReliabilityScenario(
        name="tool_api_failure",
        user_prompt="Find DNA repair proteins and compare the top hit to the training dataset.",
        expected_behavior=(
            "surface tool failure plainly",
            "do not fabricate unavailable tool results",
            "continue with verified partial results only when useful",
            "state missing information",
        ),
    ),
)


def scenario_names() -> set[str]:
    """Return the stable names used by reliability tests and eval wrappers."""

    return {scenario.name for scenario in RELIABILITY_SCENARIOS}
