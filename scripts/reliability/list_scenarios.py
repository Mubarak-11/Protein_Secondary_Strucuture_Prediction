"""Print the ProteinResearchAgent reliability scenario checklist."""

from __future__ import annotations

from Protein_agent.reliability import RELIABILITY_SCENARIOS


def main() -> None:
    """Print scenario prompts and expected behaviors for manual eval runs."""

    for scenario in RELIABILITY_SCENARIOS:
        print(f"{scenario.name}")
        print(f"  prompt: {scenario.user_prompt}")
        print("  expected behavior:")

        for behavior in scenario.expected_behavior:
            print(f"    - {behavior}")

        print()


if __name__ == "__main__":
    main()
