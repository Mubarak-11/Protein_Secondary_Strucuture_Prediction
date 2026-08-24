"""Configuration helpers for the ProteinResearchAgent runtime."""

from __future__ import annotations

import os


DEFAULT_AGENT_MODEL = "gemini-3.1-pro-preview"
AGENT_MODEL_ENV_VAR = "PROTEIN_AGENT_MODEL"


def get_agent_model() -> str:
    """Return the ADK model used for local agent runs and demos."""

    return os.getenv(AGENT_MODEL_ENV_VAR, DEFAULT_AGENT_MODEL)
