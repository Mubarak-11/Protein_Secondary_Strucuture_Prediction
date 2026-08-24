"""Tests for demo runtime hardening helpers."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from Protein_agent.config import DEFAULT_AGENT_MODEL, get_agent_model
from protein_retrieval.config import get_local_files_only
from protein_retrieval.embeddings import load_model
from protein_retrieval.runtime import configure_retrieval_runtime
from protein_retrieval_mcp_server.server import _warm_embedding_model_for_demo


class AgentConfigTests(unittest.TestCase):
    def test_demo_agent_model_defaults_to_pro_preview(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(DEFAULT_AGENT_MODEL, get_agent_model())

    def test_agent_model_can_be_overridden_for_local_experiments(self) -> None:
        with patch.dict(os.environ, {"PROTEIN_AGENT_MODEL": "gemini-test-model"}):
            self.assertEqual("gemini-test-model", get_agent_model())


class RetrievalRuntimeTests(unittest.TestCase):
    def test_configure_retrieval_runtime_sets_quiet_hugging_face_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            configure_retrieval_runtime()

            self.assertEqual("false", os.environ["TOKENIZERS_PARALLELISM"])
            self.assertEqual("error", os.environ["TRANSFORMERS_VERBOSITY"])
            self.assertEqual("error", os.environ["HF_HUB_VERBOSITY"])
            self.assertEqual("1", os.environ["HF_HUB_DISABLE_PROGRESS_BARS"])

    def test_configure_retrieval_runtime_preserves_existing_environment(self) -> None:
        existing_env = {
            "TOKENIZERS_PARALLELISM": "true",
            "TRANSFORMERS_VERBOSITY": "warning",
            "HF_HUB_VERBOSITY": "warning",
            "HF_HUB_DISABLE_PROGRESS_BARS": "0",
        }

        with patch.dict(os.environ, existing_env, clear=True):
            configure_retrieval_runtime()

            for key, value in existing_env.items():
                with self.subTest(key=key):
                    self.assertEqual(value, os.environ[key])

    def test_retrieval_model_loading_defaults_to_local_cache_only(self) -> None:
        with patch("protein_retrieval.embeddings.SentenceTransformer") as sentence_transformer:
            load_model("BAAI/bge-small-en-v1.5")

            sentence_transformer.assert_called_once_with(
                "BAAI/bge-small-en-v1.5",
                local_files_only=True,
            )

    def test_retrieval_model_loading_can_allow_remote_resolution(self) -> None:
        with patch.dict(os.environ, {"PROTEIN_RETRIEVAL_LOCAL_FILES_ONLY": "false"}):
            self.assertFalse(get_local_files_only())

        with patch("protein_retrieval.embeddings.SentenceTransformer") as sentence_transformer:
            load_model("BAAI/bge-small-en-v1.5", local_files_only=False)

            sentence_transformer.assert_called_once_with(
                "BAAI/bge-small-en-v1.5",
                local_files_only=False,
            )

    @patch("protein_retrieval.service.warm_embedding_model")
    def test_retrieval_mcp_warmup_runs_by_default(self, warm_embedding_model) -> None:
        warm_embedding_model.return_value = {
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "embedding_dimensions": 384,
        }

        with patch.dict(os.environ, {}, clear=True):
            _warm_embedding_model_for_demo()

        warm_embedding_model.assert_called_once_with()

    @patch("protein_retrieval.service.warm_embedding_model")
    def test_retrieval_mcp_warmup_can_be_disabled(self, warm_embedding_model) -> None:
        with patch.dict(os.environ, {"PROTEIN_RETRIEVAL_MCP_WARMUP": "false"}):
            _warm_embedding_model_for_demo()

        warm_embedding_model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
