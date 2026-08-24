"""Runtime setup for retrieval demos and MCP processes."""

from __future__ import annotations

import logging
import os


NOISY_RETRIEVAL_LOGGERS: tuple[str, ...] = (
    "huggingface_hub",
    "sentence_transformers",
    "transformers",
    "urllib3.connectionpool",
)


def configure_retrieval_runtime(log_level: int = logging.ERROR) -> None:
    """Quiet Hugging Face stack noise without hiding application errors."""

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    for logger_name in NOISY_RETRIEVAL_LOGGERS:
        logging.getLogger(logger_name).setLevel(log_level)

    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
        transformers_logging.disable_progress_bar()
    except Exception:
        logging.getLogger(__name__).debug("transformers logging setup skipped", exc_info=True)

    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except Exception:
        logging.getLogger(__name__).debug("huggingface_hub progress setup skipped", exc_info=True)
