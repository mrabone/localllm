# Re-export the shared logging utilities so that existing imports of
# `rag.logging_utils` continue to work without modification.
from common.logging_utils import JsonFormatter, setup_logging

__all__ = ["JsonFormatter", "setup_logging"]
