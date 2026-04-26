"""Structured logging configuration for the earnings research agent.

Every module should call get_logger(__name__) rather than using the
root logger directly. This keeps log output consistent and attributable.

Google style: one logger per module, named after the module path.
"""

from __future__ import annotations

import logging
import sys

# Suppress noisy edgar library logs — it logs at WARNING level for every
# HTTP request, which would drown out agent-level logs.
logging.getLogger("edgar").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("pinecone").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name.

    Args:
        name: Typically __name__ from the calling module.

    Returns:
        A Logger instance writing to stdout at INFO level.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    return logger