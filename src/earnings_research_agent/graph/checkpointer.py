"""Checkpointer factory.

Returns a MemorySaver for local dev/test runs and a PostgresSaver for
production when POSTGRES_URL is set. This is the single place that decides
which persistence backend the graph uses — nothing else should instantiate
a checkpointer directly.

Usage:
    from earnings_research_agent.graph.checkpointer import get_checkpointer
    graph = build_graph(checkpointer=get_checkpointer())
"""
from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

from earnings_research_agent.utils.config import settings
from earnings_research_agent.utils.exceptions import CheckpointerError
from earnings_research_agent.utils.logging import get_logger

logger = get_logger(__name__)


def get_checkpointer() -> BaseCheckpointSaver:
    """Return the appropriate checkpointer based on runtime config.

    - POSTGRES_URL set  → PostgresSaver (production; persists across restarts,
      required for interrupt/resume across process boundaries).
    - POSTGRES_URL empty → MemorySaver (dev/test only; state is lost on exit).

    Raises:
        CheckpointerError: If POSTGRES_URL is set but the connection fails.
    """
    if settings.postgres_url:
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            checkpointer = PostgresSaver.from_conn_string(settings.postgres_url)
            logger.info("Using PostgresSaver checkpointer.")
            return checkpointer
        except Exception as exc:
            raise CheckpointerError(
                f"Failed to initialise PostgresSaver: {exc}"
            ) from exc

    logger.warning(
        "POSTGRES_URL not set — using MemorySaver. "
        "Graph state will be lost when the process exits."
    )
    return MemorySaver()
