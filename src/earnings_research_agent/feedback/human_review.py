"""Human Review Node.

The graph is compiled with interrupt_before=["human_review_node"], so
LangGraph pauses before this node runs. The backend injects human_feedback
into the state via update_state() and then resumes. By the time this node
executes, state["human_feedback"] already contains the analyst's decision.
"""
from __future__ import annotations

from typing import Any

from earnings_research_agent.state.graph_state import GraphState
from earnings_research_agent.utils.logging import get_logger

logger = get_logger(__name__)


async def human_review_node(state: GraphState) -> dict[str, Any]:
    """Log the human feedback that was injected before this node ran."""
    feedback = state.get("human_feedback")
    if feedback:
        logger.info(
            "Human review received. Action: %s. Overrides: %d, Edits: %d",
            feedback.action.value,
            len(feedback.signal_overrides),
            len(feedback.table_edits),
        )
    else:
        logger.warning("human_review_node ran without human_feedback in state.")
    return {}
