"""Conditional edge routing logic for LangGraph.

This module contains the routing functions that determine the next node
based on the state, specifically handling the human feedback loop.
"""
from __future__ import annotations

from typing import Literal

from earnings_research_agent.state.graph_state import GraphState
from earnings_research_agent.state.schemas import FeedbackAction
from earnings_research_agent.utils.logging import get_logger

logger = get_logger(__name__)


def route_on_feedback(state: GraphState) -> Literal["approve", "edit", "reject"]:
    """Route the graph based on the human analyst's feedback action.

    Args:
        state: The current GraphState containing human_feedback.

    Returns:
        The string literal for the next route ("approve", "edit", or "reject").
        These map to END or refine_node in the graph definition.
    """
    feedback = state.get("human_feedback")

    if not feedback:
        logger.warning("No human feedback found in state. Defaulting to reject to halt.")
        return "reject"

    action = feedback.action
    logger.info("Routing based on feedback action: %s", action.value)

    if action == FeedbackAction.APPROVE:
        return "approve"
    elif action == FeedbackAction.EDIT:
        return "edit"
    elif action == FeedbackAction.REJECT:
        return "reject"
    else:
        logger.error("Unknown feedback action: %s. Defaulting to reject.", action)
        return "reject"