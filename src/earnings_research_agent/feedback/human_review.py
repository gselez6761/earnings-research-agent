"""Human Review Node.

Pauses the LangGraph execution to allow a human analyst to review,
approve, edit, or reject the generated report. Relies on LangGraph's
interrupt() pattern.
"""
from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from earnings_research_agent.state.graph_state import GraphState
from earnings_research_agent.state.schemas import HumanFeedback
from earnings_research_agent.utils.logging import get_logger

logger = get_logger(__name__)


def human_review_node(state: GraphState) -> dict[str, Any]:
    """Interrupt the graph to allow for human review of the merged report."""
    logger.info("Pausing graph for human review for ticker %s...", state.get("ticker"))

    # The interrupt pauses execution. The dict passed here is what the 
    # frontend/client will receive when polling the graph state.
    feedback_data = interrupt({
        "report": state.get("merged_report"),
        "signal_cards": state.get("signal_cards"),
        "temporal_deltas": state.get("temporal_deltas"),
        "competitive_table": state.get("competitive_table")
    })

    # When the graph is resumed via Command(resume=...), it passes the data here.
    # We ensure it is cast to our strict Pydantic schema.
    if isinstance(feedback_data, HumanFeedback):
        feedback = feedback_data
    else:
        feedback = HumanFeedback(**feedback_data)

    logger.info(
        "Human review complete. Analyst action: %s. Overrides: %d, Edits: %d", 
        feedback.action.value,
        len(feedback.signal_overrides),
        len(feedback.table_edits)
    )
    
    # Write the feedback into the GraphState so downstream nodes can route/log it
    return {"human_feedback": feedback}