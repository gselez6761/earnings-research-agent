"""Feedback Logging Node.

Deterministically appends the human feedback and the exact model 
output snapshot to a local JSONL file for audit and prompt tuning.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from earnings_research_agent.state.graph_state import GraphState
from earnings_research_agent.state.schemas import FeedbackEntry
from earnings_research_agent.utils.config import settings
from earnings_research_agent.utils.exceptions import FeedbackStoreError
from earnings_research_agent.utils.logging import get_logger

logger = get_logger(__name__)


def log_feedback_node(state: GraphState) -> dict[str, Any]:
    """Append the human feedback and report snapshot to the local JSONL store."""
    logger.info("Logging feedback to structured store...")

    feedback = state.get("human_feedback")
    report = state.get("merged_report")

    if not feedback or not report:
        logger.warning("Missing feedback or report; bypassing feedback logging.")
        return {}

    # Construct the full audit record
    entry = FeedbackEntry(
        thread_id=state["thread_id"],
        ticker=state["ticker"],
        report_snapshot=report,  # Exactly what the model generated
        human_feedback=feedback  # Exactly what the analyst did/said
    )

    store_path = Path(settings.feedback_store_path)
    
    try:
        # Ensure the directory exists
        store_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Append the Pydantic model as a JSON string to the JSONL file
        with open(store_path, "a", encoding="utf-8") as f:
            f.write(entry.model_dump_json() + "\n")
            
        logger.info("Successfully appended audit record to %s", store_path)
    except Exception as e:
        logger.error("Failed to write to feedback store: %s", e)
        raise FeedbackStoreError(f"Could not save feedback: {e}")

    # Return as a list so LangGraph's operator.add appends it to state.feedback_store
    return {"feedback_store": [entry]}