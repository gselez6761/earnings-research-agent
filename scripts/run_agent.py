"""Entry point to run the earnings research agent locally.

This script demonstrates the full LangGraph execution flow, including
the interrupt() for human review and the resume() with human feedback.

Usage:
    python scripts/run_agent.py --ticker AMZN --thread-id session_001
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from pathlib import Path
from pprint import pprint

# Ensure the src package is importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langgraph.types import Command

from earnings_research_agent.graph.checkpointer import get_checkpointer
from earnings_research_agent.graph.graph import build_graph
from earnings_research_agent.state.schemas import FeedbackAction, HumanFeedback
from earnings_research_agent.utils.logging import get_logger

logger = get_logger(__name__)


async def main(ticker: str, thread_id: str) -> None:
    """Execute the LangGraph agent for a given ticker, handling human review."""
    logger.info("Starting research run for %s (Thread: %s)", ticker, thread_id)

    checkpointer = get_checkpointer()
    graph = build_graph(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {"ticker": ticker, "thread_id": thread_id}

    logger.info("Invoking graph...")

    # Run graph until it hits the interrupt() in human_review_node
    async for event in graph.astream(initial_state, config=config, stream_mode="values"):
        if "merged_report" in event and event["merged_report"]:
            logger.info("Graph generated merged report. Waiting for human review...")

    state = graph.get_state(config)

    # Check if we are paused at the human_review_node
    if state.next and state.next[0] == "human_review_node":
        logger.info("Graph interrupted. Ready for human feedback.")

        print("\n" + "=" * 40)
        print(" MOCK HUMAN REVIEW INTERFACE ")
        print("=" * 40)
        action_input = input("Enter action (approve/edit/reject) [approve]: ").strip().lower()

        if action_input == "reject":
            action = FeedbackAction.REJECT
        elif action_input == "edit":
            action = FeedbackAction.EDIT
        else:
            action = FeedbackAction.APPROVE

        note = input("Enter optional analyst note: ").strip()

        feedback = HumanFeedback(
            action=action,
            free_text_note=note if note else None,
            quality_rating=4,
        )

        logger.info("Submitting human feedback: %s", action.value)

        async for event in graph.astream(
            Command(resume=feedback), config=config, stream_mode="values"
        ):
            pass

        final_state = graph.get_state(config)
        logger.info("Graph execution complete.")

        print("\n--- FINAL STATE ---")
        if final_state.values.get("merged_report"):
            pprint(final_state.values["merged_report"].model_dump())
        else:
            logger.warning("No merged report found in final state.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the LangGraph Equity Research Agent.")
    parser.add_argument("--ticker", required=True, help="Target ticker symbol (e.g., AAPL).")
    parser.add_argument(
        "--thread-id",
        default=str(uuid.uuid4()),
        help="Unique thread ID for checkpointer.",
    )
    args = parser.parse_args()
    asyncio.run(main(args.ticker, args.thread_id))
