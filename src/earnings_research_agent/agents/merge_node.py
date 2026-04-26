"""Merge node for the earnings research graph.

This node operates deterministically (no LLM call) to combine the outputs
from the parallel TRANSCRIPT and PEER branches. It enforces the Fix 3
guardrail by filtering out competitive table rows with low confidence.
"""
from __future__ import annotations

from typing import Any

from earnings_research_agent.state.graph_state import GraphState
from earnings_research_agent.state.schemas import (
    FinalReport,
    SegmentConfidence,
    CompetitiveRow,
)
from earnings_research_agent.utils.exceptions import MergeError
from earnings_research_agent.utils.logging import get_logger

logger = get_logger(__name__)


def merge_node(state: GraphState) -> dict[str, Any]:
    """Combine parallel branch outputs and enforce confidence guardrails."""
    logger.info("Merging branch outputs for %s", state.get("ticker", "Unknown"))

    # 1. Validate that all required branch outputs are present
    required_keys = [
        "executive_summary",
        "signal_cards",
        "temporal_deltas",
        "industry_trends",
        "competitive_table",
    ]

    for key in required_keys:
        if state.get(key) is None:
            raise MergeError(f"Missing required branch output: {key}")

    # 2. Enforce Fix 3 Guardrail: Filter competitive table rows
    # Suppress rows where confidence is MISSING (< 2 companies share the segment)
    filtered_table: list[CompetitiveRow] = []
    
    for row in state["competitive_table"]:
        # A valid cell is one that isn't explicitly flagged as MISSING
        valid_cells = [
            cell for cell in row.cells 
            if cell.confidence != SegmentConfidence.MISSING
        ]
        
        # If at least 2 companies share the segment, we keep the row
        if len(valid_cells) >= 2:
            filtered_table.append(CompetitiveRow(offering_name=row.offering_name, cells=valid_cells))
        else:
            logger.debug(
                "Suppressing segment row '%s' due to low confidence/missing data.", 
                row.offering_name
            )

    # 3. Construct the FinalReport schema
    merged_report = FinalReport(
        ticker=state["ticker"],
        peers=state["peers"],
        executive_summary=state["executive_summary"],
        signal_cards=state["signal_cards"],
        temporal_deltas=state.get("temporal_deltas") or [],
        industry_trends=state["industry_trends"],
        competitive_table=filtered_table,
    )

    logger.info("Merge complete. Final report constructed.")

    # Return the payload to write to GraphState
    return {"merged_report": merged_report}