"""Refine Node.

Applies the human analyst's structured edits (signal overrides and table
cell corrections) to the merged report via a focused LLM rewrite. Returns
an updated FinalReport that human_review_node will present for re-approval.
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from earnings_research_agent.state.graph_state import GraphState
from earnings_research_agent.state.schemas import FinalReport
from earnings_research_agent.utils.config import settings
from earnings_research_agent.utils.logging import get_logger

logger = get_logger(__name__)


def _build_edits_text(feedback) -> str:
    parts = []
    if feedback.signal_overrides:
        lines = []
        for o in feedback.signal_overrides:
            line = f"- Signal '{o.original_headline}':"
            if o.corrected_signal_type:
                line += f" type → {o.corrected_signal_type.value}"
            if o.corrected_headline:
                line += f" | headline → '{o.corrected_headline}'"
            if o.corrected_detail:
                line += f" | detail → '{o.corrected_detail}'"
            if o.reason:
                line += f" | reason: {o.reason}"
            lines.append(line)
        parts.append("Signal overrides:\n" + "\n".join(lines))
    if feedback.table_edits:
        lines = []
        for e in feedback.table_edits:
            line = f"- {e.offering_name}/{e.ticker}:"
            if e.corrected_revenue_current:
                line += f" revenue → {e.corrected_revenue_current}"
            if e.corrected_segment_name:
                line += f" | segment → {e.corrected_segment_name}"
            if e.reason:
                line += f" | reason: {e.reason}"
            lines.append(line)
        parts.append("Table edits:\n" + "\n".join(lines))
    if feedback.free_text_note:
        parts.append(f"Analyst note: {feedback.free_text_note}")
    return "\n\n".join(parts) if parts else "No specific edits provided."


def refine_node(state: GraphState) -> dict[str, Any]:
    """Apply human analyst edits to the merged report."""
    feedback = state.get("human_feedback")
    report = state.get("merged_report")

    if not feedback or not report:
        logger.warning("refine_node called without feedback or report; returning unchanged.")
        return {}

    logger.info("Applying human edits for %s...", state.get("ticker"))

    system_prompt = (
        "You are an equity research analyst. You have a structured report in JSON format "
        "and a precise set of corrections from a senior analyst. Apply the corrections exactly "
        "as specified. Do not change any part of the report not covered by the edits. "
        "Return the updated report matching the same JSON schema."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Original report:\n{report}\n\nCorrections to apply:\n{edits}"),
    ])

    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=0.0,
    ).with_structured_output(FinalReport)

    chain = prompt | llm

    try:
        refined: FinalReport = chain.invoke({
            "report": report.model_dump_json(indent=2),
            "edits": _build_edits_text(feedback),
        })
    except Exception as e:
        logger.error("Failed to refine report: %s", e)
        raise

    logger.info("Report refinement complete.")
    return {"merged_report": refined}
