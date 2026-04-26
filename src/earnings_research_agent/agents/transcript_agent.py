"""Transcript Analysis Agent."""
from __future__ import annotations

import json as _json
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from earnings_research_agent.rag.hallucination_checker import check_citations
from earnings_research_agent.state.graph_state import GraphState
from earnings_research_agent.state.schemas import ExecutiveSummary, SignalCard
from earnings_research_agent.utils.llm import get_llm
from earnings_research_agent.utils.logging import get_logger

logger = get_logger(__name__)


class TranscriptOutput(BaseModel):
    """LLM output before hallucination filtering."""
    executive_summary: ExecutiveSummary
    signal_cards: list[SignalCard]


def transcript_agent(state: GraphState) -> dict[str, Any]:
    """Synthesize transcripts and financials into executive summary and signal cards."""
    ticker = state["ticker"]
    logger.info("Running transcript agent for %s", ticker)

    chunks = state.get("transcript_chunks", [])
    mcp_context = state.get("mcp_context") or {}

    valid_chunk_ids = {c.get("id") for c in chunks if isinstance(c, dict)}

    chunk_context = "\n".join(
        f"Chunk ID: {c.get('id', 'unknown')}\n"
        f"Ticker: {c.get('ticker', ticker)}\n"
        f"Quarter: {c.get('quarter', 'unknown')}\n"
        f"Year: {c.get('year', 'unknown')}\n"
        f"Section: {c.get('section', 'unknown')}\n"
        f"Speaker: {c.get('speaker', 'unknown')}\n"
        f"Text: {c.get('text', '')}\n---"
        for c in chunks if isinstance(c, dict)
    )

    financials_context = ""
    if mcp_context.get("financial_statements"):
        financials_context = _json.dumps(mcp_context["financial_statements"], indent=2)
    risk_context = mcp_context.get("risk_factor_shifts", "")

    system_prompt = """You are an equity research transcript analyst. Your job is to analyze a single company's earnings call transcript and produce a structured analysis covering executive summary and key insights.

## Data Provided
You have been given:
- Earnings call transcript chunks for {ticker} (current and prior quarter context)
- SEC Financial Statements from EdgarTools MCP (use these for all numerical metrics)
- Risk Factor Changes (diff vs. prior year)

Transcript Chunks:
{chunks}

SEC Financial Statements:
{financials}

Risk Factor Changes:
{risk_shifts}

## Step 1: Produce Executive Summary
From the transcript and financial data, extract:

- beat_miss: "beat", "miss", or "inline" based on whether reported results exceeded, missed, or met expectations as discussed in the call.

- metrics: an array of exactly 4-5 financial metrics in this order:
  1. Total Revenue (or Net Sales) — use the top-line revenue figure from the financial statements
  2. Net Income — use the net income figure from the financial statements
  3. Operating Margin — calculate from operating income / total revenue, express as a percentage with % symbol (e.g. "8.2%"). For yoy_change, express the change in operating margin as percentage points with a sign (e.g. "+1.5%" means margin expanded 1.5pp, "-0.8%" means it contracted 0.8pp). Never write "bps" or "pp" — always use the % symbol.
  4-5. The two fastest-growing segments — identify which business segments had the highest YoY revenue growth rate (e.g. Advertising Revenue, Cloud Revenue). Label = segment name + " Revenue".
  For each metric provide: label, value (formatted with $ and B/M), yoy_change as a signed percentage string (e.g. "+17%", "-3.2%", "+1.5%"). yoy_change must always be a number with % — never "N/A", never empty, never qualitative text. Use actual reported figures.

- headline_takeaway: 2-3 sentence summary. First sentence states the beat/miss result with key numbers. Second sentence identifies the narrative management is pushing. Third sentence names the single biggest risk or concern raised.

- key_drivers: 3-5 specific phrases (2-4 words each) that are the most important themes — these will be bolded in the frontend (e.g. "AWS acceleration", "advertising growth", "margin expansion").

- primary_driver: one sentence identifying the single most important driver of the quarter's results, with a specific number or growth rate.

- forward_guidance: comprehensive paragraph covering ALL forward-looking commentary from the call — next quarter guidance ranges, full-year targets, segment-level guidance with specific numbers, operational targets, CapEx plans, margin commitments, qualitative outlook. Capture every forward-looking data point mentioned. Do not fabricate guidance.

## Step 2: Produce Key Insights (signal_cards)
Extract 4-6 key signals from the earnings call. Each signal needs:
- signal_type: "bullish", "bearish", or "neutral"
- headline: short descriptive title (5-8 words, max 120 chars)
- detail: 2-3 sentences with specific numbers, percentages, product names, or speaker attribution. Synthesize in your own analytical voice.
- citation: a structured source reference. Populate ALL fields from the chunk header above:
    - chunk_id: the exact Chunk ID string from the chunk header
    - ticker: the Ticker field from the chunk header
    - quarter: the Quarter field (e.g. "Q4")
    - year: the Year field (e.g. "2025")
    - section: the Section field — either "prepared_remarks" or "qa"
    - speaker: the Speaker field (normalised snake_case name)

Include a mix of bullish and bearish signals. At minimum one bearish signal — every company has risks. If analysts pushed back on something in Q&A, that is likely bearish. If management hedged or deflected, flag it.

Rules:
- Every field must reference specific numbers, percentages, product names, or named risks. No vague language like "strong performance" without data.
- metrics array must contain exactly 4-5 items.
- signal_cards must contain 4-6 items with at least one bearish signal.
- The current year is 2026.
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Generate the structured transcript analysis for {ticker}."),
    ])

    chain = prompt | get_llm(role="powerful", temperature=0.1).with_structured_output(TranscriptOutput)

    try:
        logger.info("Calling LLM for structured transcript analysis...")
        result: TranscriptOutput = chain.invoke({
            "ticker": ticker,
            "chunks": chunk_context,
            "financials": financials_context,
            "risk_shifts": risk_context,
        })
    except Exception as e:
        logger.error("Failed to generate transcript output: %s", e)
        raise

    validated_signals, suppressed = check_citations(result.signal_cards, valid_chunk_ids)
    logger.info(
        "Extracted %d validated signals (%d suppressed).",
        len(validated_signals),
        suppressed,
    )

    return {
        "executive_summary": result.executive_summary,
        "signal_cards": validated_signals,
    }
