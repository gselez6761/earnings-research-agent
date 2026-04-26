"""Transcript Analysis Agent.

This node uses the Gemini model to synthesize graded transcript chunks
and MCP financial data into an executive summary, signal cards, and
temporal deltas. It enforces Fix 1 by verifying citations.
"""
from __future__ import annotations

from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from earnings_research_agent.rag.hallucination_checker import check_citations
from earnings_research_agent.state.graph_state import GraphState
from earnings_research_agent.state.schemas import (
    ExecutiveSummary,
    SignalCard,
    TemporalDelta,
)
from earnings_research_agent.utils.config import settings
from earnings_research_agent.utils.logging import get_logger

logger = get_logger(__name__)


class TranscriptOutput(BaseModel):
    """Temporary schema for the LLM output before hallucination filtering."""
    executive_summary: ExecutiveSummary
    signal_cards: list[SignalCard]
    temporal_deltas: list[TemporalDelta]


def transcript_agent(state: GraphState) -> dict[str, Any]:
    """Synthesize transcripts and financials, enforcing chunk citations."""
    ticker = state["ticker"]
    logger.info("Running transcript agent for %s", ticker)

    # 1. Gather Context
    chunks = state.get("transcript_chunks", [])
    mcp_context = state.get("mcp_context") or {}

    # Extract valid chunk IDs for the Fix 1 hallucination check
    valid_chunk_ids = {c.get("id") for c in chunks if isinstance(c, dict)}

    chunk_context = "\n".join(
        f"Chunk ID: {c.get('id', 'unknown')}\n"
        f"Speaker: {c.get('speaker', 'unknown')}\n"
        f"Text: {c.get('text', '')}\n---"
        for c in chunks
        if isinstance(c, dict)
    )

    financials_context = ""
    if mcp_context.get("financial_statements"):
        import json as _json
        financials_context = _json.dumps(mcp_context["financial_statements"], indent=2)
    risk_context = mcp_context.get("risk_factor_shifts", "")

    # 2. Build the LLM Chain
    system_prompt = """You are an elite equity research analyst.
    Analyze the provided earnings call transcript chunks and financial data for {ticker}.

    You must extract:
    1. An Executive Summary grounded in the financial statements below.
    2. Signal Cards (Bullish, Bearish, or Neutral). Every signal MUST cite the exact chunk_id it came from.
    3. Temporal Deltas comparing current vs. prior quarter metrics using the financial statements.

    Transcript Chunks:
    {chunks}

    SEC Financial Statements (EdgarTools MCP):
    {financials}

    Risk Factor Changes (diff vs. prior year):
    {risk_shifts}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Generate the structured transcript analysis for {ticker}.")
    ])

    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=0.1  # Low temperature for factual grounding
    )
    
    # Force output to match our Pydantic schemas
    structured_llm = llm.with_structured_output(TranscriptOutput)
    chain = prompt | structured_llm

    # 3. Invoke Model
    try:
        logger.info("Calling Gemini API for structured transcript analysis...")
        result: TranscriptOutput = chain.invoke({
            "ticker": ticker,
            "chunks": chunk_context,
            "financials": financials_context,
            "risk_shifts": risk_context,
        })
    except Exception as e:
        logger.error("Failed to generate transcript output: %s", e)
        raise

    # 4. Hallucination Check (Fix 1 Guardrail)
    validated_signals, suppressed = check_citations(result.signal_cards, valid_chunk_ids)
    logger.info(
        "Extracted %d validated signals (%d suppressed) and %d temporal deltas.",
        len(validated_signals),
        suppressed,
        len(result.temporal_deltas),
    )

    # 5. Return payload to update GraphState
    return {
        "executive_summary": result.executive_summary,
        "signal_cards": validated_signals,
        "temporal_deltas": result.temporal_deltas
    }