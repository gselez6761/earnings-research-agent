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
    # Assume chunks are stored as dicts or objects with an 'id' attribute from the retriever node
    chunks = state.get("retrieved_chunks", [])
    
    # Extract valid chunk IDs for the Fix 1 hallucination check
    valid_chunk_ids = {
        c.get("id") if isinstance(c, dict) else getattr(c, "id", None) 
        for c in chunks
    }

    # Format chunks for the LLM prompt
    chunk_context = "\n".join(
        f"Chunk ID: {c.get('id') if isinstance(c, dict) else getattr(c, 'id', 'unknown')}\n"
        f"Speaker: {c.get('speaker') if isinstance(c, dict) else getattr(c, 'speaker', 'unknown')}\n"
        f"Text: {c.get('text') if isinstance(c, dict) else getattr(c, 'text', '')}\n---"
        for c in chunks
    )

    # 2. Build the LLM Chain
    system_prompt = """You are an elite equity research analyst.
    Analyze the provided earnings call transcript chunks and financial data for {ticker}.
    
    You must extract:
    1. An Executive Summary.
    2. Signal Cards (Bullish, Bearish, or Neutral). Every signal MUST cite the exact chunk_id it came from.
    3. Temporal Deltas comparing current vs. prior quarter metrics.
    
    Transcript Chunks:
    {chunks}
    
    [Note: MCP Financial Statement Data would be injected here in a full implementation]
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
            "chunks": chunk_context
        })
    except Exception as e:
        logger.error("Failed to generate transcript output: %s", e)
        raise

    # 4. Hallucination Check (Fix 1 Guardrail)
    validated_signals: list[SignalCard] = []
    for card in result.signal_cards:
        if card.citation.chunk_id in valid_chunk_ids:
            validated_signals.append(card)
        else:
            logger.warning(
                "Suppressing signal '%s' - Invalid or hallucinated chunk_id citation: %s", 
                card.headline, 
                card.citation.chunk_id
            )

    logger.info(
        "Extracted %d validated signals and %d temporal deltas.",
        len(validated_signals),
        len(result.temporal_deltas)
    )

    # 5. Return payload to update GraphState
    return {
        "executive_summary": result.executive_summary,
        "signal_cards": validated_signals,
        "temporal_deltas": result.temporal_deltas
    }