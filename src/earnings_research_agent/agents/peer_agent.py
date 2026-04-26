"""Peer Analysis Node.

Analyzes retrieved chunks from peer transcripts alongside MCP financial data
to generate macro industry trends and build the competitive landscape table.
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from earnings_research_agent.state.graph_state import GraphState
from earnings_research_agent.state.schemas import CompetitiveRow, IndustryTrend
from earnings_research_agent.utils.config import settings
from earnings_research_agent.utils.logging import get_logger

logger = get_logger(__name__)


class PeerOutput(BaseModel):
    """Temporary schema for the LLM output targeting peer analysis."""
    industry_trends: list[IndustryTrend]
    competitive_table: list[CompetitiveRow]


def _format_peer_financials(peer_mcp_context: dict[str, Any]) -> str:
    """Render per-peer financial statements as a readable string for the prompt."""
    if not peer_mcp_context:
        return "No SEC financial data available."
    parts = []
    for ticker, data in peer_mcp_context.items():
        if data is None:
            parts.append(f"{ticker}: MCP fetch failed — use transcript chunks only.")
        else:
            parts.append(f"{ticker}:\n{json.dumps(data, indent=2)}")
    return "\n\n".join(parts)


def peer_analysis_node(state: GraphState) -> dict[str, Any]:
    """Synthesize peer transcript chunks and MCP financials into macro trends and a competitive matrix."""
    ticker = state["ticker"]
    peers = state.get("peers", [])
    logger.info("Running peer analysis for %s against peers: %s", ticker, peers)

    # 1. Gather Context
    chunks = state.get("peer_chunks", [])
    peer_mcp_context = state.get("peer_mcp_context") or {}

    chunk_context = "\n".join(
        f"Ticker: {c.get('ticker', 'unknown')} | Text: {c.get('text', '')}"
        for c in chunks
        if isinstance(c, dict)
    )
    financials_context = _format_peer_financials(peer_mcp_context)

    # 2. Build the LLM Chain
    system_prompt = """You are an elite equity research analyst.
    Analyze the retrieved transcript chunks and SEC financial data for {ticker}
    and its competitors: {peers}.

    You must extract:
    1. Exactly 3 Industry Trends (Dominant, Emerging, or Persistent) spanning across the peers.
    2. A Competitive Landscape Table mapping shared business segments.

    GUARDRAIL (Fix 3): For each segment cell assign a confidence level:
    - 'exact': Segment names matched directly (e.g. 'AWS' == 'AWS').
    - 'inferred': Similar but distinct names mapped (e.g. 'AWS' ~ 'Cloud Services').
    - 'missing': Data absent for this company in this segment.

    Transcript Context:
    {chunk_context}

    SEC Financial Statements per Peer (EdgarTools MCP):
    {financials}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Generate the structured peer analysis."),
    ])

    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=0.1,
    )

    chain = prompt | llm.with_structured_output(PeerOutput)

    # 3. Invoke Model
    try:
        logger.info("Calling Gemini API for competitive landscape synthesis...")
        result: PeerOutput = chain.invoke({
            "ticker": ticker,
            "peers": ", ".join(peers),
            "chunk_context": chunk_context,
            "financials": financials_context,
        })
    except Exception as e:
        logger.error("Failed to generate peer output: %s", e)
        raise

    logger.info(
        "Extracted %d industry trends and mapped %d competitive rows.",
        len(result.industry_trends),
        len(result.competitive_table),
    )

    return {
        "industry_trends": result.industry_trends,
        "competitive_table": result.competitive_table,
    }