"""Peer Analysis Node.

Analyzes retrieved chunks from peer transcripts alongside MCP financial data
to generate macro industry trends and build the competitive landscape table.
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from earnings_research_agent.state.graph_state import GraphState
from earnings_research_agent.state.schemas import CompetitiveRow, IndustryTrend
from earnings_research_agent.utils.llm import get_llm
from earnings_research_agent.utils.logging import get_logger

logger = get_logger(__name__)


class PeerOutput(BaseModel):
    """Temporary schema for the LLM output targeting peer analysis."""
    industry_trends: list[IndustryTrend]
    competitive_table: list[CompetitiveRow]


def _format_financials(ticker: str, data: Any) -> str:
    if data is None:
        return f"{ticker}: SEC data unavailable — use transcript chunks only."
    return f"{ticker}:\n{json.dumps(data, indent=2)}"


def peer_analysis_node(state: GraphState) -> dict[str, Any]:
    """Synthesize transcript chunks and MCP financials for all companies into macro trends and a competitive matrix."""
    ticker = state["ticker"]
    peers = state.get("peers", [])
    logger.info("Running peer analysis for %s against peers: %s", ticker, peers)

    # Peer chunks (WMT, TGT, etc.) + target ticker chunks for segment context
    peer_chunks = state.get("peer_chunks", [])
    target_chunks = state.get("transcript_chunks", [])
    all_chunks = target_chunks + peer_chunks

    chunk_context = "\n".join(
        f"Ticker: {c.get('ticker', ticker)} | Text: {c.get('text', '')}"
        for c in all_chunks
        if isinstance(c, dict)
    )

    # Target company SEC financials (from transcript branch mcp_context)
    mcp_context = state.get("mcp_context") or {}
    target_financials = mcp_context.get("financial_statements")

    # Peer SEC financials (from peer branch peer_mcp_context)
    peer_mcp_context = state.get("peer_mcp_context") or {}

    # Build combined financials block: target first, then each peer
    financials_parts = [_format_financials(ticker, target_financials)]
    for peer, data in peer_mcp_context.items():
        financials_parts.append(_format_financials(peer, data))
    financials_context = "\n\n".join(financials_parts)

    system_prompt = """You are an elite equity research analyst.
    Analyze the retrieved transcript chunks and SEC financial data for {ticker}
    and its competitors: {peers}.

    You must extract:
    1. Exactly 3 Industry Trends (Dominant, Emerging, or Persistent) spanning across the companies.
    2. A Competitive Landscape Table mapping shared business segments across ALL companies including {ticker}.

    CRITICAL: The SEC Financial Statements below contain actual reported revenue figures for every company
    including {ticker}. You MUST use these numbers to populate revenue_current, revenue_prior, and yoy_growth
    for every cell. Do NOT write "Not specified" — use the financial data provided.

    For each segment cell assign a confidence level:
    - 'exact': Segment names matched directly (e.g. 'AWS' == 'AWS').
    - 'inferred': Similar but distinct names mapped (e.g. 'AWS' ~ 'Cloud Services').
    - 'missing': Data genuinely absent for this company in this segment.

    Transcript Context (all companies):
    {chunk_context}

    SEC Financial Statements — ALL companies including {ticker} (use these for revenue figures):
    {financials}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Generate the structured peer analysis."),
    ])

    chain = prompt | get_llm(role="powerful", temperature=0.1).with_structured_output(PeerOutput)

    try:
        logger.info("Calling LLM for competitive landscape synthesis...")
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