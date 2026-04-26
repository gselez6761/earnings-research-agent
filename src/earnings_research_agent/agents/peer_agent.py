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
        return f"{ticker}: SEC data unavailable."

    # data is a dict with optional keys: financial_statements, xbrl_segments
    # (transcript branch stores financial_statements directly at top level)
    if isinstance(data, dict) and "xbrl_segments" in data:
        fin = data.get("financial_statements")
        xbrl = data.get("xbrl_segments") or {}
    else:
        fin = data
        xbrl = {}

    parts = [f"{ticker}:"]

    if xbrl:
        period = xbrl.get("period_current", "")
        parts.append(f"  XBRL Segment Revenue ({period}):")
        for seg_name, vals in xbrl.get("business_segments", {}).items():
            parts.append(
                f"    {seg_name}: current={vals['revenue_current']}  "
                f"prior={vals['revenue_prior']}  yoy={vals['yoy_growth']}"
            )
        for seg_name, vals in xbrl.get("product_segments", {}).items():
            parts.append(
                f"    {seg_name}: current={vals['revenue_current']}  "
                f"prior={vals['revenue_prior']}  yoy={vals['yoy_growth']}"
            )

    if fin:
        parts.append(f"  SEC Financial Statements:\n{json.dumps(fin, indent=4)}")

    return "\n".join(parts)


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

    # Target company data (mcp_context has financial_statements + xbrl_segments at top level)
    mcp_context = state.get("mcp_context") or {}

    # Peer data (peer_mcp_context wraps each peer's financial_statements + xbrl_segments)
    peer_mcp_context = state.get("peer_mcp_context") or {}

    # Build combined financials block: target first, then each peer
    financials_parts = [_format_financials(ticker, mcp_context)]
    for peer, data in peer_mcp_context.items():
        financials_parts.append(_format_financials(peer, data))
    financials_context = "\n\n".join(financials_parts)

    system_prompt = """You are an elite equity research analyst.
    Analyze the retrieved transcript chunks and SEC financial data for {ticker}
    and its competitors: {peers}.

    You must extract:
    1. Exactly 3 Industry Trends (Dominant, Emerging, or Persistent) spanning across the companies.
    2. A Competitive Landscape Table mapping shared business segments across ALL companies including {ticker}.

    ## Revenue Data Rules (CRITICAL)
    The financial data below contains XBRL Segment Revenue pulled directly from 10-K filings.
    These are authoritative numbers — use them verbatim for revenue_current, revenue_prior, yoy_growth.
    - revenue_current / revenue_prior: formatted dollar string e.g. "$128.7B".
    - yoy_growth: signed percentage string e.g. "+19.6%".
    - NEVER write: "Not available", "Not specified", "N/A", "Not available from transcript",
      "Not disclosed", "No Data", or any placeholder text in revenue fields.

    ## Confidence Levels — STRICT RULES
    - 'exact': Segment names matched directly AND revenue figures are available.
    - 'inferred': Segment names mapped approximately AND revenue figures are available.
    - 'missing': Revenue data is absent for this company in this segment — use this whenever
      you cannot find an actual dollar figure. A cell with confidence 'missing' will be
      hidden from the table. NEVER use 'exact' or 'inferred' when revenue is unknown.

    The rule is simple: if you have the revenue number, use exact or inferred.
    If you do not have the revenue number, use missing — do not invent a segment name
    with blank revenue. A missing cell is better than a misleading one.

    Transcript Context (all companies):
    {chunk_context}

    Financial Data — XBRL segments + SEC statements for ALL companies including {ticker}:
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