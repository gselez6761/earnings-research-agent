"""Peer Selector Node.

Uses the Gemini model to identify relevant comparable companies for the
target ticker. The LLM prompt is grounded in SEC-reported sector and
segment data fetched from the EdgarTools MCP server before any inference.
This prevents the model from hallucinating peers with incompatible business
mixes (e.g., pairing a pure-cloud company with a diversified conglomerate).
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from earnings_research_agent.mcp.edgar_tools import get_company_facts
from earnings_research_agent.state.graph_state import GraphState
from earnings_research_agent.utils.config import settings
from earnings_research_agent.utils.llm import get_llm
from earnings_research_agent.utils.logging import get_logger

logger = get_logger(__name__)


class PeerSelection(BaseModel):
    """Structured output for peer ticker selection."""
    peers: list[str] = Field(
        ...,
        description="List of peer ticker symbols.",
    )


async def peer_selector(state: GraphState) -> dict[str, Any]:
    """Determine peer tickers grounded in SEC sector and segment data."""
    ticker = state["ticker"]
    logger.info("Selecting peers for %s...", ticker)

    # Fetch SEC-reported sector and segment data to anchor the LLM prompt.
    # If the MCP call fails, fall back to pure LLM selection with a warning.
    company_facts_context = ""
    try:
        facts = await get_company_facts(ticker)
        company_facts_context = json.dumps(facts, indent=2)
        logger.info("Fetched company facts for %s from EdgarTools MCP.", ticker)
    except Exception as exc:
        logger.warning(
            "Could not fetch company facts for %s from MCP (%s). "
            "Peer selection will rely on LLM knowledge only.",
            ticker,
            exc,
        )

    n = settings.num_peers
    system_prompt = """You are an equity research assistant. Given a stock ticker, identify the {n} closest publicly traded U.S. peers by business model and industry.

Peer Selection Criteria (in order of priority):

1. Peers must compete directly with the target company in at least 2 major revenue segments. If a candidate company does not share at least 2 core business lines with the target, it is NOT a valid peer.
2. Peers should be in the same industry or sector. A restaurant chain is not a peer to a tech company. A bank is not a peer to a retailer. An oil company is not a peer to a SaaS platform.
3. Peers should be comparable in business model — if the target is a platform/marketplace, pick other platforms/marketplaces. If the target is a subscription SaaS company, pick other subscription SaaS companies.
4. Peers should be similar in scale where possible (large-cap to large-cap, mid-cap to mid-cap), but business model overlap is more important than size.

Disqualification Rules:
- Do NOT pick a company just because it is large or well-known. McDonald's is not a peer to Amazon.
- Do NOT pick conglomerates or diversified companies unless the target company is one.
- Do NOT pick companies from a completely different industry even if they share one minor business line (e.g., both having "subscription services" does not make Netflix a peer to Costco).
- Ask yourself: "Would an equity research analyst at a bank put these two companies in the same coverage universe?" If not, do not select it.

Do not include the target ticker itself. Return exactly {n} tickers.

SEC company profile for the target:
{company_facts}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Identify the {n} closest peers for {ticker}."),
    ])

    chain = prompt | get_llm(role="standard", temperature=0.0).with_structured_output(PeerSelection)

    try:
        result: PeerSelection = chain.invoke({
            "ticker": ticker,
            "company_facts": company_facts_context or "Not available.",
            "n": n,
        })
    except Exception as e:
        logger.error("Failed to select peers: %s", e)
        raise

    logger.info("Selected peers for %s: %s", ticker, result.peers)
    return {"peers": result.peers}