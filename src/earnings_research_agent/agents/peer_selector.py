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
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from earnings_research_agent.mcp.edgar_tools import get_company_facts
from earnings_research_agent.state.graph_state import GraphState
from earnings_research_agent.utils.config import settings
from earnings_research_agent.utils.logging import get_logger

logger = get_logger(__name__)


class PeerSelection(BaseModel):
    """Structured output for peer ticker selection."""
    peers: list[str] = Field(
        ...,
        min_length=1,
        max_length=5,
        description="List of 3-5 peer ticker symbols.",
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

    system_prompt = """You are an expert equity research analyst.
    Given a target company and its SEC-reported sector and segment data,
    identify 3 to 5 of its closest publicly traded US competitors.

    Selection criteria:
    - Same primary sector as the target.
    - Overlap in at least 2 major revenue segments.
    - Publicly traded on a US exchange.

    Return ONLY ticker symbols. Do not include the target itself.

    SEC company profile for the target:
    {company_facts}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Identify 3-5 peers for {ticker}."),
    ])

    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=0.0,
    ).with_structured_output(PeerSelection)

    chain = prompt | llm

    try:
        result: PeerSelection = chain.invoke({
            "ticker": ticker,
            "company_facts": company_facts_context or "Not available.",
        })
    except Exception as e:
        logger.error("Failed to select peers: %s", e)
        raise

    logger.info("Selected peers for %s: %s", ticker, result.peers)
    return {"peers": result.peers}