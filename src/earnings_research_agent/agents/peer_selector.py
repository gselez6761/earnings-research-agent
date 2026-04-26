"""Peer Selector Node.

Uses the Gemini model to identify relevant comparable companies for the 
target ticker. It determines the state.peers list, which dictates the 
subsequent parallel RAG fetches.
"""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from earnings_research_agent.state.graph_state import GraphState
from earnings_research_agent.utils.config import settings
from earnings_research_agent.utils.logging import get_logger

logger = get_logger(__name__)

class PeerSelection(BaseModel):
    """Temporary schema for structuring the LLM output."""
    peers: list[str] = Field(
        ..., 
        min_length=1,
        max_length=5, 
        description="List of 3-5 peer ticker symbols."
    )

def peer_selector(state: GraphState) -> dict[str, Any]:
    """Determine peer tickers for the given target company."""
    ticker = state["ticker"]
    logger.info("Selecting peers for %s...", ticker)

    # Note: To fully satisfy the architecture, an MCP ToolNode call to 
    # company_brief(ticker) would be injected here to ground the prompt 
    # in accurate SEC-reported segments before asking the LLM to choose peers.

    system_prompt = """You are an expert equity research analyst. 
    Given a target company ticker, identify 3 to 5 of its closest publicly 
    traded competitors in the US market. Return ONLY their ticker symbols.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Identify peers for the ticker: {ticker}")
    ])

    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model, 
        google_api_key=settings.gemini_api_key,
        temperature=0.0 # Strict determinism for ticker selection
    )
    
    structured_llm = llm.with_structured_output(PeerSelection)
    chain = prompt | structured_llm

    try:
        result: PeerSelection = chain.invoke({"ticker": ticker})
    except Exception as e:
        logger.error("Failed to select peers: %s", e)
        raise

    logger.info("Selected peers for %s: %s", ticker, result.peers)
    
    # Writes to state.peers
    return {"peers": result.peers}