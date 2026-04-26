"""Peer Analysis Node.

Analyzes retrieved chunks from peer transcripts alongside MCP financial data
to generate macro industry trends and build the competitive landscape table.
"""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from earnings_research_agent.state.graph_state import GraphState
from earnings_research_agent.state.schemas import IndustryTrend, CompetitiveRow
from earnings_research_agent.utils.config import settings
from earnings_research_agent.utils.logging import get_logger

logger = get_logger(__name__)

class PeerOutput(BaseModel):
    """Temporary schema for the LLM output targeting peer analysis."""
    industry_trends: list[IndustryTrend]
    competitive_table: list[CompetitiveRow]

def peer_analysis_node(state: GraphState) -> dict[str, Any]:
    """Synthesize peer data into macro trends and a competitive matrix."""
    ticker = state["ticker"]
    peers = state.get("peers", [])
    logger.info("Running peer analysis for %s against peers: %s", ticker, peers)

    # 1. Gather Context
    # In full runtime, this extracts from state.retrieved_chunks (graded) 
    # and MCP outputs (financial_statements for target + all peers).
    chunks = state.get("retrieved_chunks", [])
    
    chunk_context = "\n".join(
        f"Ticker: {c.get('ticker', 'unknown')} | Text: {c.get('text', '')}"
        for c in chunks if isinstance(c, dict)
    )

    # 2. Build the LLM Chain
    system_prompt = """You are an elite equity research analyst.
    Analyze the retrieved transcript chunks and financial data for {ticker} 
    and its competitors: {peers}.

    You must extract:
    1. Exactly 3 Industry Trends (Dominant, Emerging, or Persistent) spanning across the peers.
    2. A Competitive Landscape Table mapping shared business segments.

    GUARDRAIL (Fix 3): For the competitive table, you must assign a confidence level to each segment cell:
    - 'exact': Segment names matched directly.
    - 'inferred': You mapped similar but distinct names (e.g., 'AWS' to 'Cloud Services').
    - 'missing': The data is absent.
    
    Transcript Context:
    {context}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Generate the structured peer analysis.")
    ])

    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model, 
        google_api_key=settings.gemini_api_key, 
        temperature=0.1
    )
    
    structured_llm = llm.with_structured_output(PeerOutput)
    chain = prompt | structured_llm

    # 3. Invoke Model
    try:
        logger.info("Calling Gemini API for competitive landscape synthesis...")
        result: PeerOutput = chain.invoke({
            "ticker": ticker, 
            "peers": ", ".join(peers),
            "context": chunk_context
        })
    except Exception as e:
        logger.error("Failed to generate peer output: %s", e)
        raise

    logger.info(
        "Extracted %d industry trends and mapped %d competitive rows.", 
        len(result.industry_trends), 
        len(result.competitive_table)
    )

    # Return payload to write to GraphState. The merge_node will later read 
    # this and filter out the 'missing' confidence rows.
    return {
        "industry_trends": result.industry_trends,
        "competitive_table": result.competitive_table
    }