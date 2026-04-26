"""Agentic RAG Retriever Nodes."""
from __future__ import annotations

from typing import Any

from earnings_research_agent.rag.grader import grade_chunks
from earnings_research_agent.rag.query_rewriter import rewrite_query
from earnings_research_agent.state.graph_state import GraphState
from earnings_research_agent.tools.auto_ingest import ensure_ticker_ingested
from earnings_research_agent.tools.pinecone_tool import vector_search
from earnings_research_agent.utils.config import settings
from earnings_research_agent.utils.logging import get_logger

logger = get_logger(__name__)


def _run_agentic_retrieval(ticker: str, initial_query: str, grader_mode: str) -> list[dict]:
    """Retrieve → grade → rewrite → retry (max 2 attempts)."""
    current_query = initial_query

    for attempt in range(1, settings.rag_max_retrieval_attempts + 1):
        logger.info("Retrieval attempt %d for %s (mode=%s)", attempt, ticker, grader_mode)
        chunks = vector_search(query=current_query, ticker=ticker, top_k=settings.rag_top_k)
        relevant = grade_chunks(current_query, chunks, mode=grader_mode)

        if relevant:
            logger.info("Found %d relevant chunks on attempt %d.", len(relevant), attempt)
            return relevant

        logger.warning("No relevant chunks on attempt %d.", attempt)
        if attempt < settings.rag_max_retrieval_attempts:
            current_query = rewrite_query(current_query)

    return []


def transcript_retriever(state: GraphState) -> dict[str, Any]:
    mode = state.get("grader_mode", "keyword")
    ticker = state["ticker"]
    chunks = _run_agentic_retrieval(
        ticker, f"{ticker} revenue, operating income, and forward guidance", mode
    )
    return {"transcript_chunks": chunks}


def peer_retriever(state: GraphState) -> dict[str, Any]:
    mode = state.get("grader_mode", "keyword")
    peers = state.get("peers", [])
    all_chunks: list = []

    for peer in peers:
        # Auto-ingest from HuggingFace if this peer has no data in Pinecone yet
        ensure_ticker_ingested(peer)
        chunks = _run_agentic_retrieval(
            peer, f"{peer} core business segments, revenue growth, and market trends", mode
        )
        all_chunks.extend(chunks)

    return {"peer_chunks": all_chunks}
