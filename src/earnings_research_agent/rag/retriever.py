"""Agentic RAG Retriever Nodes.

Executes the self-correcting retrieval loop. It fetches chunks from Pinecone,
grades them for relevance, and rewrites the query if the chunks are irrelevant,
up to a maximum of 2 attempts.
"""
from __future__ import annotations

from typing import Any

from earnings_research_agent.rag.grader import grade_chunk
from earnings_research_agent.rag.query_rewriter import rewrite_query
from earnings_research_agent.state.graph_state import GraphState
from earnings_research_agent.tools.pinecone_tool import vector_search
from earnings_research_agent.utils.config import settings
from earnings_research_agent.utils.logging import get_logger

logger = get_logger(__name__)

def _run_agentic_retrieval(ticker: str, initial_query: str) -> dict[str, Any]:
    """Core agentic loop: retrieve -> grade -> rewrite -> retry (max 2 loops)."""
    current_query = initial_query
    attempts = 0
    max_attempts = settings.rag_max_retrieval_attempts  # Set to 2 via config
    
    final_chunks = []
    rewritten = None

    while attempts < max_attempts:
        attempts += 1
        logger.info("Retrieval attempt %d for %s with query: '%s'", attempts, ticker, current_query)
        
        # 1. Query Pinecone (requires current + prior quarter filter based on architecture)
        # Using a generic vector_search tool assumption here
        chunks = vector_search(query=current_query, ticker=ticker, top_k=settings.rag_top_k)
        
        # 2. Grade chunks for relevance
        relevant_chunks = []
        for chunk in chunks:
            if grade_chunk(query=current_query, chunk_text=chunk.get("text", "")):
                relevant_chunks.append(chunk)

        # 3. Assess results
        if relevant_chunks:
            logger.info("Found %d relevant chunks on attempt %d.", len(relevant_chunks), attempts)
            final_chunks = relevant_chunks
            break  # Success, exit loop
        else:
            logger.warning("No relevant chunks found on attempt %d.", attempts)
            if attempts < max_attempts:
                # Rewrite query for the next loop
                current_query = rewrite_query(current_query)
                rewritten = current_query

    return {
        "retrieved_chunks": final_chunks,
        "retrieval_attempts": attempts,
        "rewritten_query": rewritten
    }

def transcript_retriever(state: GraphState) -> dict[str, Any]:
    """RAG node for the main transcript branch."""
    ticker = state["ticker"]
    initial_query = f"{ticker} revenue, operating income, and forward guidance"
    result = _run_agentic_retrieval(ticker, initial_query)
    return {"transcript_chunks": result["retrieved_chunks"]}


def peer_retriever(state: GraphState) -> dict[str, Any]:
    """RAG node for the peer branch. Runs the agentic loop for each peer ticker."""
    peers = state.get("peers", [])
    all_peer_chunks: list = []
    max_attempts_seen = 0

    for peer in peers:
        initial_query = f"{peer} core business segments, revenue growth, and market trends"
        result = _run_agentic_retrieval(peer, initial_query)
        all_peer_chunks.extend(result["retrieved_chunks"])
        max_attempts_seen = max(max_attempts_seen, result["retrieval_attempts"])

    return {"peer_chunks": all_peer_chunks}