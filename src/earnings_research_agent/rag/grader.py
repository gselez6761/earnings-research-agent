"""RAG Relevance Grader.

Uses fast keyword matching instead of per-chunk LLM calls.
Pinecone already handles semantic relevance; this layer filters out
chunks that contain none of the query's key terms.
"""
from __future__ import annotations

from earnings_research_agent.utils.logging import get_logger

logger = get_logger(__name__)

# Financial boilerplate that signals an off-topic chunk
_BOILERPLATE = {
    "safe harbor", "forward-looking statements", "risk factors",
    "legal proceedings", "signature", "exhibit", "pursuant to",
}

_FINANCIAL_TERMS = {
    "revenue", "income", "margin", "growth", "segment", "quarter",
    "guidance", "eps", "earnings", "operating", "profit", "cash",
    "capex", "cloud", "aws", "azure", "advertising", "retail",
}


def grade_chunk(query: str, chunk_text: str) -> bool:
    """Return True if the chunk is relevant to the query.

    Checks that at least one query keyword appears in the chunk and that
    the chunk is not pure boilerplate. O(n) string ops — no LLM call.
    """
    text_lower = chunk_text.lower()

    # Reject obvious boilerplate if it has none of the financial terms
    if any(b in text_lower for b in _BOILERPLATE):
        if not any(t in text_lower for t in _FINANCIAL_TERMS):
            logger.debug("Chunk rejected as boilerplate.")
            return False

    # Accept if any query token (length > 3) appears in the chunk
    query_tokens = [t.lower() for t in query.split() if len(t) > 3]
    if any(tok in text_lower for tok in query_tokens):
        return True

    # Accept if the chunk contains common financial terms
    return any(t in text_lower for t in _FINANCIAL_TERMS)
