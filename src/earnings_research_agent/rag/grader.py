"""RAG Relevance Grader.

Two modes, toggled via the frontend:
  keyword  — O(n) string matching, zero LLM calls (fast, cheap, dumb)
  llm      — single batched LLM call for all chunks (semantic, one API call per retrieval attempt)
"""
from __future__ import annotations

import json

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from earnings_research_agent.utils.llm import get_llm
from earnings_research_agent.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Keyword filter
# ---------------------------------------------------------------------------

_BOILERPLATE = {
    "safe harbor", "forward-looking statements", "legal proceedings",
    "signature", "exhibit", "pursuant to",
}
_FINANCIAL_TERMS = {
    "revenue", "income", "margin", "growth", "segment", "quarter",
    "guidance", "eps", "earnings", "operating", "profit", "cash",
    "capex", "cloud", "aws", "azure", "advertising", "retail",
}


def _grade_keyword(query: str, chunk_text: str) -> bool:
    text = chunk_text.lower()
    if any(b in text for b in _BOILERPLATE):
        if not any(t in text for t in _FINANCIAL_TERMS):
            return False
    tokens = [t.lower() for t in query.split() if len(t) > 3]
    return any(tok in text for tok in tokens) or any(t in text for t in _FINANCIAL_TERMS)


# ---------------------------------------------------------------------------
# Batched LLM grader
# ---------------------------------------------------------------------------

class _GradeResult(BaseModel):
    relevant_indices: list[int]


def _grade_llm(query: str, chunks: list[str]) -> list[bool]:
    """Grade all chunks in a single LLM call. Returns a bool list parallel to chunks."""
    if not chunks:
        return []

    numbered = "\n\n".join(f"[{i}] {text[:800]}" for i, text in enumerate(chunks))

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a relevance grader for an equity research RAG system. "
         "Given a query and numbered document chunks, return the indices of chunks "
         "that contain information relevant to the query. "
         "Reject generic boilerplate, legal disclaimers, and off-topic content."),
        ("human", "Query: {query}\n\nChunks:\n{chunks}\n\n"
                  "Return only the indices of relevant chunks as a JSON list in relevant_indices."),
    ])

    chain = prompt | get_llm(role="fast", temperature=0.0).with_structured_output(_GradeResult)

    try:
        result: _GradeResult = chain.invoke({"query": query, "chunks": numbered})
        relevant_set = set(result.relevant_indices)
        logger.debug("LLM grader: %d/%d chunks relevant", len(relevant_set), len(chunks))
        return [i in relevant_set for i in range(len(chunks))]
    except Exception as e:
        logger.error("LLM batch grader failed (%s), falling back to keyword.", e)
        return [_grade_keyword(query, c) for c in chunks]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def grade_chunks(query: str, chunks: list[dict], mode: str = "keyword") -> list[dict]:
    """Return only relevant chunks. mode is 'keyword' or 'llm'."""
    texts = [c.get("text", "") for c in chunks]

    if mode == "llm":
        flags = _grade_llm(query, texts)
    else:
        flags = [_grade_keyword(query, t) for t in texts]

    return [c for c, keep in zip(chunks, flags) if keep]
