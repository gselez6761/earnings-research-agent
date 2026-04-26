"""Pinecone vector search tool.

Embeds a query with OpenAI text-embedding-3-small, then queries the Pinecone
index filtered by ticker. Returns a list of chunk dicts compatible with the
shape expected by transcript_retriever and peer_retriever.
"""
from __future__ import annotations

from typing import Any

from openai import OpenAI
from pinecone import Pinecone

from earnings_research_agent.utils.config import settings
from earnings_research_agent.utils.logging import get_logger

logger = get_logger(__name__)

_oai_client: OpenAI | None = None
_pc_index = None


def _get_clients():
    global _oai_client, _pc_index
    if _oai_client is None:
        _oai_client = OpenAI(api_key=settings.openai_api_key)
    if _pc_index is None:
        pc = Pinecone(api_key=settings.pinecone_api_key)
        _pc_index = pc.Index(settings.pinecone_index)
    return _oai_client, _pc_index


def vector_search(query: str, ticker: str, top_k: int = 10) -> list[dict[str, Any]]:
    """Embed query and run a filtered Pinecone search for the given ticker.

    Args:
        query:  Natural language search string.
        ticker: Uppercase ticker symbol used as a metadata filter.
        top_k:  Maximum number of results to return.

    Returns:
        List of chunk dicts with keys: id, score, text, ticker, speaker,
        quarter, year, section.
    """
    oai, index = _get_clients()

    embed_resp = oai.embeddings.create(model=settings.openai_embed_model, input=query)
    vector = embed_resp.data[0].embedding

    results = index.query(
        vector=vector,
        top_k=top_k,
        filter={"ticker": {"$eq": ticker}},
        include_metadata=True,
    )

    chunks = []
    for match in results.matches:
        meta = match.metadata or {}
        chunks.append({
            "id": match.id,
            "score": match.score,
            "text": meta.get("text", ""),
            "ticker": meta.get("ticker", ticker),
            "speaker": meta.get("speaker", "unknown"),
            "quarter": meta.get("quarter", ""),
            "year": meta.get("year", ""),
            "section": meta.get("section", ""),
        })

    logger.debug("Pinecone returned %d chunks for ticker=%s query='%s'", len(chunks), ticker, query)
    return chunks
