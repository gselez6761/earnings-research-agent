"""Auto-ingest: check if a ticker exists in Pinecone; fetch and ingest if not.

Called by peer_retriever before retrieval so any peer selected by the agent
that hasn't been ingested yet is automatically fetched from HuggingFace,
chunked, embedded, and upserted before the RAG query runs.
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

from openai import OpenAI
from pinecone import Pinecone

from earnings_research_agent.utils.config import settings
from earnings_research_agent.utils.logging import get_logger

logger = get_logger(__name__)

HF_URL = (
    "https://huggingface.co/datasets/defeatbeta/yahoo-finance-data"
    "/resolve/main/data/stock_earning_call_transcripts.parquet"
)
TARGET_MIN = 400
TARGET_MAX = 600
BATCH_SIZE = 100


# ---------------------------------------------------------------------------
# Pinecone presence check
# ---------------------------------------------------------------------------

def ticker_in_pinecone(ticker: str) -> bool:
    """Return True if any vectors for this ticker exist in the Pinecone index."""
    from earnings_research_agent.tools.pinecone_tool import vector_search
    results = vector_search(query="revenue earnings guidance", ticker=ticker, top_k=1)
    return len(results) > 0


# ---------------------------------------------------------------------------
# Ingest helpers (extracted from scripts/ingest_transcripts.py)
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    return len(text) // 4

def _normalize_speaker(name: str) -> str:
    return re.sub(r"\s+", "_", name.strip().lower())

def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip()) if s.strip()]

def _chunk_by_tokens(text: str) -> list[str]:
    sentences = _split_sentences(text)
    chunks, current, current_tokens = [], [], 0
    for s in sentences:
        st = _estimate_tokens(s)
        if current_tokens + st > TARGET_MAX and current_tokens >= TARGET_MIN:
            chunks.append(" ".join(current))
            current, current_tokens = [s], st
        else:
            current.append(s)
            current_tokens += st
    if current:
        chunks.append(" ".join(current))
    return chunks

def _paragraphs_to_turns(paragraphs) -> list[dict]:
    turns = []
    for p in paragraphs:
        speaker = p.get("speaker", "") if isinstance(p, dict) else getattr(p, "speaker", "")
        content = p.get("content", "") if isinstance(p, dict) else getattr(p, "content", "")
        if speaker and content:
            turns.append({"speaker": speaker, "content": content})
    return turns

def _filter_fwd_looking(paragraphs) -> list:
    filtered, removed = [], False
    for p in paragraphs:
        content = p.get("content", "") if isinstance(p, dict) else getattr(p, "content", "")
        if not removed and "forward-looking statements" in content.lower():
            removed = True
            continue
        filtered.append(p)
    return filtered

def _find_boundaries(turns: list[dict]) -> tuple[int, int]:
    ceo_first_idx = 0
    qa_start_idx = len(turns)
    for i, t in enumerate(turns):
        if t["speaker"].lower() == "operator":
            cl = t["content"].lower()
            if "first question" in cl or ("next question" in cl and i > 0):
                qa_start_idx = i
                break
    return ceo_first_idx, qa_start_idx

def _build_qa_exchanges(turns: list[dict]) -> list[list[dict]]:
    exchanges, current = [], []
    for t in turns:
        if t["speaker"].lower() == "operator":
            if current:
                exchanges.append(current)
                current = []
        else:
            current.append(t)
    if current:
        exchanges.append(current)
    return exchanges

def _build_chunks(row) -> list[dict]:
    ticker = row["symbol"]
    quarter = f"Q{row['fiscal_quarter']}"
    year = str(row["fiscal_year"])
    date = str(row["report_date"])

    turns = _paragraphs_to_turns(_filter_fwd_looking(list(row["transcripts"])))
    ceo_first_idx, qa_start_idx = _find_boundaries(turns)
    prepared_turns = turns[ceo_first_idx:qa_start_idx]
    qa_turns = turns[qa_start_idx:]
    management_names = {t["speaker"].lower() for t in prepared_turns if t["speaker"].lower() != "operator"}

    chunks: list[dict] = []

    for turn in prepared_turns:
        if turn["speaker"].lower() == "operator":
            continue
        for sc in _chunk_by_tokens(turn["content"]):
            chunks.append({"text": sc, "ticker": ticker, "quarter": quarter,
                           "year": year, "date": date, "section": "prepared_remarks",
                           "speaker": _normalize_speaker(turn["speaker"])})

    for exchange in _build_qa_exchanges(qa_turns):
        text = "\n\n".join(f"[{t['speaker']}]: {t['content']}" for t in exchange)
        analyst = next((t["speaker"] for t in exchange if t["speaker"].lower() not in management_names), "unknown")
        chunks.append({"text": text, "ticker": ticker, "quarter": quarter,
                       "year": year, "date": date, "section": "qa",
                       "speaker": _normalize_speaker(analyst)})

    return chunks

_UNICODE_REPLACEMENTS = str.maketrans({
    "—": "--",   # em dash
    "–": "-",    # en dash
    "‘": "'",    # left single quote
    "’": "'",    # right single quote
    "“": '"',    # left double quote
    "”": '"',    # right double quote
    "…": "...",  # ellipsis
    " ": " ",    # non-breaking space
})


def _sanitize(text: str) -> str:
    return text.translate(_UNICODE_REPLACEMENTS).encode("latin-1", errors="replace").decode("latin-1")


def _chunk_id(chunk: dict) -> str:
    key = (f"{chunk['ticker']}_{chunk['quarter']}_{chunk['year']}_"
           f"{chunk['section']}_{chunk['speaker']}_{chunk['text'][:64]}")
    return hashlib.md5(key.encode()).hexdigest()

def _upsert_chunks(chunks: list[dict], openai_client: OpenAI, index) -> int:
    # Sanitize text in-place before embedding or storing as metadata
    for chunk in chunks:
        chunk["text"] = _sanitize(chunk["text"])

    vectors = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        response = openai_client.embeddings.create(
            model=settings.openai_embed_model,
            input=[c["text"] for c in batch],
        )
        for chunk, emb in zip(batch, response.data):
            vectors.append({
                "id": _chunk_id(chunk),
                "values": emb.embedding,
                "metadata": {k: chunk[k] for k in ("text", "ticker", "quarter", "year", "date", "section", "speaker")},
            })

    for i in range(0, len(vectors), BATCH_SIZE):
        index.upsert(vectors=vectors[i:i + BATCH_SIZE])

    return len(vectors)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def ingest_ticker(ticker: str) -> int:
    """Fetch from HuggingFace, chunk, embed, upsert. Returns number of vectors upserted."""
    import duckdb

    logger.info("Auto-ingesting %s from HuggingFace...", ticker)
    con = duckdb.connect()

    dates_df = con.execute(
        f"SELECT symbol, MAX(report_date) AS report_date FROM '{HF_URL}' "
        f"WHERE symbol = '{ticker}' GROUP BY symbol"
    ).df()

    if dates_df.empty:
        logger.warning("No transcript found for %s in HuggingFace dataset.", ticker)
        return 0

    row = dates_df.iloc[0]
    rows = con.execute(
        f"SELECT symbol, fiscal_year, fiscal_quarter, report_date, transcripts "
        f"FROM '{HF_URL}' WHERE symbol = '{ticker}' AND report_date = '{row.report_date}'"
    ).df()

    if rows.empty:
        return 0

    chunks = _build_chunks(rows.iloc[0])
    logger.info("%s: %d chunks ready for upsert.", ticker, len(chunks))

    openai_client = OpenAI(api_key=settings.openai_api_key)
    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index(settings.pinecone_index)

    count = _upsert_chunks(chunks, openai_client, index)
    logger.info("%s: auto-ingest complete (%d vectors).", ticker, count)
    return count


def ensure_ticker_ingested(ticker: str) -> None:
    """Ingest ticker if not already in Pinecone. No-op if already present."""
    if ticker_in_pinecone(ticker):
        logger.debug("%s already in Pinecone, skipping ingest.", ticker)
        return
    logger.info("%s not found in Pinecone — starting auto-ingest.", ticker)
    ingest_ticker(ticker)
