"""Ingest earnings call transcripts from HuggingFace and upsert to Pinecone.

This is a one-time / periodic data pipeline script, not part of the
live LangGraph graph. Run it to populate the Pinecone index before
starting the agent.

Usage:
    python scripts/ingest_transcripts.py
    python scripts/ingest_transcripts.py --tickers AMZN GOOG META

The script:
1. Pulls the most recent transcript per ticker from the public
   HuggingFace dataset (defeatbeta/yahoo-finance-data) via DuckDB.
2. Filters forward-looking statement disclaimers.
3. Splits at the prepared-remarks / Q&A boundary.
4. Chunks to 400-600 tokens per chunk.
5. Embeds with OpenAI text-embedding-3-small.
6. Upserts to Pinecone index 'dbmain' with full metadata.

Metadata stored per vector:
    ticker, quarter, year, date, section, speaker, text

This metadata is what the agentic RAG retriever filters on at runtime
(e.g. ticker='AMZN', quarter='Q4', year='2025').
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import duckdb
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

# Make sure the src package is importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from earnings_research_agent.utils.config import settings
from earnings_research_agent.utils.logging import get_logger

load_dotenv()
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HF_URL = (
    "https://huggingface.co/datasets/defeatbeta/yahoo-finance-data"
    "/resolve/main/data/stock_earning_call_transcripts.parquet"
)
TARGET_MIN = 400
TARGET_MAX = 600
BATCH_SIZE = 100
EMBED_MODEL = settings.openai_embed_model

# ---------------------------------------------------------------------------
# Chunking helpers (unchanged from original notebook logic)
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Estimate token count — 1 token ≈ 4 characters."""
    return len(text) // 4


def normalize_speaker(name: str) -> str:
    """Lowercase and snake_case a speaker name for consistent metadata."""
    return re.sub(r"\s+", "_", name.strip().lower())


def split_sentences(text: str) -> list[str]:
    """Split text at sentence boundaries."""
    return [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())
        if s.strip()
    ]


def chunk_by_tokens(
    text: str,
    target_min: int = TARGET_MIN,
    target_max: int = TARGET_MAX,
) -> list[str]:
    """Split a block of text into token-bounded chunks."""
    sentences = split_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        st = estimate_tokens(sentence)
        if current_tokens + st > target_max and current_tokens >= target_min:
            chunks.append(" ".join(current))
            current, current_tokens = [sentence], st
        else:
            current.append(sentence)
            current_tokens += st

    if current:
        chunks.append(" ".join(current))
    return chunks


def parse_turns(lines: list) -> list[dict]:
    """Parse [Speaker]: content lines into structured turn dicts."""
    pattern = re.compile(r"^\[([^\]]+)\]:\s*(.*)", re.DOTALL)
    turns = []
    for line in lines:
        line = line.strip()
        m = pattern.match(line)
        if m:
            speaker, content = m.group(1).strip(), m.group(2).strip()
            if speaker and content:
                turns.append({"speaker": speaker, "content": content})
    return turns


def find_boundaries(
    turns: list[dict],
    ceo_hint: str | None = None,
) -> tuple[int, int]:
    """Return (ceo_first_idx, qa_start_idx) for splitting the transcript."""
    ceo_name = ceo_hint
    if not ceo_name:
        for t in turns:
            m = re.search(
                r"(?:over to|introduce|welcome)\s+"
                r"([A-Z][a-z]+(?: [A-Z][a-z]+)+)"
                r",?\s*(?:our\s+)?(?:President|CEO)",
                t["content"],
            )
            if m:
                ceo_name = m.group(1)
                break

    ceo_first_idx = 0
    if ceo_name:
        for i, t in enumerate(turns):
            if t["speaker"].lower() == ceo_name.lower():
                ceo_first_idx = i
                break

    qa_start_idx = len(turns)
    for i, t in enumerate(turns):
        if t["speaker"].lower() == "operator":
            cl = t["content"].lower()
            if "first question" in cl or ("next question" in cl and i > 0):
                qa_start_idx = i
                break

    return ceo_first_idx, qa_start_idx


def is_management(speaker: str, management_names: set[str]) -> bool:
    """Return True if the speaker is on the management team."""
    return speaker.lower() in {n.lower() for n in management_names}


def build_qa_exchanges(turns: list[dict]) -> list[list[dict]]:
    """Group Q&A turns into (analyst question + management response) exchanges."""
    exchanges: list[list[dict]] = []
    current: list[dict] = []
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


def filter_fwd_looking(paragraphs: list) -> list:
    """Remove the forward-looking statements disclaimer paragraph."""
    filtered: list = []
    removed = False
    for p in paragraphs:
        content = (
            p.get("content", "") if isinstance(p, dict) else getattr(p, "content", "")
        )
        if not removed and "forward-looking statements" in content.lower():
            removed = True
            continue
        filtered.append(p)
    return filtered


def paragraphs_to_turns(paragraphs: list) -> list[dict]:
    """Convert paragraph objects (dict or dataclass) to turn dicts."""
    turns = []
    for p in paragraphs:
        speaker = (
            p.get("speaker", "") if isinstance(p, dict) else getattr(p, "speaker", "")
        )
        content = (
            p.get("content", "") if isinstance(p, dict) else getattr(p, "content", "")
        )
        if speaker and content:
            turns.append({"speaker": speaker, "content": content})
    return turns


# ---------------------------------------------------------------------------
# Chunking pipeline
# ---------------------------------------------------------------------------


def build_chunks(row) -> list[dict]:
    """Build all chunks for a single transcript row."""
    ticker = row["symbol"]
    quarter = f"Q{row['fiscal_quarter']}"
    year = str(row["fiscal_year"])
    date = str(row["report_date"])

    turns = paragraphs_to_turns(filter_fwd_looking(list(row["transcripts"])))
    ceo_first_idx, qa_start_idx = find_boundaries(turns)

    prepared_turns = turns[ceo_first_idx:qa_start_idx]
    qa_turns = turns[qa_start_idx:]
    management_names = {
        t["speaker"]
        for t in prepared_turns
        if t["speaker"].lower() != "operator"
    }

    chunks: list[dict] = []

    # Prepared remarks — one chunk per speaker turn, split to token bounds
    for turn in prepared_turns:
        if turn["speaker"].lower() == "operator":
            continue
        for sc in chunk_by_tokens(turn["content"]):
            chunks.append(
                {
                    "text": sc,
                    "ticker": ticker,
                    "quarter": quarter,
                    "year": year,
                    "date": date,
                    "section": "prepared_remarks",
                    "speaker": normalize_speaker(turn["speaker"]),
                }
            )

    # Q&A — one chunk per exchange (analyst question + management response)
    for exchange in build_qa_exchanges(qa_turns):
        exchange_text = "\n\n".join(
            f"[{t['speaker']}]: {t['content']}" for t in exchange
        )
        analyst = next(
            (
                t["speaker"]
                for t in exchange
                if not is_management(t["speaker"], management_names)
            ),
            "unknown",
        )

        if estimate_tokens(exchange_text) <= TARGET_MAX:
            chunks.append(
                {
                    "text": exchange_text,
                    "ticker": ticker,
                    "quarter": quarter,
                    "year": year,
                    "date": date,
                    "section": "qa",
                    "speaker": normalize_speaker(analyst),
                }
            )
        else:
            # Exchange too long — split per speaker turn within the exchange
            sub_group: list[str] = []
            sub_analyst = analyst
            for t in exchange:
                if not is_management(t["speaker"], management_names) and sub_group:
                    chunks.append(
                        {
                            "text": "\n\n".join(sub_group),
                            "ticker": ticker,
                            "quarter": quarter,
                            "year": year,
                            "date": date,
                            "section": "qa",
                            "speaker": normalize_speaker(sub_analyst),
                        }
                    )
                    sub_group, sub_analyst = [], t["speaker"]
                sub_group.append(f"[{t['speaker']}]: {t['content']}")
            if sub_group:
                chunks.append(
                    {
                        "text": "\n\n".join(sub_group),
                        "ticker": ticker,
                        "quarter": quarter,
                        "year": year,
                        "date": date,
                        "section": "qa",
                        "speaker": normalize_speaker(sub_analyst),
                    }
                )

    return chunks


# ---------------------------------------------------------------------------
# Embedding and upsert
# ---------------------------------------------------------------------------


def chunk_id(chunk: dict) -> str:
    """Deterministic MD5 vector ID — same chunk always gets the same ID."""
    key = (
        f"{chunk['ticker']}_{chunk['quarter']}_{chunk['year']}_"
        f"{chunk['section']}_{chunk['speaker']}_{chunk['text'][:64]}"
    )
    return hashlib.md5(key.encode()).hexdigest()


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using OpenAI text-embedding-3-small."""
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [r.embedding for r in response.data]


def upsert_chunks(
    chunks: list[dict],
    openai_client: OpenAI,
    index,
    label: str,
) -> None:
    """Embed and upsert chunks to Pinecone in BATCH_SIZE batches."""
    vectors = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        embeddings = embed_texts(openai_client, [c["text"] for c in batch])
        for chunk, embedding in zip(batch, embeddings):
            vectors.append(
                {
                    "id": chunk_id(chunk),
                    "values": embedding,
                    "metadata": {
                        "text": chunk["text"],
                        "ticker": chunk["ticker"],
                        "quarter": chunk["quarter"],
                        "year": chunk["year"],
                        "date": chunk["date"],
                        "section": chunk["section"],
                        "speaker": chunk["speaker"],
                    },
                }
            )

    for i in range(0, len(vectors), BATCH_SIZE):
        index.upsert(vectors=vectors[i : i + BATCH_SIZE])

    logger.info("%s: %d vectors upserted to Pinecone.", label, len(vectors))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(tickers: list[str]) -> None:
    """Pull transcripts, chunk, embed, and upsert for the given tickers."""
    tickers_sql = ", ".join(f"'{t}'" for t in tickers)
    con = duckdb.connect()

    # Step 1: find the most recent report_date per ticker (fast — no transcripts)
    dates_df = con.execute(
        f"""
        SELECT symbol, MAX(report_date) AS report_date
        FROM '{HF_URL}'
        WHERE symbol IN ({tickers_sql})
        GROUP BY symbol
        """
    ).df()
    logger.info("Most recent dates fetched:\n%s", dates_df)

    # Step 2: fetch only the matching rows with transcripts
    conditions = " OR ".join(
        f"(symbol = '{row.symbol}' AND report_date = '{row.report_date}')"
        for row in dates_df.itertuples()
    )
    rows = con.execute(
        f"""
        SELECT symbol, fiscal_year, fiscal_quarter, report_date, transcripts
        FROM '{HF_URL}'
        WHERE {conditions}
        """
    ).df()

    # Step 3: chunk all transcripts
    all_chunks: dict[str, list[dict]] = {}
    for _, row in rows.iterrows():
        label = f"{row['symbol']}_Q{row['fiscal_quarter']}_{row['fiscal_year']}"
        chunks = build_chunks(row)
        all_chunks[label] = chunks
        logger.info("%s: %d chunks ready.", label, len(chunks))

    # Step 4: embed and upsert
    openai_client = OpenAI(api_key=settings.openai_api_key)
    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index(settings.pinecone_index)

    for label, chunks in all_chunks.items():
        upsert_chunks(chunks, openai_client, index, label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest earnings transcripts and upsert to Pinecone."
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["AMZN", "GOOG", "META"],
        help="Ticker symbols to ingest (default: AMZN GOOG META).",
    )
    args = parser.parse_args()
    main(args.tickers)