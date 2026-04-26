import duckdb
import re
from pathlib import Path

import os
from dotenv import load_dotenv


TICKERS = [  "WMT"]
TARGET_MIN = 400
TARGET_MAX = 600

URL = "https://huggingface.co/datasets/defeatbeta/yahoo-finance-data/resolve/main/data/stock_earning_call_transcripts.parquet"

tickers_sql = ", ".join(f"'{t}'" for t in TICKERS)

con = duckdb.connect()

# Step 1: find the most recent report_date per ticker without fetching transcripts (fast)
dates_df = con.execute(f"""
    SELECT symbol, MAX(report_date) AS report_date
    FROM '{URL}'
    WHERE symbol IN ({tickers_sql})
    GROUP BY symbol
""").df()

print("Most recent dates:")
print(dates_df)

# Step 2: fetch only those exact rows with transcripts
conditions = " OR ".join(
    f"(symbol = '{row.symbol}' AND report_date = '{row.report_date}')"
    for row in dates_df.itertuples()
)

rows = con.execute(f"""
    SELECT symbol, fiscal_year, fiscal_quarter, report_date, transcripts
    FROM '{URL}'
    WHERE {conditions}
""").df()

# ── Helpers ───────────────────────────────────────────────────────────────────

def estimate_tokens(text):
    return len(text) // 4

def normalize_speaker(name):
    return re.sub(r"\s+", "_", name.strip().lower())

def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip()) if s.strip()]

def chunk_by_tokens(text, target_min, target_max):
    sentences = split_sentences(text)
    chunks, current, current_tokens = [], [], 0
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

def parse_turns(lines):
    pattern = re.compile(r'^\[([^\]]+)\]:\s*(.*)', re.DOTALL)
    turns = []
    for line in lines:
        line = line.strip()
        m = pattern.match(line)
        if m:
            speaker, content = m.group(1).strip(), m.group(2).strip()
            if speaker and content:
                turns.append({"speaker": speaker, "content": content})
    return turns

def find_boundaries(turns, ceo_hint=None):
    ceo_name = ceo_hint
    if not ceo_name:
        for t in turns:
            m = re.search(
                r'(?:over to|introduce|welcome)\s+([A-Z][a-z]+(?: [A-Z][a-z]+)+)'
                r',?\s*(?:our\s+)?(?:President|CEO)',
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

def is_management(speaker, management_names):
    return speaker.lower() in {n.lower() for n in management_names}

def build_qa_exchanges(turns):
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

def filter_fwd_looking(paragraphs):
    filtered, removed = [], False
    for p in paragraphs:
        content = p.get('content', '') if isinstance(p, dict) else getattr(p, 'content', '')
        if not removed and 'forward-looking statements' in content.lower():
            removed = True
            continue
        filtered.append(p)
    return filtered

def paragraphs_to_turns(paragraphs):
    turns = []
    for p in paragraphs:
        speaker = p.get('speaker', '') if isinstance(p, dict) else getattr(p, 'speaker', '')
        content = p.get('content', '') if isinstance(p, dict) else getattr(p, 'content', '')
        if speaker and content:
            turns.append({"speaker": speaker, "content": content})
    return turns

# ── Process each transcript — results stored in all_transcripts ───────────────

all_transcripts = {}  # label -> list of chunk dicts

for _, row in rows.iterrows():
    ticker = row['symbol']
    quarter = f"Q{row['fiscal_quarter']}"
    year = str(row['fiscal_year'])
    date = str(row['report_date'])
    label = f"{ticker}_Q{row['fiscal_quarter']}_{row['fiscal_year']}"

    turns = paragraphs_to_turns(filter_fwd_looking(list(row["transcripts"])))
    ceo_first_idx, qa_start_idx = find_boundaries(turns)

    prepared_turns = turns[ceo_first_idx:qa_start_idx]
    qa_turns = turns[qa_start_idx:]
    management_names = {t["speaker"] for t in prepared_turns if t["speaker"].lower() != "operator"}

    chunks = []

    for turn in prepared_turns:
        if turn["speaker"].lower() == "operator":
            continue
        for sc in chunk_by_tokens(turn["content"], TARGET_MIN, TARGET_MAX):
            chunks.append({
                "text": sc,
                "ticker": ticker,
                "quarter": quarter,
                "year": year,
                "date": date,
                "section": "prepared_remarks",
                "speaker": normalize_speaker(turn["speaker"]),
            })

    for exchange in build_qa_exchanges(qa_turns):
        exchange_text = "\n\n".join(f"[{t['speaker']}]: {t['content']}" for t in exchange)
        analyst = next(
            (t["speaker"] for t in exchange if not is_management(t["speaker"], management_names)),
            "unknown",
        )
        if estimate_tokens(exchange_text) <= TARGET_MAX:
            chunks.append({
                "text": exchange_text,
                "ticker": ticker,
                "quarter": quarter,
                "year": year,
                "date": date,
                "section": "qa",
                "speaker": normalize_speaker(analyst),
            })
        else:
            sub_group, sub_analyst = [], analyst
            for t in exchange:
                if not is_management(t["speaker"], management_names) and sub_group:
                    chunks.append({
                        "text": "\n\n".join(sub_group),
                        "ticker": ticker,
                        "quarter": quarter,
                        "year": year,
                        "date": date,
                        "section": "qa",
                        "speaker": normalize_speaker(sub_analyst),
                    })
                    sub_group, sub_analyst = [], t["speaker"]
                sub_group.append(f"[{t['speaker']}]: {t['content']}")
            if sub_group:
                chunks.append({
                    "text": "\n\n".join(sub_group),
                    "ticker": ticker,
                    "quarter": quarter,
                    "year": year,
                    "date": date,
                    "section": "qa",
                    "speaker": normalize_speaker(sub_analyst),
                })

    all_transcripts[label] = chunks
    print(f"{label}: {len(chunks)} chunks ready")


load_dotenv()

import hashlib
from openai import OpenAI
from pinecone import Pinecone

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index("earnings-research-agent")

EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100


def embed_texts(texts: list[str]) -> list[list[float]]:
    response = openai_client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [r.embedding for r in response.data]


def chunk_id(chunk: dict) -> str:
    key = f"{chunk['ticker']}_{chunk['quarter']}_{chunk['year']}_{chunk['section']}_{chunk['speaker']}_{chunk['text'][:64]}"
    return hashlib.md5(key.encode()).hexdigest()


for label, chunks in all_transcripts.items():
    vectors = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        embeddings = embed_texts([c["text"] for c in batch])
        for chunk, embedding in zip(batch, embeddings):
            vectors.append({
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
            })

    for i in range(0, len(vectors), BATCH_SIZE):
        index.upsert(vectors=vectors[i : i + BATCH_SIZE])

    print(f"{label}: {len(vectors)} vectors upserted to Pinecone")