# Earnings Research Agent

A LangGraph multi-agent system that takes a stock ticker, runs parallel agentic RAG over earnings call transcripts and SEC filings, and produces a structured financial report with a human-in-the-loop review step.

**Output:** Executive Summary · Signal Cards (bullish/bearish/neutral with citations) · Industry Trends · Competitive Landscape Table

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10+ | 3.11 recommended |
| Pinecone account | — | Free tier works; needs one index |
| Gemini API key | — | Google AI Studio (free quota available) |
| OpenAI API key | — | Embeddings only (`text-embedding-3-small`) |
| SEC EDGAR identity | — | Your name + email, required by SEC fair-access rules |

---

## 1. Clone and install

```bash
git clone https://github.com/gselez6761/earnings-research-agent.git
cd earnings-research-agent
pip install -e . && pip install -r requirements.txt
```

Or use Make:

```bash
make install
```

---

## 2. Configure environment variables

Copy the template and fill in your keys:

```bash
cp .env.example .env   # then edit .env
```

`.env` must contain:

```dotenv
# --- Required ---

# Google Gemini (primary LLM for all agents)
GEMINI_API_KEY=your-gemini-api-key

# OpenAI (embeddings only — text-embedding-3-small)
OPENAI_API_KEY=your-openai-api-key

# Pinecone (transcript vector store)
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX=dbmain           # must match the index you create in step 3

# SEC EDGAR identity — required by SEC for programmatic access
# Format: "First Last email@example.com"  (include the quotes in the value if your shell needs it)
EDGAR_IDENTITY=Your Name you@email.com

# --- Optional ---

# Override the Gemini model (default: gemini-2.5-flash)
# GEMINI_MODEL=gemini-2.0-flash

# Override the embedding model (default: text-embedding-3-small)
# OPENAI_EMBED_MODEL=text-embedding-3-small

# RAG tuning
# RAG_TOP_K=10
# RAG_MAX_RETRIEVAL_ATTEMPTS=2

# Feedback store location (default: data/feedback_store/feedback.jsonl)
# FEEDBACK_STORE_PATH=data/feedback_store/feedback.jsonl

# Production checkpointer — leave empty to use in-memory (dev only)
# POSTGRES_URL=postgresql://user:pass@localhost:5432/dbname
```

> **EDGAR_IDENTITY** is validated at startup. If the format is wrong the process exits immediately with a clear error message before making any MCP calls.

---

## 3. Create the Pinecone index

In the [Pinecone console](https://app.pinecone.io):

1. Create an index named `dbmain` (or whatever you put in `PINECONE_INDEX`).
2. Dimensions: **1536** (matches `text-embedding-3-small`).
3. Metric: **cosine**.

The free Starter plan supports one index with up to 100k vectors, which is enough for 5-10 tickers.

---

## 4. Ingest transcripts into Pinecone

This is a one-time step (re-run when you want to add new tickers).

```bash
# Default: AMZN, GOOG, META
make ingest

# Custom tickers
make ingest TICKERS="AMZN MSFT AAPL GOOG META"

# Or directly
python scripts/ingest_transcripts.py --tickers AMZN MSFT
```

The script pulls the most recent earnings transcript per ticker from a public HuggingFace dataset, chunks by speaker turn (400–600 tokens), embeds with OpenAI, and upserts to Pinecone. Takes ~1–2 minutes per ticker on a cold run.

---

## 5. Run the agent

```bash
# Interactive (prompts for approve / edit / reject)
make run TICKER=AMZN

# Or directly with a custom thread ID
python scripts/run_agent.py --ticker AMZN --thread-id session_001
```

### What happens

```
peer_selector          →  picks 3-5 peers via Gemini + EdgarTools MCP
       ↓ (parallel)
transcript_retriever   →  agentic RAG loop (retrieve → grade → rewrite, max 2x)
transcript_mcp_node    →  fetches SEC financials + risk factor diffs for target
transcript_agent       →  synthesises Executive Summary, Signal Cards, Temporal Deltas

peer_retriever         →  agentic RAG loop for each peer
peer_mcp_node          →  fetches SEC financials for all peers (concurrent)
peer_analysis_node     →  synthesises Industry Trends + Competitive Landscape Table
       ↓ (merge)
merge_node             →  combines branches; suppresses low-confidence table rows
human_review_node      →  PAUSES — prints report, waits for your input
log_feedback_node      →  appends action + report snapshot to feedback.jsonl
       ↓ (conditional)
refine_node            →  (edit path only) applies analyst edits via LLM rewrite
→ loops back to human_review_node until approve or reject
```

### Human review options

When the graph pauses:

```
Enter action (approve/edit/reject) [approve]:
```

- **approve** — accepts the report, writes to feedback store, exits.
- **reject** — discards the report, writes to feedback store, exits.
- **edit** — accepts a free-text note; the refine node applies the edit and presents the updated report for re-review.

---

## 6. Export feedback for analysis

```bash
make export
# Writes to data/feedback_store/export.csv
```

Columns: `thread_id, ticker, timestamp, action, quality_rating, free_text_note, num_signal_overrides, num_table_edits`

---

## Project structure

```
src/earnings_research_agent/
├── state/
│   ├── graph_state.py      # GraphState TypedDict — all inter-node data
│   └── schemas.py          # Pydantic models: SignalCard, FinalReport, HumanFeedback, …
├── graph/
│   ├── graph.py            # StateGraph builder + node/edge wiring
│   ├── edges.py            # route_on_feedback conditional routing
│   └── checkpointer.py     # MemorySaver (dev) vs PostgresSaver (prod) factory
├── agents/
│   ├── peer_selector.py    # Gemini + MCP → state.peers
│   ├── transcript_agent.py # Gemini synthesis → ExecutiveSummary + SignalCards
│   ├── peer_agent.py       # Gemini synthesis → IndustryTrends + CompetitiveTable
│   ├── merge_node.py       # deterministic merge, Fix 3 confidence filter
│   └── refine_node.py      # LLM applies human edits to FinalReport
├── rag/
│   ├── retriever.py        # agentic loop: retrieve → grade → rewrite → retry
│   ├── grader.py           # LLM relevance scorer (yes/no per chunk)
│   ├── query_rewriter.py   # rewrites failed queries for better Pinecone recall
│   └── hallucination_checker.py  # Fix 1: suppresses signals with invented chunk_ids
├── mcp/
│   ├── edgar_tools.py      # MCP client + transcript_mcp_node + peer_mcp_node
│   └── auth.py             # EDGAR_IDENTITY format validation
├── feedback/
│   ├── human_review.py     # interrupt() pause node
│   ├── log_feedback.py     # JSONL append node
│   └── feedback_store.py   # read interface: iter_entries, quality_summary
├── tools/
│   ├── pinecone_tool.py    # vector_search (embed + query)
│   └── edgar_mcp_client.py # @tool wrappers for optional ToolNode usage
└── utils/
    ├── config.py           # pydantic-settings Settings singleton
    ├── exceptions.py       # custom exception hierarchy
    └── logging.py          # structured logger factory

scripts/
├── ingest_transcripts.py   # one-time Pinecone population pipeline
├── run_agent.py            # local CLI entry point
└── export_feedback.py      # export feedback.jsonl → CSV

configs/
├── gemini.yaml             # LLM temperature reference
├── pinecone.yaml           # index / embedding settings reference
├── rag.yaml                # top_k and loop limit reference
└── edgar_mcp.yaml          # MCP tool / node mapping reference
```

---

## Troubleshooting

**`ValueError: EDGAR_IDENTITY must be 'First Last email@example.com'`**
Set `EDGAR_IDENTITY` in `.env` to your full name and email, e.g. `Jane Smith jane@example.com`.

**`EdgarMCPError: MCP tool 'get_financial_statements' failed`**
The EdgarTools MCP server spawns a subprocess on each call. Make sure `edgartools[ai]` is installed (`pip install "edgartools[ai]"`). If it times out, the SEC may be rate-limiting — wait a minute and retry.

**`RetrievalError` / empty `transcript_chunks`**
The Pinecone index has no vectors for that ticker. Run `make ingest TICKERS="YOUR_TICKER"` first.

**Graph hangs at `human_review_node`**
This is expected — the graph is waiting for your input at the CLI prompt. Type `approve`, `edit`, or `reject` and press Enter.

**`ModuleNotFoundError: No module named 'langgraph'`**
Run `pip install -r requirements.txt` — langgraph is not on PyPI under that name in older pip caches. Ensure you're using Python 3.10+.
