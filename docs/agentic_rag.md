# Agentic RAG

## Why not vanilla RAG?

Static top-k retrieval (fire one query, take 10 chunks, pass them in) has a predictable failure mode for earnings transcripts: a query for "AWS revenue guidance" might return chunks from the prepared remarks but miss the CFO clarification buried in Q&A. There is no feedback loop — if the chunks are off-topic, the LLM hallucinates.

Agentic RAG adds a self-correcting loop:

```
retrieve (Pinecone top-k)
    │
    ▼
grade each chunk for relevance (LLM binary scorer)
    │
    ├── relevant chunks found → pass to synthesis agent
    │
    └── no relevant chunks, attempts < max
           │
           ▼
        rewrite query (LLM)
           │
           └── retrieve again (loop, max 2 attempts)
```

If the loop exhausts attempts with no relevant chunks, the empty set is passed downstream. The synthesis agent handles this gracefully — it produces lower-confidence output rather than crashing.

---

## Implementation

### `rag/retriever.py` — `_run_agentic_retrieval(ticker, initial_query)`

The core loop. Shared by both branches via a private helper; the two node functions just set the initial query and write to the correct state key.

**`transcript_retriever`** — initial query: `"{ticker} revenue, operating income, and forward guidance"`
Writes to `state.transcript_chunks`.

**`peer_retriever`** — runs the loop independently for each peer; initial query: `"{peer} core business segments, revenue growth, and market trends"`. Writes to `state.peer_chunks`.

```python
while attempts < max_attempts:
    chunks = vector_search(query=current_query, ticker=ticker, top_k=settings.rag_top_k)
    relevant = [c for c in chunks if grade_chunk(query, c["text"])]
    if relevant:
        return relevant          # success
    current_query = rewrite_query(current_query)
    attempts += 1
return []                        # exhausted — pass empty to synthesis
```

Configured via `RAG_TOP_K` (default 10) and `RAG_MAX_RETRIEVAL_ATTEMPTS` (default 2).

---

### `rag/grader.py` — `grade_chunk(query, chunk_text) → bool`

Calls Gemini with `temperature=0.0` and structured output `ChunkGrade(score: "yes"|"no", reasoning: str)`.

**Fail-open**: if the grader LLM call fails (network error, quota), the chunk is kept rather than dropped. Losing data is worse than including a slightly off-topic chunk.

---

### `rag/query_rewriter.py` — `rewrite_query(original_query) → str`

Calls Gemini with `temperature=0.2` and structured output `RewrittenQuery(query: str)`.

Strategies injected via the system prompt:
- Expand financial acronyms (e.g. "CapEx" → "capital expenditure")
- Focus on core segments and forward-looking language
- Remove conversational filler

Falls back to the original query if the rewrite LLM call fails.

---

### `rag/hallucination_checker.py` — `check_citations(signal_cards, valid_chunk_ids)`

Run after synthesis, not during retrieval. The grader filters *chunks*; the hallucination checker filters *generated claims*.

Every `SignalCard` must include a `citation.chunk_id` — the MD5 Pinecone vector ID of the source chunk. After the synthesis LLM produces signal cards, this function compares each `chunk_id` against the set of IDs that were actually retrieved. Any card referencing an invented ID is suppressed and logged.

Returns `(validated_cards, suppressed_count)`. A non-zero `suppressed_count` is logged at WARNING level so it shows up in the audit trail.

---

## Pinecone metadata filters

`vector_search` in `tools/pinecone_tool.py` passes `filter={"ticker": {"$eq": ticker}}` on every query. This means the two parallel branches never cross-contaminate — peer chunks don't pollute the transcript agent's context and vice versa.

The metadata schema stored during ingestion:

| Field | Example | Used as filter? |
|-------|---------|----------------|
| `ticker` | `"AMZN"` | Yes — always |
| `quarter` | `"Q4"` | No (could be added) |
| `year` | `"2025"` | No (could be added) |
| `section` | `"qa"` | No (could be added) |
| `speaker` | `"andy_jassy"` | No |
| `text` | raw chunk text | Returned in results |

---

## Tuning

| Setting | Env var | Default | Effect |
|---------|---------|---------|--------|
| Chunks per query | `RAG_TOP_K` | 10 | Higher = more recall, more grader LLM calls |
| Loop limit | `RAG_MAX_RETRIEVAL_ATTEMPTS` | 2 | Set to 1 to disable rewriting |

Setting `RAG_MAX_RETRIEVAL_ATTEMPTS=1` turns the agentic loop into a single-pass retrieval (vanilla RAG behaviour). Useful for debugging or cost control.
