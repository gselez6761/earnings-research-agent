# System Architecture

## Overview

Multi-Agent Equity Research Terminal built on LangGraph. Takes a target stock ticker and produces a structured financial report by running two parallel agentic branches — one focused on the target company's transcript, one on peer companies — then merging them through a human review loop.

**Three design principles that drive every architectural decision:**

1. **Agentic RAG** — retrieval loops until chunks are relevant; no static top-k fetch.
2. **EdgarTools MCP** — deterministic SEC data replaces LLM-generated numbers.
3. **Structured human feedback** — interrupt/resume loop captures analyst corrections as an append-only audit log.

---

## Graph Execution Flow

```
START
  │
  ▼
peer_selector ──── calls EdgarTools MCP (get_company_facts) + Gemini
  │                → writes state.peers
  │
  ├─────────────────────────────────────────────────┐
  │  TRANSCRIPT BRANCH                              │  PEER BRANCH
  ▼                                                 ▼
transcript_retriever                           peer_retriever
  (agentic RAG loop, target ticker)              (agentic RAG loop, each peer)
  → state.transcript_chunks                      → state.peer_chunks
  │                                                 │
  ▼                                                 ▼
transcript_mcp_node                            peer_mcp_node
  (get_financial_statements + search_filings)    (get_financial_statements, all peers)
  → state.mcp_context                            → state.peer_mcp_context
  │                                                 │
  ▼                                                 ▼
transcript_agent                               peer_analysis_node
  (Gemini synthesis)                             (Gemini synthesis)
  → state.executive_summary                      → state.industry_trends
  → state.signal_cards  (Fix 1: citation check) → state.competitive_table
  → state.temporal_deltas
  │                                                 │
  └──────────────────────┬──────────────────────────┘
                         ▼
                     merge_node  (no LLM — pure Python)
                       Fix 3: suppress competitive rows where
                       confidence == MISSING (< 2 companies share segment)
                       → state.merged_report
                         │
                         ▼
                  human_review_node
                    interrupt() — graph pauses, yields report to caller
                    waits for Command(resume=HumanFeedback)
                         │
                         ▼
                  log_feedback_node
                    appends FeedbackEntry to feedback.jsonl
                         │
              ┌──────────┼──────────┐
           approve     edit       reject
              │          │           │
             END    refine_node     END
                         │
                         ▼
                  human_review_node  (loop)
```

---

## State: GraphState

Single TypedDict shared by all nodes. LangGraph checkpoints it after every node so the `interrupt()` in `human_review_node` can pause indefinitely and resume across process restarts (when using `PostgresSaver`).

| Field | Type | Written by | Read by |
|-------|------|-----------|---------|
| `ticker` | `str` | caller | all nodes |
| `thread_id` | `str` | caller | `log_feedback_node` |
| `peers` | `list[str]` | `peer_selector` | `peer_retriever`, `peer_mcp_node`, `peer_analysis_node` |
| `transcript_chunks` | `list[dict]` | `transcript_retriever` | `transcript_agent` |
| `peer_chunks` | `list[dict]` | `peer_retriever` | `peer_analysis_node` |
| `mcp_context` | `dict` | `transcript_mcp_node` | `transcript_agent` |
| `peer_mcp_context` | `dict[ticker→data]` | `peer_mcp_node` | `peer_analysis_node` |
| `executive_summary` | `ExecutiveSummary` | `transcript_agent` | `merge_node` |
| `signal_cards` | `list[SignalCard]` | `transcript_agent` | `merge_node` |
| `temporal_deltas` | `list[TemporalDelta]` | `transcript_agent` | `merge_node` |
| `industry_trends` | `list[IndustryTrend]` | `peer_analysis_node` | `merge_node` |
| `competitive_table` | `list[CompetitiveRow]` | `peer_analysis_node` | `merge_node` |
| `merged_report` | `FinalReport` | `merge_node`, `refine_node` | `human_review_node`, `log_feedback_node` |
| `human_feedback` | `HumanFeedback` | `human_review_node` | `log_feedback_node`, `route_on_feedback`, `refine_node` |
| `feedback_store` | `list[FeedbackEntry]` | `log_feedback_node` | — |

`feedback_store` uses `Annotated[list, operator.add]` so parallel branches can safely append to it without a write conflict.

---

## Parallel Branch Design

After `peer_selector`, LangGraph fans out to `transcript_retriever` and `peer_retriever` simultaneously. Both branches write to different state keys (`transcript_chunks` vs `peer_chunks`, `mcp_context` vs `peer_mcp_context`) so there are no reducer conflicts. They reconverge at `merge_node`, which LangGraph treats as a fan-in — it waits for both branches to complete before executing.

---

## Conditional Routing

`route_on_feedback` (in `graph/edges.py`) reads `state.human_feedback.action`:

- `approve` → `END`
- `reject` → `END`
- `edit` → `refine_node` → `human_review_node` (loop until approve/reject)

If `human_feedback` is missing (e.g. a resume payload was malformed), it defaults to `reject` and logs a warning rather than routing to an unknown state.

---

## Fix Summary

| Fix | Problem | Solution |
|-----|---------|---------|
| Fix 1 | LLM invents chunk citations | `hallucination_checker.check_citations()` suppresses any `SignalCard` whose `chunk_id` is not in the retrieved set |
| Fix 2 | No temporal comparison | `TemporalDelta` schema + `search_filings(diff_only=True)` in `transcript_mcp_node` |
| Fix 3 | Table rows with < 2 companies | `merge_node` suppresses `CompetitiveRow` entries where all cells have `confidence == MISSING` |
