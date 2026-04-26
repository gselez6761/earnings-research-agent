# LangGraph Graph Reference

## Building and compiling

The graph is constructed in `graph/graph.py` via `build_graph(checkpointer)`. The checkpointer is injected so the same graph definition works in dev (MemorySaver) and prod (PostgresSaver).

```python
from earnings_research_agent.graph.graph import build_graph
from earnings_research_agent.graph.checkpointer import get_checkpointer

graph = build_graph(checkpointer=get_checkpointer())
```

`build_graph` returns a compiled `CompiledStateGraph`. Recompile only when the graph structure changes — the compiled object is safe to reuse across invocations.

---

## Node inventory

| Node | Type | Module | Writes to state |
|------|------|--------|----------------|
| `peer_selector` | async LLM | `agents/peer_selector.py` | `peers` |
| `transcript_retriever` | sync | `rag/retriever.py` | `transcript_chunks` |
| `transcript_mcp_node` | async | `mcp/edgar_tools.py` | `mcp_context` |
| `transcript_agent` | sync LLM | `agents/transcript_agent.py` | `executive_summary`, `signal_cards`, `temporal_deltas` |
| `peer_retriever` | sync | `rag/retriever.py` | `peer_chunks` |
| `peer_mcp_node` | async | `mcp/edgar_tools.py` | `peer_mcp_context` |
| `peer_analysis_node` | sync LLM | `agents/peer_agent.py` | `industry_trends`, `competitive_table` |
| `merge_node` | sync (no LLM) | `agents/merge_node.py` | `merged_report` |
| `human_review_node` | sync (interrupt) | `feedback/human_review.py` | `human_feedback` |
| `log_feedback_node` | sync | `feedback/log_feedback.py` | `feedback_store` (append) |
| `refine_node` | sync LLM | `agents/refine_node.py` | `merged_report` |

---

## Edge wiring

```python
START → peer_selector

# Parallel fork
peer_selector → transcript_retriever
peer_selector → peer_retriever

# Transcript branch (sequential)
transcript_retriever → transcript_mcp_node → transcript_agent → merge_node

# Peer branch (sequential)
peer_retriever → peer_mcp_node → peer_analysis_node → merge_node

# Fan-in at merge_node (LangGraph waits for both branches)
merge_node → human_review_node → log_feedback_node

# Conditional routing
log_feedback_node --[approve]--> END
log_feedback_node --[reject]---> END
log_feedback_node --[edit]-----> refine_node → human_review_node  (loop)
```

---

## Running the graph

### Stream mode (recommended for local dev)

```python
config = {"configurable": {"thread_id": "session_001"}}
initial_state = {"ticker": "AMZN", "thread_id": "session_001"}

for event in graph.stream(initial_state, config=config, stream_mode="values"):
    if "merged_report" in event and event["merged_report"]:
        print("Report ready — waiting for human review...")
```

The stream stops when the graph hits `interrupt()`. At that point, `graph.get_state(config).next` contains `("human_review_node",)`.

### Resuming after interrupt

```python
from langgraph.types import Command
from earnings_research_agent.state.schemas import HumanFeedback, FeedbackAction

feedback = HumanFeedback(action=FeedbackAction.APPROVE, quality_rating=4)

for event in graph.stream(Command(resume=feedback), config=config, stream_mode="values"):
    pass  # graph runs to END (or back to human_review_node if edit)
```

### Checking interrupt state

```python
state = graph.get_state(config)
print(state.next)          # ('human_review_node',) when paused
print(state.values.get("merged_report"))  # the report to show the analyst
```

---

## Async nodes and sync nodes

LangGraph handles mixed sync/async nodes transparently. Async nodes (`peer_selector`, `transcript_mcp_node`, `peer_mcp_node`) are awaited automatically. No special configuration needed.

Within async nodes, `asyncio.gather` is used to run concurrent MCP calls:

```python
# peer_mcp_node — fetches all peers in parallel
results = await asyncio.gather(
    *[get_financial_statements(ticker=p) for p in peers],
    return_exceptions=True
)
```

---

## Checkpointer reference

| Backend | Config | Use case |
|---------|--------|---------|
| `MemorySaver` | `POSTGRES_URL` not set | Dev/test — state lost on exit |
| `PostgresSaver` | `POSTGRES_URL=postgresql://...` | Production — survives restarts, required for long-lived interrupt/resume |

The factory in `graph/checkpointer.py` selects the backend automatically. See `docs/feedback_loop.md` for details on why PostgresSaver matters for the interrupt pattern.

---

## Thread IDs

Every graph invocation must include a `thread_id` in the config. This is the key the checkpointer uses to isolate state between concurrent runs.

```python
config = {"configurable": {"thread_id": "amzn-20250115-001"}}
```

Use a unique ID per analyst session. If you reuse a thread ID for a different ticker, the checkpointer will restore stale state from the previous run.

---

## Adding a new node

1. Implement the node function (sync or async) in the appropriate module.
2. Register it in `graph/graph.py`:
   ```python
   builder.add_node("my_node", my_node)
   builder.add_edge("upstream_node", "my_node")
   builder.add_edge("my_node", "downstream_node")
   ```
3. Add any new state fields it reads/writes to `state/graph_state.py`.
4. Add the corresponding Pydantic schema to `state/schemas.py` if needed.
