# Human Feedback Loop

## Overview

The graph pauses after `merge_node` and waits for an analyst decision. This is not RLHF — the model is never retrained. The feedback is an append-only audit log for:

- Catching systematic errors (e.g. "table confidence is always wrong for segment X")
- Manual prompt tuning based on patterns in analyst corrections
- Quality tracking over time (ratings per ticker, edit frequency)

---

## The interrupt/resume pattern

LangGraph's `interrupt()` suspends a node mid-execution and serialises the graph state to the checkpointer. Execution resumes when the caller sends `Command(resume=payload)`.

```python
# human_review_node
feedback_data = interrupt({
    "report": state.get("merged_report"),
    "signal_cards": state.get("signal_cards"),
    ...
})
# execution pauses here — caller sees the interrupt payload
# when Command(resume=HumanFeedback(...)) is sent, execution continues here
feedback = HumanFeedback(**feedback_data)
return {"human_feedback": feedback}
```

The interrupt payload (the dict passed to `interrupt()`) is what the frontend or CLI receives when it polls the graph state. The resume payload is validated against the `HumanFeedback` Pydantic schema.

---

## Feedback actions

### `approve`

No edits. The report is accepted as-is.

```
route_on_feedback → END
```

### `reject`

The report is discarded. Still logged to the feedback store so the failure is auditable.

```
route_on_feedback → END
```

### `edit`

Analyst provides corrections via `signal_overrides`, `table_edits`, and/or `free_text_note`. The graph routes to `refine_node`, which applies the edits via a focused LLM rewrite and loops back to `human_review_node` for re-approval.

```
route_on_feedback → refine_node → human_review_node → log_feedback_node → ...
```

The loop continues until the analyst approves or rejects.

---

## HumanFeedback schema

```python
class HumanFeedback(BaseModel):
    action: FeedbackAction               # approve | edit | reject

    signal_overrides: list[SignalOverride]  # correct signal type, headline, or detail
    table_edits: list[TableCellEdit]        # correct a cell in the competitive table
    free_text_note: Optional[str]           # general note for the audit log
    quality_rating: Optional[int]           # 1-5 quality score
```

**SignalOverride** — targets a specific signal by its original headline:

```python
SignalOverride(
    original_headline="AWS revenue growth accelerated to 17%",
    corrected_signal_type=SignalType.NEUTRAL,
    corrected_detail="Growth was 17%, down from 20% prior quarter",
    reason="Headline was misleading — growth decelerated, not accelerated"
)
```

**TableCellEdit** — targets a specific (offering, ticker) cell:

```python
TableCellEdit(
    offering_name="Cloud Services",
    ticker="MSFT",
    corrected_revenue_current="$28.9B",
    reason="MCP returned prior quarter data for this cell"
)
```

---

## refine_node

When action is `edit`, `refine_node` calls Gemini with:
- The full `FinalReport` serialised as JSON
- The analyst's corrections as structured text

It returns a new `FinalReport` with corrections applied. Only the specified fields are changed — the LLM is instructed not to alter anything not covered by the edits.

---

## Feedback store

Every run through `log_feedback_node` appends one line to `data/feedback_store/feedback.jsonl`:

```json
{
  "thread_id": "session_001",
  "ticker": "AMZN",
  "timestamp": "2025-01-15T14:23:11",
  "report_snapshot": { ... },
  "human_feedback": {
    "action": "edit",
    "quality_rating": 3,
    "signal_overrides": [...],
    "free_text_note": "Competitive table had wrong MSFT cloud revenue"
  }
}
```

`report_snapshot` is the exact `FinalReport` that was shown to the analyst — not the refined version. This preserves the model's raw output for error analysis.

### Reading the store

```python
from earnings_research_agent.feedback.feedback_store import (
    load_all, filter_by_ticker, quality_summary
)
from pathlib import Path

entries = load_all(Path("data/feedback_store/feedback.jsonl"))
amzn_entries = filter_by_ticker(Path("data/feedback_store/feedback.jsonl"), "AMZN")
summary = quality_summary(Path("data/feedback_store/feedback.jsonl"))
# {"AMZN": {"count": 12, "avg_rating": 3.8, "actions": {"approve": 8, "edit": 3, "reject": 1}}}
```

### Exporting to CSV

```bash
python scripts/export_feedback.py
# → data/feedback_store/export.csv
```

Columns: `thread_id, ticker, timestamp, action, quality_rating, free_text_note, num_signal_overrides, num_table_edits`

---

## Checkpointing and the interrupt

`human_review_node` can pause for as long as the analyst takes. For this to survive process restarts, the graph must be compiled with `PostgresSaver` rather than `MemorySaver`:

```dotenv
POSTGRES_URL=postgresql://user:pass@localhost:5432/dbname
```

With `MemorySaver` (the default dev config), the interrupt state lives in memory and is lost if the process exits before the analyst responds. The graph will need to be re-run from the start.

See `graph/checkpointer.py` — the factory automatically picks the right backend based on whether `POSTGRES_URL` is set.
