"""FastAPI backend for the Earnings Research Agent.

Exposes three endpoints:
  POST /api/research/{ticker}          — start a graph run, return thread_id
  GET  /api/research/{thread_id}/stream — SSE stream of status/review/complete events
  POST /api/research/{thread_id}/feedback — submit human feedback to resume the graph
"""
from __future__ import annotations

import asyncio
import json
import sys
import uuid
import re
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from earnings_research_agent.graph.graph import build_graph
from earnings_research_agent.graph.checkpointer import get_checkpointer
from earnings_research_agent.state.schemas import (
    FinalReport,
    HumanFeedback,
    FeedbackAction,
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Earnings Research Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------


@dataclass
class RunSession:
    thread_id: str
    event_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    feedback_queue: asyncio.Queue = field(default_factory=asyncio.Queue)


sessions: dict[str, RunSession] = {}

# ---------------------------------------------------------------------------
# Graph singleton
# ---------------------------------------------------------------------------

_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph(checkpointer=get_checkpointer())
    return _graph


# ---------------------------------------------------------------------------
# Report transformation helpers
# ---------------------------------------------------------------------------


def _parse_dollar(s):
    if not s:
        return None
    m = re.match(r'\$?([\d.]+)([BMK]?)', str(s).strip().upper().replace(',', ''))
    if not m:
        return None
    mul = {'B': 1e9, 'M': 1e6, 'K': 1e3}.get(m.group(2), 1)
    try:
        return float(m.group(1)) * mul
    except Exception:
        return None


def _parse_growth(s):
    if not s:
        return None
    try:
        return float(str(s).strip().replace('%', '').replace('+', ''))
    except Exception:
        return None


def transform_report(report: FinalReport) -> dict:
    ex = report.executive_summary

    # Base metrics
    metrics = [
        {"label": "Total Revenue", "value": ex.total_revenue or "N/A"},
        {"label": "Net Income",    "value": ex.net_income or "N/A"},
        {"label": "Op. Margin",    "value": ex.operating_margin or "N/A"},
    ]
    # Add temporal deltas as extra metric cards
    for delta in (ex.temporal_deltas or [])[:3]:
        yoy = 1.0 if delta.direction == "up" else (-1.0 if delta.direction == "down" else 0.0)
        curr = _parse_dollar(delta.current_value)
        prior = _parse_dollar(delta.prior_value)
        if curr and prior and prior != 0:
            yoy = round((curr - prior) / prior * 100, 1)
        metrics.append({"label": delta.metric, "value": delta.current_value, "yoy_change": yoy})

    # Signal cards → key_insights
    key_insights = [
        {"signal": c.signal_type.value, "title": c.headline, "detail": c.detail}
        for c in (report.signal_cards or [])
    ]

    # IndustryTrends → themes
    themes = [
        {
            "category": t.category.value,
            "title": t.theme,
            "detail": " ".join(t.data_points[:2]) if t.data_points else "",
            "mentioned_by": t.tickers_discussed or [],
        }
        for t in (report.industry_trends or [])
    ]

    # CompetitiveTable → competitive_landscape
    all_tickers = [report.ticker] + [p for p in (report.peers or []) if p != report.ticker]
    companies = list(dict.fromkeys(all_tickers))  # deduplicate, preserve order

    offerings = []
    for row in (report.competitive_table or []):
        cells_by_ticker = {c.ticker: c for c in row.cells}
        positions = {}
        for i, company in enumerate(companies):
            cell = cells_by_ticker.get(company)
            if cell:
                positions[i] = {
                    "has_segment": True,
                    "segment_name": cell.segment_name,
                    "revenue": cell.revenue_current,
                    "revenue_prior_year": cell.revenue_prior,
                    "yoy_growth": _parse_growth(cell.yoy_growth),
                }
            else:
                positions[i] = {"has_segment": False}
        offerings.append({"category": row.offering_name, "positions": positions})

    return {
        "ticker": report.ticker,
        "quarter": "",
        "executive_summary": {
            "metrics": metrics,
            "headline_takeaway": ex.headline_takeaway or "",
            "primary_driver": ex.primary_driver or "",
            "forward_guidance": ex.forward_guidance or "",
            "key_drivers": ex.top_growth_segments or [],
        },
        "key_insights": key_insights,
        "industry_trends": {
            "themes": themes,
            "sources": [f"{report.ticker} Earnings"] + [f"{p} Earnings" for p in (report.peers or [])[:3]],
        },
        "competitive_landscape": {
            "companies": companies,
            "offerings": offerings,
        },
    }


# ---------------------------------------------------------------------------
# Background graph task
# ---------------------------------------------------------------------------

node_phases = {
    "peer_selector": "Identifying peer companies...",
    "transcript_retriever": "Scanning earnings transcripts...",
    "transcript_mcp_node": "Fetching SEC financials...",
    "transcript_agent": "Analyzing transcript...",
    "peer_retriever": "Scanning peer transcripts...",
    "peer_mcp_node": "Fetching peer financials...",
    "peer_analysis_node": "Analyzing competitive landscape...",
    "merge_node": "Compiling research report...",
}


async def run_graph_task(session: RunSession, ticker: str) -> None:
    graph = get_graph()
    config = {"configurable": {"thread_id": session.thread_id}}
    initial_state = {"ticker": ticker, "thread_id": session.thread_id}

    try:
        async for event in graph.astream(initial_state, config=config, stream_mode="updates"):
            for node_name in event:
                phase = node_phases.get(node_name)
                if phase:
                    await session.event_queue.put({"type": "status", "phase": phase})

        # Loop to handle edit cycles. The graph uses interrupt_before=["human_review_node"],
        # so it pauses before that node. We inject feedback via update_state(), then
        # resume by passing None as input — avoids interrupt()/get_config() entirely.
        while True:
            state = graph.get_state(config)
            if not (state.next and "human_review_node" in state.next):
                break

            report = state.values.get("merged_report")
            await session.event_queue.put({
                "type": "review",
                "report": transform_report(report) if report else None,
            })

            # Wait for human feedback
            feedback = await session.feedback_queue.get()

            # Inject feedback into graph state, then resume from the interrupt point
            graph.update_state(config, {"human_feedback": feedback})
            async for event in graph.astream(
                None, config=config, stream_mode="updates"
            ):
                for node_name in event:
                    phase = node_phases.get(node_name)
                    if phase:
                        await session.event_queue.put({"type": "status", "phase": phase})

        final_state = graph.get_state(config)
        report = final_state.values.get("merged_report")
        await session.event_queue.put({
            "type": "complete",
            "report": transform_report(report) if report else None,
        })

    except Exception as e:
        import traceback
        await session.event_queue.put({
            "type": "error",
            "message": str(e),
            "detail": traceback.format_exc(),
        })

    finally:
        await session.event_queue.put(None)  # sentinel closes SSE stream


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/api/research/{ticker}")
async def start_research(ticker: str):
    thread_id = str(uuid.uuid4())
    session = RunSession(thread_id=thread_id)
    sessions[thread_id] = session
    asyncio.create_task(run_graph_task(session, ticker.upper()))
    return {"thread_id": thread_id}


@app.get("/api/research/{thread_id}/stream")
async def stream_research(thread_id: str):
    session = sessions.get(thread_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    async def generate():
        while True:
            try:
                event = await asyncio.wait_for(session.event_queue.get(), timeout=300)
            except asyncio.TimeoutError:
                yield {"comment": "ping"}
                continue
            if event is None:
                return
            yield {"data": json.dumps(event)}

    return EventSourceResponse(generate())


@app.post("/api/research/{thread_id}/feedback")
async def submit_feedback(thread_id: str, body: dict):
    session = sessions.get(thread_id)
    if not session:
        raise HTTPException(status_code=404)

    action = body.get("action", "approve")
    note = body.get("note", "")

    feedback = HumanFeedback(
        action=FeedbackAction(action),
        free_text_note=note or None,
        quality_rating=body.get("quality_rating"),
    )
    await session.feedback_queue.put(feedback)
    return {"ok": True}
