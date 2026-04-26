"""Pydantic schemas for all typed objects in the graph state.

Every structured object that flows through GraphState is defined here.
Agents read and write these — never plain dicts.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SignalType(str, Enum):
    """Directional classification of a key insight signal."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class SectionType(str, Enum):
    """Which part of the earnings call the chunk came from."""

    PREPARED_REMARKS = "prepared_remarks"
    QA = "qa"


class SegmentConfidence(str, Enum):
    """How confident the peer agent is that two companies share a segment.

    exact   — segment names matched directly (e.g. 'AWS' == 'AWS').
    inferred — agent mapped similar but distinct names (e.g. 'AWS' ~ 'Cloud Services').
    missing  — fewer than 2 companies share this segment; row is suppressed by merge_node.
    """

    EXACT = "exact"
    INFERRED = "inferred"
    MISSING = "missing"


class FeedbackAction(str, Enum):
    """Analyst decision at the human review interrupt."""

    APPROVE = "approve"
    EDIT = "edit"
    REJECT = "reject"


# ---------------------------------------------------------------------------
# Fix 1 — Citations
# ---------------------------------------------------------------------------


class Citation(BaseModel):
    """Grounded source reference attached to every signal card.

    chunk_id ties back to the Pinecone vector ID so the frontend
    can deep-link to the exact transcript passage.
    """

    chunk_id: str = Field(..., description="MD5 Pinecone vector ID of the source chunk.")
    ticker: str
    quarter: str  # e.g. "Q4"
    year: str     # e.g. "2025"
    section: SectionType
    speaker: str  # normalised snake_case speaker name


# ---------------------------------------------------------------------------
# Fix 2 — Temporal deltas
# ---------------------------------------------------------------------------


class TemporalDelta(BaseModel):
    """Quarter-over-quarter change for a single metric or guidance item."""

    metric: str = Field(..., description="Human-readable metric name, e.g. 'AWS Revenue Growth'.")
    current_value: str   # raw string as reported, e.g. "$28.8B" or "17%"
    prior_value: str     # same format, prior quarter
    direction: str       # "up" | "down" | "flat"
    commentary: str      # one sentence from transcript explaining the change


# ---------------------------------------------------------------------------
# Signal cards (executive summary key insights)
# ---------------------------------------------------------------------------


class SignalCard(BaseModel):
    """A single bullish / bearish / neutral insight with grounded citation."""

    signal_type: SignalType
    headline: str = Field(..., max_length=120)
    detail: str
    citation: Citation  # Fix 1 — always required, never optional


# ---------------------------------------------------------------------------
# Executive summary
# ---------------------------------------------------------------------------


class ExecutiveSummary(BaseModel):
    """Top-line financial metrics and guidance for the target ticker."""

    total_revenue: str
    net_income: str
    operating_margin: str
    top_growth_segments: list[str] = Field(..., max_length=2)
    headline_takeaway: str
    primary_driver: str
    forward_guidance: str  # next-quarter targets, CapEx, segment outlook
    temporal_deltas: list[TemporalDelta]  # Fix 2


# ---------------------------------------------------------------------------
# Industry trends (peer agent output)
# ---------------------------------------------------------------------------


class TrendCategory(str, Enum):
    DOMINANT = "dominant"
    EMERGING = "emerging"
    PERSISTENT = "persistent"


class IndustryTrend(BaseModel):
    """One of exactly three cross-company themes from peer transcripts."""

    category: TrendCategory
    theme: str
    data_points: list[str]
    tickers_discussed: list[str]  # which peers mentioned this theme


# ---------------------------------------------------------------------------
# Competitive landscape (Fix 3 — confidence per cell)
# ---------------------------------------------------------------------------


class SegmentCell(BaseModel):
    """Revenue data for one company in one shared segment."""

    ticker: str
    segment_name: str            # company's own label for the segment
    revenue_current: str         # as reported, e.g. "$28.8B"
    revenue_prior: str
    yoy_growth: str              # e.g. "+17%"
    confidence: SegmentConfidence  # Fix 3


class CompetitiveRow(BaseModel):
    """One row in the competitive landscape table (one shared segment)."""

    offering_name: str           # normalised label, e.g. "Cloud Services"
    cells: list[SegmentCell]     # one per company; missing ones omitted (suppressed)


# ---------------------------------------------------------------------------
# Merged final report
# ---------------------------------------------------------------------------


class FinalReport(BaseModel):
    """Complete structured output emitted after merge_node."""

    ticker: str
    peers: list[str]
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    executive_summary: ExecutiveSummary
    signal_cards: list[SignalCard]
    industry_trends: list[IndustryTrend]  # exactly 3
    competitive_table: list[CompetitiveRow]


# ---------------------------------------------------------------------------
# Human feedback (replaces RLHF — audit log only)
# ---------------------------------------------------------------------------


class SignalOverride(BaseModel):
    """Analyst correction to a specific signal card."""

    original_headline: str
    corrected_signal_type: Optional[SignalType] = None
    corrected_headline: Optional[str] = None
    corrected_detail: Optional[str] = None
    reason: Optional[str] = None


class TableCellEdit(BaseModel):
    """Analyst correction to a competitive table cell."""

    offering_name: str
    ticker: str
    corrected_revenue_current: Optional[str] = None
    corrected_segment_name: Optional[str] = None
    reason: Optional[str] = None


class HumanFeedback(BaseModel):
    """Full payload from the analyst at the human_review_node interrupt."""

    action: FeedbackAction
    signal_overrides: list[SignalOverride] = Field(default_factory=list)
    table_edits: list[TableCellEdit] = Field(default_factory=list)
    free_text_note: Optional[str] = None
    quality_rating: Optional[int] = Field(None, ge=1, le=5)


class FeedbackEntry(BaseModel):
    """Append-only audit record written by log_feedback_node.

    This is NOT an RLHF training record — the model is never retrained.
    It exists for auditing, manual prompt analysis, and systematic error
    detection (e.g. 'table confidence is always wrong for segment X').
    """

    thread_id: str
    ticker: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    report_snapshot: FinalReport
    human_feedback: HumanFeedback