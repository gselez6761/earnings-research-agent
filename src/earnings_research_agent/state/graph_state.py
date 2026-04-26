"""GraphState — the single TypedDict shared across all LangGraph nodes.

Every node reads from and writes back to this state.
No agent passes data directly to another agent — everything goes through here.

LangGraph persists a checkpoint of this state after every node via the
configured checkpointer (InMemorySaver in dev, PostgresSaver in prod).
The interrupt() in human_review_node can pause indefinitely; when
Command(resume=...) is called, this exact state is restored.
"""

from __future__ import annotations

from typing import Annotated, Any, Optional
import operator

from langgraph.graph import MessagesState

from earnings_research_agent.state.schemas import (
    FeedbackEntry,
    FinalReport,
    HumanFeedback,
    IndustryTrend,
    CompetitiveRow,
    ExecutiveSummary,
    SignalCard,
    TemporalDelta,
)


class GraphState(MessagesState):
    """Full state for the earnings research graph.

    Fields are typed explicitly so LangGraph can checkpoint them correctly.
    Annotated[list, operator.add] means LangGraph will *append* to that list
    across parallel branches rather than overwrite — used for feedback_store.
    """

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------
    ticker: str
    thread_id: str

    # ------------------------------------------------------------------
    # Peer selection (set by peer_selector node)
    # ------------------------------------------------------------------
    peers: list[str]

    # ------------------------------------------------------------------
    # Agentic RAG outputs — separate keys per branch so parallel writes
    # don't collide (no reducer needed; each branch owns its own key).
    # ------------------------------------------------------------------
    transcript_chunks: list[dict[str, Any]]  # written by transcript_retriever
    peer_chunks: list[dict[str, Any]]        # written by peer_retriever

    # ------------------------------------------------------------------
    # MCP context — one per branch, set by their respective MCP nodes
    # ------------------------------------------------------------------
    mcp_context: Optional[dict[str, Any]]           # transcript branch
    peer_mcp_context: Optional[dict[str, Any]]      # peer branch: ticker → financials

    # ------------------------------------------------------------------
    # Parallel branch outputs
    # transcript_agent writes these; peer_agent writes the others.
    # merge_node reads all four.
    # ------------------------------------------------------------------
    executive_summary: Optional[ExecutiveSummary]
    signal_cards: list[SignalCard]
    temporal_deltas: list[TemporalDelta]     # Fix 2 — QoQ deltas
    industry_trends: list[IndustryTrend]     # exactly 3, from peer_agent
    competitive_table: list[CompetitiveRow]  # Fix 3 — confidence per cell

    # ------------------------------------------------------------------
    # Merged output (set by merge_node, read by human_review_node)
    # ------------------------------------------------------------------
    merged_report: Optional[FinalReport]

    # ------------------------------------------------------------------
    # Human feedback (set by Command(resume=...) after interrupt())
    # ------------------------------------------------------------------
    human_feedback: Optional[HumanFeedback]

    # ------------------------------------------------------------------
    # Feedback store — append-only audit log
    # operator.add means parallel branches can both append safely.
    # This is NOT an RLHF dataset — the model is never retrained.
    # Used for auditing and manual prompt analysis.
    # ------------------------------------------------------------------
    feedback_store: Annotated[list[FeedbackEntry], operator.add]