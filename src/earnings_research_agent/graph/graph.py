"""LangGraph compilation for the Multi-Agent Equity Research Terminal.

This module defines the StateGraph, adds all nodes, maps the edges for parallel 
execution, and establishes the interrupt-driven human review loop.
"""
from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.base import BaseCheckpointSaver

from earnings_research_agent.state.graph_state import GraphState

# Import nodes (Assuming these will be implemented in their respective files)
from earnings_research_agent.agents.peer_selector import peer_selector
from earnings_research_agent.rag.retriever import transcript_retriever, peer_retriever
from earnings_research_agent.mcp.edgar_tools import transcript_mcp_node, peer_mcp_node
from earnings_research_agent.agents.peer_agent import peer_analysis_node
from earnings_research_agent.agents.transcript_agent import transcript_agent
from earnings_research_agent.agents.merge_node import merge_node
from earnings_research_agent.feedback.human_review import human_review_node
from earnings_research_agent.feedback.log_feedback import log_feedback_node
from earnings_research_agent.agents.refine_node import refine_node
from earnings_research_agent.graph.edges import route_on_feedback


def build_graph(checkpointer: BaseCheckpointSaver) -> StateGraph:
    """Construct and compile the LangGraph workflow."""
    
    builder = StateGraph(GraphState)

    # ------------------------------------------------------------------
    # Add Nodes
    # ------------------------------------------------------------------
    builder.add_node("peer_selector", peer_selector)
    
    # Transcript Branch Nodes
    builder.add_node("transcript_retriever", transcript_retriever)
    builder.add_node("transcript_mcp_node", transcript_mcp_node)
    builder.add_node("transcript_agent", transcript_agent)
    
    # Peer Branch Nodes
    builder.add_node("peer_retriever", peer_retriever)
    builder.add_node("peer_mcp_node", peer_mcp_node)
    builder.add_node("peer_analysis_node", peer_analysis_node)
    
    # Merge & Feedback Nodes
    builder.add_node("merge_node", merge_node)
    builder.add_node("human_review_node", human_review_node)
    builder.add_node("log_feedback_node", log_feedback_node)
    builder.add_node("refine_node", refine_node)

    # ------------------------------------------------------------------
    # Define Edges (Flow)
    # ------------------------------------------------------------------
    builder.add_edge(START, "peer_selector")
    
    # Fork into parallel branches
    builder.add_edge("peer_selector", "transcript_retriever")
    builder.add_edge("peer_selector", "peer_retriever")

    # Transcript Branch Flow
    builder.add_edge("transcript_retriever", "transcript_mcp_node")
    builder.add_edge("transcript_mcp_node", "transcript_agent")
    builder.add_edge("transcript_agent", "merge_node")

    # Peer Branch Flow
    builder.add_edge("peer_retriever", "peer_mcp_node")
    builder.add_edge("peer_mcp_node", "peer_analysis_node")
    builder.add_edge("peer_analysis_node", "merge_node")

    # The Fan-in / Merge
    builder.add_edge("merge_node", "human_review_node")
    
    # Human Review to Logging
    builder.add_edge("human_review_node", "log_feedback_node")

    # Conditional routing based on analyst action (Approve/Edit/Reject)
    builder.add_conditional_edges(
        "log_feedback_node",
        route_on_feedback,
        {
            "approve": END,
            "reject": END,
            "edit": "refine_node"
        }
    )

    # If edited, loop back to human review for final approval
    builder.add_edge("refine_node", "human_review_node")

    # Compile the graph with checkpointer
    return builder.compile(checkpointer=checkpointer)