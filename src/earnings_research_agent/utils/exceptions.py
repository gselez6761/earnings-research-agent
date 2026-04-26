"""Project-specific exception hierarchy.

All custom exceptions inherit from EarningsResearchError so callers
can catch the entire family with a single except clause when needed.

Google style: use built-in exception classes where they fit;
define custom ones only when the semantics differ meaningfully.
"""

from __future__ import annotations


class EarningsResearchError(Exception):
    """Base exception for the earnings research agent."""


class RetrievalError(EarningsResearchError):
    """Raised when Pinecone retrieval fails after max attempts."""


class GradingError(EarningsResearchError):
    """Raised when the RAG grader cannot assess chunk relevance."""


class EdgarMCPError(EarningsResearchError):
    """Raised when the EdgarTools MCP server returns an error or is unreachable."""


class PeerSelectionError(EarningsResearchError):
    """Raised when the peer selector cannot find valid comparable companies."""


class MergeError(EarningsResearchError):
    """Raised when merge_node receives incomplete branch outputs."""


class FeedbackStoreError(EarningsResearchError):
    """Raised when writing to the feedback store JSONL file fails."""


class CheckpointerError(EarningsResearchError):
    """Raised when the LangGraph checkpointer cannot save or restore state."""