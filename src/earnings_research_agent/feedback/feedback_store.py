"""Read interface for the append-only JSONL feedback store.

The write side lives in log_feedback.py. This module provides typed read
access — iterating entries, filtering by ticker, and summarising quality
ratings — for use by export scripts and diagnostic tooling.

The store is a plain JSONL file: one FeedbackEntry JSON object per line.
Missing or malformed lines are skipped with a warning rather than raising,
so a partially-corrupted file doesn't block reads of valid entries.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterator

from earnings_research_agent.state.schemas import FeedbackEntry
from earnings_research_agent.utils.logging import get_logger

logger = get_logger(__name__)


def iter_entries(store_path: Path) -> Iterator[FeedbackEntry]:
    """Yield parsed FeedbackEntry records from the JSONL store.

    Args:
        store_path: Path to the feedback JSONL file.

    Yields:
        FeedbackEntry objects in insertion order.
    """
    if not store_path.exists():
        logger.warning("Feedback store not found at %s.", store_path)
        return

    with open(store_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield FeedbackEntry.model_validate_json(line)
            except Exception as exc:
                logger.warning(
                    "Skipping malformed entry on line %d: %s", line_num, exc
                )


def load_all(store_path: Path) -> list[FeedbackEntry]:
    """Return all valid FeedbackEntry records from the store."""
    return list(iter_entries(store_path))


def filter_by_ticker(store_path: Path, ticker: str) -> list[FeedbackEntry]:
    """Return only entries for the given ticker symbol."""
    return [e for e in iter_entries(store_path) if e.ticker == ticker]


def quality_summary(store_path: Path) -> dict[str, dict]:
    """Compute per-ticker quality rating statistics.

    Returns:
        Dict mapping ticker → {"count": int, "avg_rating": float | None,
        "actions": {"approve": int, "edit": int, "reject": int}}.
    """
    summary: dict[str, dict] = defaultdict(
        lambda: {"count": 0, "ratings": [], "actions": defaultdict(int)}
    )

    for entry in iter_entries(store_path):
        t = entry.ticker
        summary[t]["count"] += 1
        summary[t]["actions"][entry.human_feedback.action.value] += 1
        if entry.human_feedback.quality_rating is not None:
            summary[t]["ratings"].append(entry.human_feedback.quality_rating)

    result = {}
    for ticker, data in summary.items():
        ratings = data["ratings"]
        result[ticker] = {
            "count": data["count"],
            "avg_rating": round(sum(ratings) / len(ratings), 2) if ratings else None,
            "actions": dict(data["actions"]),
        }
    return result
