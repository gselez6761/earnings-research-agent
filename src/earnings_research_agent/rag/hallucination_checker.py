"""Citation-based hallucination checker.

Verifies that every SignalCard citation references a chunk_id that was
actually retrieved from Pinecone. Cards with invented chunk_ids are
suppressed rather than passed downstream.

This is Fix 1: grounding citations to specific, verified chunk IDs.
"""
from __future__ import annotations

from earnings_research_agent.state.schemas import SignalCard
from earnings_research_agent.utils.logging import get_logger

logger = get_logger(__name__)


def check_citations(
    signal_cards: list[SignalCard],
    valid_chunk_ids: set[str],
) -> tuple[list[SignalCard], int]:
    """Filter out signal cards whose chunk_id was not in the retrieved set.

    Args:
        signal_cards:    Raw list of SignalCards from the LLM.
        valid_chunk_ids: Set of chunk IDs returned by Pinecone (ground truth).

    Returns:
        A tuple of (validated_cards, suppressed_count).
        suppressed_count > 0 signals systematic hallucination for audit review.
    """
    validated: list[SignalCard] = []
    suppressed = 0

    for card in signal_cards:
        if card.citation.chunk_id in valid_chunk_ids:
            validated.append(card)
        else:
            logger.warning(
                "Suppressing signal '%s' — chunk_id '%s' not in retrieved set.",
                card.headline,
                card.citation.chunk_id,
            )
            suppressed += 1

    if suppressed:
        logger.warning(
            "%d / %d signals suppressed for hallucinated citations.",
            suppressed,
            len(signal_cards),
        )

    return validated, suppressed
