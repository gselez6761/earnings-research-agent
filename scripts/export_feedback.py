"""Export the append-only JSONL feedback store to CSV for analysis.

This reads the feedback_store.jsonl and extracts the high-level
metrics, analyst actions, and ratings into a flat tabular format
suitable for manual prompt tuning and systematic error auditing.

Usage:
    python scripts/export_feedback.py --output data/feedback_store/export.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Ensure the src package is importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from earnings_research_agent.utils.config import settings
from earnings_research_agent.utils.logging import get_logger

logger = get_logger(__name__)


def export_to_csv(input_jsonl: Path, output_csv: Path) -> None:
    """Read the JSONL feedback store and export flattened metrics to CSV."""
    if not input_jsonl.exists():
        logger.error("Feedback store not found at %s", input_jsonl)
        sys.exit(1)

    logger.info("Reading feedback from %s", input_jsonl)

    rows: list[dict] = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue
            
            try:
                entry = json.loads(line)
                feedback = entry.get("human_feedback", {})

                row = {
                    "thread_id": entry.get("thread_id"),
                    "ticker": entry.get("ticker"),
                    "timestamp": entry.get("timestamp"),
                    "action": feedback.get("action"),
                    "quality_rating": feedback.get("quality_rating", ""),
                    "free_text_note": feedback.get("free_text_note", ""),
                    "num_signal_overrides": len(feedback.get("signal_overrides", [])),
                    "num_table_edits": len(feedback.get("table_edits", [])),
                }
                rows.append(row)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON on line %d", line_idx + 1)

    if not rows:
        logger.info("Feedback store is empty. Nothing to export.")
        return

    fieldnames = [
        "thread_id", 
        "ticker", 
        "timestamp", 
        "action",
        "quality_rating", 
        "free_text_note",
        "num_signal_overrides", 
        "num_table_edits"
    ]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Exported %d feedback entries to %s", len(rows), output_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export feedback store to CSV.")
    
    default_out = (
        Path(__file__).resolve().parents[1] 
        / "data" 
        / "feedback_store" 
        / "export.csv"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=default_out,
        help="Output CSV file path."
    )
    
    args = parser.parse_args()
    input_path = Path(settings.feedback_store_path)
    
    export_to_csv(input_path, args.output)