"""Direct edgartools XBRL segment revenue fetch.

Bypasses the MCP subprocess and calls edgartools Python API directly to get
structured segment revenue from the latest 10-K. Used by the MCP nodes to
enrich the financial context before peer_analysis_node runs.
"""
from __future__ import annotations

import math
from typing import Any

import edgar

from earnings_research_agent.utils.config import settings
from earnings_research_agent.utils.logging import get_logger

logger = get_logger(__name__)

# Dimension axes we care about, in priority order for the competitive table.
_SEGMENT_AXES = [
    "us-gaap:StatementBusinessSegmentsAxis",  # North America / International / AWS
    "srt:ProductOrServiceAxis",               # Online stores / Advertising / etc.
]


def _fmt_billions(val: float) -> str:
    b = val / 1e9
    if b >= 1:
        return f"${b:.1f}B"
    return f"${val / 1e6:.0f}M"


def _safe_float(val: Any) -> float | None:
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def fetch_xbrl_segments(ticker: str) -> dict[str, Any]:
    """Return segment revenue for the two most recent fiscal years.

    Returns a dict like:
    {
        "period_current": "2025-12-31",
        "period_prior": "2024-12-31",
        "business_segments": {
            "North America": {"revenue_current": "$426.3B", "revenue_prior": "$387.5B", "yoy_growth": "+10.0%"},
            "AWS": {"revenue_current": "$128.7B", "revenue_prior": "$107.6B", "yoy_growth": "+19.6%"},
        },
        "product_segments": {
            "Online stores": ...,
        }
    }
    Returns {} on any failure so callers degrade gracefully.
    """
    try:
        edgar.set_identity(settings.edgar_identity)
        from edgar import Company  # noqa: PLC0415

        company = Company(ticker)
        filings = company.get_filings(form="10-K")
        if not filings:
            logger.warning("No 10-K filings found for %s", ticker)
            return {}

        filing = filings[0]
        xbrl = filing.xbrl()
        income = xbrl.statements.income_statement(view="detailed")
        df = income.to_dataframe()

        # Identify the two most recent period columns (contain parentheses, e.g. "2025-12-31 (FY)")
        period_cols = [c for c in df.columns if "(" in c]
        if len(period_cols) < 2:
            logger.warning("%s: only %d period columns found in XBRL income statement", ticker, len(period_cols))
            return {}

        curr_col, prior_col = period_cols[0], period_cols[1]
        period_current = curr_col.split(" ")[0]
        period_prior = prior_col.split(" ")[0]

        # Key top-level income statement labels to always extract
        _TOP_LABELS = {
            "total net sales": "Total Revenue",
            "net sales": "Total Revenue",
            "total revenue": "Total Revenue",
            "revenues": "Total Revenue",
            "operating income": "Operating Income",
            "operating income (loss)": "Operating Income",
            "net income": "Net Income",
            "net income (loss)": "Net Income",
            "income before income taxes": "Income Before Tax",
            "provision for income taxes": "Income Tax",
        }

        top_df = df[df["dimension"] == False].copy()
        income_metrics: dict[str, dict] = {}
        for _, row in top_df.iterrows():
            label_raw = str(row.get("label", "")).strip().lower()
            canonical = _TOP_LABELS.get(label_raw)
            if not canonical or canonical in income_metrics:
                continue
            curr = _safe_float(row.get(curr_col))
            prior = _safe_float(row.get(prior_col))
            if curr is None:
                continue
            yoy: str
            if prior and prior != 0:
                pct = (curr - prior) / abs(prior) * 100
                yoy = f"{'+' if pct >= 0 else ''}{pct:.1f}%"
            else:
                yoy = "No Data"
            income_metrics[canonical] = {
                "revenue_current": _fmt_billions(curr),
                "revenue_prior": _fmt_billions(prior) if prior else "No Data",
                "yoy_growth": yoy,
            }

        result: dict[str, Any] = {
            "period_current": period_current,
            "period_prior": period_prior,
            "income_metrics": income_metrics,
            "business_segments": {},
            "product_segments": {},
        }

        seg_df = df[df["dimension"] == True].copy()

        for axis, key in [
            ("us-gaap:StatementBusinessSegmentsAxis", "business_segments"),
            ("srt:ProductOrServiceAxis", "product_segments"),
        ]:
            axis_rows = seg_df[seg_df["dimension_axis"] == axis]
            seen: set[str] = set()

            for _, row in axis_rows.iterrows():
                label = str(row.get("dimension_member_label") or row.get("label", "")).strip()
                if not label or label in seen:
                    continue

                curr = _safe_float(row.get(curr_col))
                if curr is None:
                    continue

                seen.add(label)
                prior = _safe_float(row.get(prior_col))
                yoy: str
                if prior and prior != 0:
                    pct = (curr - prior) / abs(prior) * 100
                    yoy = f"{'+' if pct >= 0 else ''}{pct:.1f}%"
                else:
                    yoy = "No Data"

                result[key][label] = {
                    "revenue_current": _fmt_billions(curr),
                    "revenue_prior": _fmt_billions(prior) if prior else "No Data",
                    "yoy_growth": yoy,
                }

        seg_count = len(result["business_segments"]) + len(result["product_segments"])
        logger.info("%s XBRL: %d segments extracted (%s)", ticker, seg_count, period_current)
        return result

    except Exception as e:
        logger.warning("XBRL segment fetch failed for %s: %s", ticker, e)
        return {}
