"""LangChain @tool wrappers around the EdgarTools MCP client.

The raw async functions live in mcp/edgar_tools.py. This module exposes
them as LangChain tools so they can be bound to an LLM via bind_tools()
or invoked through a ToolNode if the architecture evolves in that direction.

Nodes that need deterministic, typed MCP results (transcript_mcp_node,
peer_mcp_node) call mcp/edgar_tools.py directly — not through these
wrappers. These tools are for agentic usage where the LLM decides which
MCP tools to call and with what arguments.
"""
from __future__ import annotations

import asyncio

from langchain_core.tools import tool

from earnings_research_agent.mcp.edgar_tools import (
    get_company_facts as _get_company_facts,
    get_financial_statements as _get_financial_statements,
    search_filings as _search_filings,
)


@tool
def get_financial_statements_tool(ticker: str, num_periods: int = 2) -> dict:
    """Fetch income statement and segment data for a public company.

    Args:
        ticker:      Uppercase ticker symbol (e.g. 'AMZN').
        num_periods: Number of fiscal periods to return (default 2 for QoQ delta).

    Returns:
        Dict with revenue, net_income, operating_income, and segments.
    """
    return asyncio.run(_get_financial_statements(ticker=ticker, num_periods=num_periods))


@tool
def search_filings_tool(ticker: str, query: str, diff_only: bool = False) -> str:
    """Search narrative text inside 10-K and 10-Q SEC filings.

    Args:
        ticker:    Uppercase ticker symbol.
        query:     Natural language query, e.g. 'AI infrastructure CapEx'.
        diff_only: If True, only return passages changed from the prior year.

    Returns:
        Matching disclosure passages as a string.
    """
    return asyncio.run(_search_filings(ticker=ticker, query=query, diff_only=diff_only))


@tool
def get_company_facts_tool(ticker: str) -> dict:
    """Fetch sector, industry, and segment classification for a company.

    Args:
        ticker: Uppercase ticker symbol.

    Returns:
        Dict with sector, industry, segments, and business description.
    """
    return asyncio.run(_get_company_facts(ticker=ticker))


# Convenience list for bind_tools() or ToolNode construction.
EDGAR_TOOLS = [
    get_financial_statements_tool,
    search_filings_tool,
    get_company_facts_tool,
]
