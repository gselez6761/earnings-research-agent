"""EdgarTools MCP client for the earnings research agent.

This module wraps the edgartools MCP server (edgar.ai.mcp) which runs
as a local subprocess over stdio transport. It is the *only* place in
the project that knows how to speak MCP — all agents call the typed
methods here rather than constructing raw tool calls themselves.

The pattern is taken directly from the original financial agent script
shared by the team and from the MCP Python SDK documentation:

    async with stdio_client(MCP_SERVER) as (read, write):
        async with ClientSession(read, write) as mcp:
            await mcp.initialize()
            result = await mcp.call_tool(name, args)

Three tools used by this project:
    - get_financial_statements  — income statement, segments, 2 fiscal years
    - search_filings            — disclosure search with diff_only support
    - get_company_facts         — sector, segment classification for peer selection

The MCP server is started fresh per call to keep the architecture
stateless and compatible with LangGraph's async execution model.
Gemini parallel function calls in transcript_agent and peer_agent
both resolve through this client concurrently via asyncio.gather().

References:
    Original financial agent: src/earnings_research_agent/mcp/edgar_tools.py
    EdgarTools docs: https://edgartools.readthedocs.io/en/latest/ai/mcp-setup/
    MCP Python SDK: https://github.com/modelcontextprotocol/python-sdk
"""

from __future__ import annotations

import json
import sys
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from earnings_research_agent.utils.config import settings
from earnings_research_agent.utils.exceptions import EdgarMCPError
from earnings_research_agent.utils.logging import get_logger
from earnings_research_agent.state.graph_state import GraphState
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# MCP server process parameters
# Mirrors the original script exactly — spawns edgar.ai.mcp as a subprocess.
# EDGAR_IDENTITY must be "Full Name email@example.com" per SEC fair-access rules.
# ---------------------------------------------------------------------------

_MCP_SERVER = StdioServerParameters(
    command=sys.executable,
    args=[
        "-c",
        (
            "import logging; "
            "logging.getLogger('edgar').setLevel(logging.ERROR); "
            "from edgar.ai.mcp import main; main()"
        ),
    ],
    env={
        "EDGAR_IDENTITY": settings.edgar_identity,
    },
)


# ---------------------------------------------------------------------------
# Low-level call helper
# ---------------------------------------------------------------------------


async def _call_tool(tool_name: str, arguments: dict[str, Any]) -> str:
    """Open an MCP session, call one tool, return the text result.

    Opens a fresh subprocess and ClientSession per call. This keeps
    each invocation independent and avoids state leaking between
    parallel branches in the LangGraph graph.

    Args:
        tool_name:  Name of the MCP tool to invoke.
        arguments:  Dict of arguments matching the tool's input schema.

    Returns:
        The concatenated text content from the tool's response.

    Raises:
        EdgarMCPError: If the MCP server cannot be reached or the tool
            returns no text content.
    """
    try:
        async with stdio_client(_MCP_SERVER) as (read, write):
            async with ClientSession(read, write) as mcp:
                await mcp.initialize()
                result = await mcp.call_tool(tool_name, arguments)
                content = "\n".join(
                    c.text for c in result.content if hasattr(c, "text")
                )
                if not content:
                    raise EdgarMCPError(
                        f"Tool '{tool_name}' returned no text content "
                        f"for arguments: {arguments}"
                    )
                logger.debug("MCP tool '%s' succeeded: %d chars.", tool_name, len(content))
                return content
    except EdgarMCPError:
        raise
    except Exception as exc:
        raise EdgarMCPError(
            f"MCP tool '{tool_name}' failed: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Typed public API — one method per tool used in this project
# ---------------------------------------------------------------------------


async def get_financial_statements(
    ticker: str,
    num_periods: int = 2,
) -> dict[str, Any]:
    """Fetch income statement and segment data for a ticker.

    Returns structured financial data for the two most recent fiscal
    periods. Used by transcript_agent (target ticker) and peer_agent
    (target + all peers).

    Args:
        ticker:      Uppercase ticker symbol, e.g. "AMZN".
        num_periods: Number of fiscal periods to return (default 2 for QoQ delta).

    Returns:
        Parsed JSON dict with revenue, net_income, operating_income, segments.
    """
    raw = await _call_tool(
        "get_financial_statements",
        {"ticker": ticker, "num_periods": num_periods},
    )
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Some responses are narrative text rather than JSON — return as-is.
        return {"raw": raw}


async def search_filings(
    ticker: str,
    query: str,
    diff_only: bool = False,
) -> str:
    """Search narrative text inside 10-K and 10-Q filings.

    When diff_only=True, returns only passages that changed from the
    prior year — how analysts spot material changes in risk factors.
    This is Fix 2 (temporal comparison) applied to filing language,
    not just numeric deltas.

    Args:
        ticker:    Uppercase ticker symbol.
        query:     Natural language query, e.g. "AI infrastructure CapEx".
        diff_only: If True, only passages changed from prior year are returned.

    Returns:
        Matching disclosure passages as a string.
    """
    return await _call_tool(
        "search_filings",
        {"ticker": ticker, "query": query, "diff_only": diff_only},
    )


async def get_company_facts(ticker: str) -> dict[str, Any]:
    """Fetch sector, industry, and segment classification for a company.

    Used by peer_selector to verify that candidate peers share at least
    two major revenue segments and belong to the same sector before
    committing to the peer set.

    Args:
        ticker: Uppercase ticker symbol.

    Returns:
        Dict with sector, industry, segments, and business description.
    """
    raw = await _call_tool("get_company_facts", {"ticker": ticker})
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}


async def list_available_tools() -> list[str]:
    """Return the names of all tools exposed by the MCP server.

    Useful for debugging and verifying the server is reachable.
    """
    async with stdio_client(_MCP_SERVER) as (read, write):
        async with ClientSession(read, write) as mcp:
            await mcp.initialize()
            tools_result = await mcp.list_tools()
            return [t.name for t in tools_result.tools]




async def transcript_mcp_node(state: GraphState) -> dict[str, Any]:
    """LangGraph Node: Fetch exact financials and diff-based risk disclosures.
    
    This executes the parallel read-only tool calls against the EdgarTools 
    MCP server for the target ticker, utilizing the client functions defined above.
    """
    ticker = state["ticker"]
    logger.info("Executing EdgarTools MCP calls for %s", ticker)

    mcp_data = {}

    try:
        # Run the I/O bound MCP calls concurrently via asyncio
        # 1. Fetch exact financials for the last 2 quarters
        financials_task = get_financial_statements(ticker=ticker, num_periods=2)
        
        # 2. Fetch diff-only disclosures to automatically surface material risk factor changes
        disclosures_task = search_filings(
            ticker=ticker, 
            query="material changes in risk factors, business segments, or guidance", 
            diff_only=True
        )

        financials, risk_shifts = await asyncio.gather(financials_task, disclosures_task)

        mcp_data["financial_statements"] = financials
        mcp_data["risk_factor_shifts"] = risk_shifts

    except Exception as e:
        logger.error("EdgarTools MCP call failed for %s: %s", ticker, e)
        # Fail open: graph will not halt; agents will rely solely on transcript RAG chunks.
        mcp_data["financial_statements"] = None
        mcp_data["risk_factor_shifts"] = None

    # Return the retrieved structured data so it can be passed into the GraphState
    # and read by the transcript_agent node.
    return {"mcp_context": mcp_data}