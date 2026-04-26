# EdgarTools MCP Setup

## What is the MCP server?

The EdgarTools MCP server (`edgar.ai.mcp`) provides deterministic access to SEC EDGAR data — income statements, segment breakdowns, and filing text — without scraping HTML or maintaining a local dataset. It runs as a subprocess on your machine via stdio transport. No port, no network config, no Docker.

The project spawns a fresh subprocess per MCP call, which keeps each invocation independent and safe for LangGraph's concurrent async execution.

---

## Installation

EdgarTools and the MCP Python SDK are both installed with the project dependencies:

```bash
pip install "edgartools[ai]" mcp
```

Verify the MCP server can start:

```bash
python -c "from edgar.ai.mcp import main; print('EdgarTools MCP OK')"
```

---

## SEC EDGAR identity (required)

The SEC requires programmatic access to identify the requester. EdgarTools reads this from the `EDGAR_IDENTITY` environment variable and sends it as the User-Agent header on every EDGAR HTTP request.

**Format:** `First Last email@example.com`

```dotenv
EDGAR_IDENTITY=Jane Smith jane@example.com
```

The project validates this format at startup (`mcp/auth.py`). If it's wrong or missing, the process exits immediately with:

```
ValueError: EDGAR_IDENTITY must be 'First Last email@example.com' — got: ''
```

This is intentional: a bad identity causes SEC rate-limiting (HTTP 403) mid-run, which is much harder to diagnose.

Reference: [SEC EDGAR fair-access policy](https://www.sec.gov/os/accessing-edgar-data)

---

## Tools used

Three tools from the MCP server are called in this project:

### `get_financial_statements`

Fetches income statement and segment data for a ticker.

```python
await get_financial_statements(ticker="AMZN", num_periods=2)
```

Returns JSON: revenue, net_income, operating_income, segments for the two most recent fiscal periods. Used to populate `TemporalDelta` (Fix 2) and ground the `ExecutiveSummary` with actual numbers rather than LLM estimates.

Called by:
- `transcript_mcp_node` — target ticker
- `peer_mcp_node` — all peers, concurrently via `asyncio.gather`

### `search_filings`

Searches narrative text inside 10-K and 10-Q filings.

```python
await search_filings(
    ticker="AMZN",
    query="material changes in risk factors",
    diff_only=True   # only passages that changed from prior year
)
```

`diff_only=True` is the key parameter. It returns only disclosure passages that changed from the prior year's filing, which is how analysts catch material risk factor shifts without reading entire 10-Ks. This is Fix 2 applied to text rather than numbers.

Called by `transcript_mcp_node`.

### `get_company_facts`

Fetches sector, industry, and segment classification.

```python
await get_company_facts(ticker="AMZN")
```

Returns sector, industry, business description, and major revenue segments from SEC filings. Used by `peer_selector` to ground the peer selection prompt — the LLM is told the target's actual SEC-reported segments and asked to pick peers that overlap in at least two.

Called by `peer_selector`.

---

## How the subprocess works

`mcp/edgar_tools.py` defines:

```python
_MCP_SERVER = StdioServerParameters(
    command=sys.executable,
    args=["-c", "from edgar.ai.mcp import main; main()"],
    env={"EDGAR_IDENTITY": settings.edgar_identity},
)
```

Every call to `_call_tool()` opens a fresh `stdio_client` context, initialises the MCP session, calls the tool, and closes the subprocess. This is intentionally stateless — no persistent connection to manage.

---

## Fail-open behaviour

Both `transcript_mcp_node` and `peer_mcp_node` catch all `Exception`s and continue with `None` values rather than crashing the graph. Downstream agents check for `None` and fall back to transcript RAG chunks only. This means:

- A rate-limited EDGAR response degrades quality, not availability.
- The graph still produces a report; it just won't have SEC-grounded financials.

The error is logged at `ERROR` level and is visible in the feedback store snapshot for that run.

---

## LangChain tool wrappers

`tools/edgar_mcp_client.py` exposes the same three tools as LangChain `@tool`-decorated functions. These are **not** used by the current graph nodes (which call `mcp/edgar_tools.py` directly), but are available for use with `bind_tools()` or a `ToolNode` if the architecture evolves toward fully agentic MCP tool calling.

```python
from earnings_research_agent.tools.edgar_mcp_client import EDGAR_TOOLS

llm.bind_tools(EDGAR_TOOLS)
```
