import os
import sys
import json
import asyncio
import logging
from dotenv import load_dotenv
from openai import AsyncOpenAI
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

logging.getLogger("edgar").setLevel(logging.ERROR)

load_dotenv()

client = AsyncOpenAI(
    api_key=os.environ["XAI_API_KEY"],
    base_url="https://api.x.ai/v1",
)

MCP_SERVER = StdioServerParameters(
    command=sys.executable,
    args=["-c", "import logging; logging.getLogger('edgar').setLevel(logging.ERROR); from edgar.ai.mcp import main; main()"],
    env={**os.environ, "EDGAR_IDENTITY": os.environ["EDGAR_IDENTITY"]},
)


def mcp_to_openai_tool(tool) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema,
        },
    }


async def run(query: str) -> str:
    async with stdio_client(MCP_SERVER) as (read, write):
        async with ClientSession(read, write) as mcp:
            await mcp.initialize()

            tools_result = await mcp.list_tools()
            tools = [mcp_to_openai_tool(t) for t in tools_result.tools]
            messages = [{"role": "user", "content": query}]

            while True:
                response = await client.chat.completions.create(
                    model="grok-3",
                    max_tokens=4096,
                    tools=tools,
                    messages=messages,
                )

                choice = response.choices[0]
                messages.append(choice.message.model_dump(exclude_unset=False))

                if choice.finish_reason == "stop":
                    return choice.message.content or ""

                if choice.finish_reason == "tool_calls":
                    for call in choice.message.tool_calls:
                        result = await mcp.call_tool(
                            call.function.name,
                            json.loads(call.function.arguments),
                        )
                        content = "\n".join(
                            c.text for c in result.content if hasattr(c, "text")
                        )
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": content,
                        })


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "What are Apple's latest quarterly earnings?"
    print(asyncio.run(run(query)))
