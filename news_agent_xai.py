import os
import sys
from datetime import date
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["XAI_API_KEY"],
    base_url="https://api.x.ai/v1",
)


def get_news(ticker: str) -> dict:
    response = client.responses.create(
        model="grok-4",
        input=[{
            "role": "user",
            "content": (
                f"Today is {date.today().strftime('%B %d, %Y')}. "
                f"Find the most recent news for {ticker} stock. "
                "Include earnings updates, analyst ratings, product announcements, "
                "and any other market-moving events. "
                "For each item include the headline, source, date, and a brief summary."
            ),
        }],
        tools=[{"type": "web_search"}],
    )

    text = ""
    for item in response.output:
        if item.type == "message":
            for block in item.content:
                if block.type == "output_text":
                    text += block.text

    citations = getattr(response, "citations", []) or []

    return {"ticker": ticker, "summary": text, "citations": citations}


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    result = get_news(ticker)
    print(result["summary"])
    if result["citations"]:
        print("\nSources:")
        for c in result["citations"]:
            print(f"  - {c}")
