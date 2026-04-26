"""LLM factory for the earnings research agent.

Grok (xAI) is the default. Gemini is the fallback when XAI_API_KEY is absent.

Model assignment by role:
    fast     → grok-3-mini       simple tasks: grading, query rewriting
    standard → grok-3-fast       medium tasks: peer selection, report editing
    powerful → grok-3            heavy tasks: transcript and peer analysis
"""
from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from earnings_research_agent.utils.config import settings

_GROK_MODELS = {
    "fast": "grok-3-mini",
    "standard": "grok-3-fast",
    "powerful": "grok-3",
}

XAI_BASE_URL = "https://api.x.ai/v1"


def get_llm(role: str = "standard", temperature: float = 0.0):
    """Return the best available LLM for the given role.

    Uses Grok if XAI_API_KEY is configured, otherwise falls back to Gemini.
    """
    if settings.xai_api_key:
        return ChatOpenAI(
            model=_GROK_MODELS[role],
            api_key=settings.xai_api_key,
            base_url=XAI_BASE_URL,
            temperature=temperature,
        )
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=temperature,
    )
