"""Centralised configuration loaded from environment variables.

Uses pydantic-settings so every value is typed, validated at startup,
and never hardcoded. Load a .env file for local development.

Usage:
    from earnings_research_agent.utils.config import settings
    settings.pinecone_api_key
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration for the earnings research agent."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------
    # Gemini (primary LLM for all agents)
    # ------------------------------------------------------------------
    gemini_api_key: str = Field(..., alias="GEMINI_API_KEY")
    gemini_model: str = Field("gemini-2.5-flash", alias="GEMINI_MODEL")

    # ------------------------------------------------------------------
    # OpenAI (embeddings only — text-embedding-3-small)
    # ------------------------------------------------------------------
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    openai_embed_model: str = Field(
        "text-embedding-3-small", alias="OPENAI_EMBED_MODEL"
    )

    # ------------------------------------------------------------------
    # Pinecone (transcript vector store)
    # ------------------------------------------------------------------
    pinecone_api_key: str = Field(..., alias="PINECONE_API_KEY")
    pinecone_index: str = Field("earnings-research-agent", alias="PINECONE_INDEX")

    # ------------------------------------------------------------------
    # EdgarTools MCP server (stdio transport, open-source library)
    # EDGAR_IDENTITY must be "Full Name email@example.com" per SEC rules.
    # ------------------------------------------------------------------
    edgar_identity: str = Field(..., alias="EDGAR_IDENTITY")

    # ------------------------------------------------------------------
    # LangGraph checkpointer (PostgresSaver in production)
    # Leave empty to fall back to InMemorySaver (dev/test only).
    # ------------------------------------------------------------------
    postgres_url: str = Field("", alias="POSTGRES_URL")

    # ------------------------------------------------------------------
    # Agentic RAG limits
    # ------------------------------------------------------------------
    rag_top_k: int = Field(10, alias="RAG_TOP_K")
    rag_max_retrieval_attempts: int = Field(2, alias="RAG_MAX_RETRIEVAL_ATTEMPTS")

    # ------------------------------------------------------------------
    # Feedback store path (JSONL file, append-only)
    # ------------------------------------------------------------------
    feedback_store_path: str = Field(
        "data/feedback_store/feedback.jsonl", alias="FEEDBACK_STORE_PATH"
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()


# Module-level alias for convenience.
settings = get_settings()