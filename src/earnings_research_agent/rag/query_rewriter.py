"""RAG Query Rewriter.

If the initial retrieval yields irrelevant chunks, this module rewrites 
the search query to improve Pinecone vector recall on the second pass.
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from earnings_research_agent.utils.config import settings
from earnings_research_agent.utils.logging import get_logger

logger = get_logger(__name__)

class RewrittenQuery(BaseModel):
    """Schema for the optimized vector search query."""
    query: str = Field(
        description="The optimized search query for semantic vector retrieval."
    )

def rewrite_query(original_query: str) -> str:
    """Rewrite the original query to improve semantic search matching."""
    logger.info("Rewriting query: '%s'", original_query)

    system_prompt = """You are an expert at optimizing search queries for semantic vector databases.
    The previous query returned irrelevant results. Your task is to rewrite the query to be 
    more precise for an earnings transcript context.
    
    Strategies:
    - Expand common financial acronyms.
    - Focus on core business segments and forward-looking guidance.
    - Remove conversational filler words.
    
    Return ONLY the optimized query string.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Original Query: {query}")
    ])

    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=0.2, 
    )
    
    structured_llm = llm.with_structured_output(RewrittenQuery)
    chain = prompt | structured_llm

    try:
        result: RewrittenQuery = chain.invoke({"query": original_query})
        logger.info("Query rewritten to: '%s'", result.query)
        return result.query
    except Exception as e:
        logger.error("Failed to rewrite query: %s", e)
        # Fallback to the original query if rewriting fails
        return original_query