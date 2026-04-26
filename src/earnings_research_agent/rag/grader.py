"""RAG Relevance Grader.

Evaluates retrieved Pinecone chunks to determine if they are relevant
to the current query. This acts as a filter before passing context to
the generation agents.
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from earnings_research_agent.utils.config import settings
from earnings_research_agent.utils.logging import get_logger

logger = get_logger(__name__)

class ChunkGrade(BaseModel):
    """Binary score for relevance check on a retrieved chunk."""
    score: str = Field(
        description="Relevance score 'yes' or 'no'. 'yes' means it contains useful facts for the query."
    )
    reasoning: str = Field(
        description="A brief, one-sentence explanation for the score."
    )

def get_grader() -> ChatGoogleGenerativeAI:
    """Initialize the LLM specifically for the grading task."""
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=0.0, # Strictly deterministic for grading
    ).with_structured_output(ChunkGrade)

def grade_chunk(query: str, chunk_text: str) -> bool:
    """Grade a single chunk against the current query."""
    system_prompt = """You are a strict relevance grader for an equity research RAG system.
    Your job is to assess if the retrieved document chunk contains ANY information 
    relevant to the user's query. 
    
    If the document contains keywords, metrics, or context related to the query, grade it as 'yes'.
    If it is generic boilerplate or completely unrelated, grade it as 'no'.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Query: {query}\n\nDocument Chunk: {chunk}")
    ])

    chain = prompt | get_grader()

    try:
        result: ChunkGrade = chain.invoke({"query": query, "chunk": chunk_text})
        is_relevant = result.score.lower() == "yes"
        
        if not is_relevant:
            logger.debug("Chunk rejected. Reasoning: %s", result.reasoning)
            
        return is_relevant
    except Exception as e:
        logger.error("Grader failed to evaluate chunk: %s", e)
        # Fail open: if the grader breaks, keep the chunk rather than losing data
        return True