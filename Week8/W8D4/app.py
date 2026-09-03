"""
Production RAG API

FastAPI application exposing health-check and RAG query endpoints.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_pipeline import query_rag


app = FastAPI(
    title="Production RAG API",
    description="RAG API with documentation, testing and code review.",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    """Request model for the RAG query endpoint."""

    question: str = Field(
        ...,
        min_length=2,
        max_length=500,
        description="Question to ask the RAG system.",
    )


class QueryResponse(BaseModel):
    """Response model returned by the RAG API."""

    question: str
    answer: str
    sources: list[str]


@app.get("/")
def root():
    """Return basic API information."""

    return {
        "message": "Production RAG API is running",
        "version": "1.0.0",
        "status": "healthy",
    }


@app.get("/health")
def health_check():
    """Health-check endpoint."""

    return {
        "status": "healthy",
        "service": "rag-api",
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Query the RAG system.

    The endpoint validates the user's question and
    executes the RAG pipeline.
    """

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:
        result = query_rag(question)

        return result

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"RAG processing failed: {error}",
        ) from error