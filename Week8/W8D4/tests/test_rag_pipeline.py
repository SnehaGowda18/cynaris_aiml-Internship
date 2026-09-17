"""
Tests for the RAG pipeline.
"""

from rag_pipeline import (
    retrieve_documents,
    generate_answer,
    query_rag,
)


def test_retrieve_documents():
    """Test document retrieval."""

    results = retrieve_documents("What is RAG?")

    assert isinstance(results, list)
    assert len(results) > 0


def test_retrieve_chromadb_information():
    """Test retrieval of ChromaDB information."""

    results = retrieve_documents("What is ChromaDB?")

    assert len(results) > 0
    assert any("ChromaDB" in result for result in results)


def test_generate_answer_with_context():
    """Test answer generation when context exists."""

    documents = [
        "RAG combines document retrieval with language generation."
    ]

    answer = generate_answer(
        "What is RAG?",
        documents,
    )

    assert isinstance(answer, str)
    assert len(answer) > 0
    assert "RAG" in answer


def test_generate_answer_without_context():
    """Test answer generation without retrieved documents."""

    answer = generate_answer(
        "Unknown question",
        [],
    )

    assert "could not find" in answer.lower()


def test_query_rag():
    """Test the complete RAG workflow."""

    result = query_rag("What is RAG?")

    assert isinstance(result, dict)
    assert "question" in result
    assert "answer" in result
    assert "sources" in result

    assert result["question"] == "What is RAG?"
    assert isinstance(result["sources"], list)