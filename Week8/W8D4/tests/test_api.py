"""
API Tests

Tests the FastAPI endpoints and RAG functionality.
"""

from fastapi.testclient import TestClient

from app import app


client = TestClient(app)


def test_root_endpoint():
    """Test the root endpoint."""

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert "Production RAG API" in data["message"]


def test_health_endpoint():
    """Test the health endpoint."""

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["service"] == "rag-api"


def test_query_endpoint():
    """Test a valid RAG query."""

    response = client.post(
        "/query",
        json={
            "question": "What is RAG?"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["question"] == "What is RAG?"
    assert "answer" in data
    assert "sources" in data

    assert isinstance(data["sources"], list)


def test_query_chromadb():
    """Test retrieval of ChromaDB-related information."""

    response = client.post(
        "/query",
        json={
            "question": "What is ChromaDB?"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "ChromaDB" in data["answer"]


def test_empty_query():
    """Test validation of an empty question."""

    response = client.post(
        "/query",
        json={
            "question": ""
        },
    )

    assert response.status_code == 422


def test_missing_question():
    """Test request validation when question is missing."""

    response = client.post(
        "/query",
        json={}
    )

    assert response.status_code == 422


def test_long_query():
    """Test maximum question length validation."""

    response = client.post(
        "/query",
        json={
            "question": "a" * 501
        },
    )

    assert response.status_code == 422