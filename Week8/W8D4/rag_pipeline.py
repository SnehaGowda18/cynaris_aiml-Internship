"""
RAG Pipeline Module

Handles document storage, retrieval and question answering.
"""

from typing import List


# Sample knowledge base for testing the RAG API.
DOCUMENTS = [
    {
        "id": 1,
        "content": (
            "Retrieval Augmented Generation (RAG) combines "
            "document retrieval with language generation."
        ),
    },
    {
        "id": 2,
        "content": (
            "ChromaDB is a vector database commonly used to "
            "store and retrieve embeddings."
        ),
    },
    {
        "id": 3,
        "content": (
            "FastAPI is a Python framework used to build "
            "high-performance APIs."
        ),
    },
    {
        "id": 4,
        "content": (
            "Ragas can be used to evaluate RAG applications "
            "using metrics such as faithfulness and answer relevance."
        ),
    },
]


def retrieve_documents(query: str, top_k: int = 3) -> List[str]:
    """
    Retrieve documents related to the query.

    A simple keyword-based retrieval method is used here
    so that the application can be tested without requiring
    an external LLM or embedding service.
    """

    query_words = set(query.lower().split())

    scored_documents = []

    for document in DOCUMENTS:
        content = document["content"]
        content_words = set(content.lower().split())

        score = len(query_words.intersection(content_words))

        scored_documents.append((score, content))

    scored_documents.sort(reverse=True, key=lambda item: item[0])

    results = [
        content
        for score, content in scored_documents[:top_k]
        if score > 0
    ]

    return results


def generate_answer(query: str, documents: List[str]) -> str:
    """
    Generate an answer using the retrieved context.

    In a production system, this function can be connected
    to an LLM such as Ollama, OpenAI, or another model.
    """

    if not documents:
        return (
            "I could not find relevant information "
            "in the knowledge base."
        )

    context = " ".join(documents)

    return (
        f"Based on the retrieved information: {context}"
    )


def query_rag(query: str) -> dict:
    """
    Execute the complete RAG workflow.

    Steps:
    1. Retrieve relevant documents.
    2. Generate an answer using the retrieved context.
    3. Return the answer and sources.
    """

    documents = retrieve_documents(query)

    answer = generate_answer(query, documents)

    return {
        "question": query,
        "answer": answer,
        "sources": documents,
    }