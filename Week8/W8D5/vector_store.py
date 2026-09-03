"""
Vector store for the Local AI Research Assistant.

Uses:
- ChromaDB for persistent document storage
- TF-IDF for lightweight local embeddings
"""

import os

import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer


# Persistent ChromaDB directory
CHROMA_PATH = "./chroma_db"

# ChromaDB collection name
COLLECTION_NAME = "research_documents"


def load_documents():
    """Load research documents from the data folder."""

    documents = []
    ids = []

    data_folder = "data"

    # Check that the data directory exists
    if not os.path.exists(data_folder):
        raise FileNotFoundError(
            f"Data folder not found: {data_folder}"
        )

    # Read all TXT files
    for filename in os.listdir(data_folder):

        if filename.endswith(".txt"):

            file_path = os.path.join(
                data_folder,
                filename
            )

            with open(
                file_path,
                "r",
                encoding="utf-8"
            ) as file:
                text = file.read()

            # Ignore empty documents
            if text.strip():

                documents.append(text)
                ids.append(filename)

    return documents, ids


def create_vector_store():
    """Create or load the ChromaDB collection with TF-IDF embeddings."""

    # Create persistent ChromaDB client
    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    # Get or create collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    # Load local research documents
    documents, ids = load_documents()

    if not documents:
        return collection, None

    # Create lightweight local TF-IDF embeddings
    vectorizer = TfidfVectorizer()

    embeddings = vectorizer.fit_transform(
        documents
    ).toarray().tolist()

    # Store/update documents in ChromaDB
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings
    )

    return collection, vectorizer


def search_documents(query, top_k=3):
    """
    Search research documents using ChromaDB.

    Returns:
        list[str]: Retrieved document contents.
    """

    collection, vectorizer = create_vector_store()

    # No documents available
    if vectorizer is None:
        return []

    # Convert query into TF-IDF embedding
    query_embedding = vectorizer.transform(
        [query]
    ).toarray().tolist()

    # Search ChromaDB
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    # ChromaDB returns:
    # {
    #     "ids": [[...]],
    #     "documents": [[...]]
    # }
    #
    # Return only the first result group.
    if not results.get("documents"):
        return []

    if not results["documents"]:
        return []

    return results["documents"][0]