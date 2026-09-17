from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


DATA_FILE = Path("data/documents.txt")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

COLLECTION_NAME = "w8d2_rag"


def load_documents(chunk_size=500, chunk_overlap=50):
    """Load and split the knowledge base."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    text = DATA_FILE.read_text(encoding="utf-8")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    documents = splitter.create_documents([text])

    return documents


def create_vector_store(
    chunk_size=500,
    chunk_overlap=50,
):
    """Create a ChromaDB vector store."""

    documents = load_documents(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
    )

    return vector_store


def retrieve_documents(
    question,
    k=3,
    chunk_size=500,
):
    """Retrieve relevant documents from ChromaDB."""

    vector_store = create_vector_store(
        chunk_size=chunk_size
    )

    documents = vector_store.similarity_search(
        question,
        k=k,
    )

    return documents


def search(question, k=3, chunk_size=500):
    """Return retrieved context as strings."""

    documents = retrieve_documents(
        question,
        k=k,
        chunk_size=chunk_size,
    )

    return [
        document.page_content
        for document in documents
    ]


if __name__ == "__main__":

    question = "What is Retrieval Augmented Generation?"

    print("=" * 60)
    print("W8D2 RAG PIPELINE")
    print("=" * 60)

    print(f"\nQuestion: {question}")

    contexts = search(
        question,
        k=3,
        chunk_size=500,
    )

    print(f"\nRetrieved contexts: {len(contexts)}")

    for index, context in enumerate(contexts, 1):

        print(f"\n--- Context {index} ---")
        print(context)

    print("\n" + "=" * 60)
    print("RAG retrieval completed successfully.")
    print("=" * 60)