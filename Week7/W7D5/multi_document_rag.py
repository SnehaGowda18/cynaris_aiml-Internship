from pathlib import Path

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding


# ============================================================
# Configuration
# ============================================================

DOCUMENT_DIR = Path("documents")
OUTPUT_DIR = Path("outputs")
OUTPUT_FILE = OUTPUT_DIR / "rag_output.txt"

LLM_MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "nomic-embed-text"


# ============================================================
# Ollama LLM Configuration
# ============================================================

Settings.llm = Ollama(
    model=LLM_MODEL,
    request_timeout=180.0,
    context_window=2048,
    temperature=0.2
)


# ============================================================
# Ollama Embedding Configuration
# ============================================================

Settings.embed_model = OllamaEmbedding(
    model_name=EMBEDDING_MODEL
)


# ============================================================
# Load Documents
# ============================================================

def load_documents():
    print("\nLoading documents...")

    if not DOCUMENT_DIR.exists():
        raise FileNotFoundError(
            f"Document directory not found: {DOCUMENT_DIR}"
        )

    documents = SimpleDirectoryReader(
        input_dir=str(DOCUMENT_DIR)
    ).load_data()

    print(f"Loaded {len(documents)} documents.")

    return documents


# ============================================================
# Build Vector Index
# ============================================================

def build_index(documents):
    print("\nCreating vector index...")

    index = VectorStoreIndex.from_documents(
        documents,
        show_progress=True
    )

    print("Vector index created successfully.")

    return index


# ============================================================
# Run RAG Queries
# ============================================================

def run_queries(index):

    query_engine = index.as_query_engine(
        similarity_top_k=2
    )

    questions = [
        "What is Python used for?",
        "What are the main types of machine learning?",
        "What is Retrieval Augmented Generation?",
        "What are common cybersecurity threats?",
        "How does a RAG system generate answers?"
    ]

    print("\n" + "=" * 70)
    print("MULTI-DOCUMENT RAG RESULTS")
    print("=" * 70)

    results = []

    for i, question in enumerate(questions, 1):

        print("\n" + "-" * 70)
        print(f"Question {i}: {question}")
        print("-" * 70)

        try:
            response = query_engine.query(question)

            answer = str(response)

            print("Answer:")
            print(answer)

            results.append(
                f"Question {i}: {question}\n"
                f"Answer: {answer}\n"
            )

        except Exception as error:

            print("Error:")
            print(error)

            results.append(
                f"Question {i}: {question}\n"
                f"Error: {error}\n"
            )

    return results


# ============================================================
# Save Results
# ============================================================

def save_results(results):

    OUTPUT_DIR.mkdir(exist_ok=True)

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write("W7D5 MULTI-DOCUMENT RAG OUTPUT\n")
        file.write("=" * 70 + "\n\n")

        file.write(
            f"LLM Model: {LLM_MODEL}\n"
            f"Embedding Model: {EMBEDDING_MODEL}\n\n"
        )

        for result in results:
            file.write(result)
            file.write("\n" + "-" * 70 + "\n")

    print(f"\nOutput saved to: {OUTPUT_FILE}")


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("MULTI-DOCUMENT RAG SYSTEM")
    print("=" * 70)

    print(f"LLM Model       : {LLM_MODEL}")
    print(f"Embedding Model : {EMBEDDING_MODEL}")
    print("Context Window  : 2048")

    documents = load_documents()

    index = build_index(documents)

    results = run_queries(index)

    save_results(results)

    print("\n" + "=" * 70)
    print("RAG EXECUTION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()