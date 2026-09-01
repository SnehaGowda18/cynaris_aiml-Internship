from pathlib import Path

from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore

from haystack_integrations.components.embedders.sentence_transformers import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)

from haystack.components.retrievers.in_memory import (
    InMemoryEmbeddingRetriever,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 3


# ============================================================
# EVALUATION QUESTIONS
# ============================================================

QUESTIONS = [
    "What is machine learning?",
    "What is artificial intelligence?",
    "What is deep learning?",
    "What is natural language processing?",
    "What is a neural network?",
    "What is supervised learning?",
    "What is unsupervised learning?",
    "What is reinforcement learning?",
    "What is model evaluation?",
    "What is retrieval augmented generation?",
]


# ============================================================
# BUILD DENSE RETRIEVAL PIPELINE
# ============================================================

def build_dense_pipeline():

    print("=" * 70)
    print("BUILDING DENSE RETRIEVAL PIPELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # Document Store
    # --------------------------------------------------------

    document_store = InMemoryDocumentStore()

    # --------------------------------------------------------
    # PDF Converter
    # --------------------------------------------------------

    converter = PyPDFToDocument()

    # --------------------------------------------------------
    # Document Splitter
    # --------------------------------------------------------

    splitter = DocumentSplitter(
        split_by="sentence",
        split_length=5,
        split_overlap=1,
    )

    # --------------------------------------------------------
    # Sentence Transformer Document Embedder
    # --------------------------------------------------------

    print("\nLoading embedding model:")
    print(MODEL_NAME)

    document_embedder = SentenceTransformersDocumentEmbedder(
        model=MODEL_NAME
    )

    document_embedder.warm_up()

    # --------------------------------------------------------
    # Document Writer
    # --------------------------------------------------------

    writer = DocumentWriter(
        document_store=document_store
    )

    # --------------------------------------------------------
    # Indexing Pipeline
    # --------------------------------------------------------

    indexing_pipeline = Pipeline()

    indexing_pipeline.add_component(
        "converter",
        converter
    )

    indexing_pipeline.add_component(
        "splitter",
        splitter
    )

    indexing_pipeline.add_component(
        "embedder",
        document_embedder
    )

    indexing_pipeline.add_component(
        "writer",
        writer
    )

    # --------------------------------------------------------
    # Connections
    # --------------------------------------------------------

    indexing_pipeline.connect(
        "converter.documents",
        "splitter.documents"
    )

    indexing_pipeline.connect(
        "splitter.documents",
        "embedder.documents"
    )

    indexing_pipeline.connect(
        "embedder.documents",
        "writer.documents"
    )

    # --------------------------------------------------------
    # Find PDF files
    # --------------------------------------------------------

    pdf_files = list(DATA_DIR.glob("*.pdf"))

    print("\nPDF directory:")
    print(DATA_DIR)

    print("\nPDF files found:")

    for pdf in pdf_files:
        print(" -", pdf.name)

    # --------------------------------------------------------
    # Validate PDF count
    # --------------------------------------------------------

    if len(pdf_files) != 5:

        raise ValueError(
            f"\nExpected 5 PDF files, found {len(pdf_files)}.\n"
            f"Please place exactly 5 valid PDF files inside:\n"
            f"{DATA_DIR}"
        )

    # --------------------------------------------------------
    # Index Documents
    # --------------------------------------------------------

    print("\nIndexing documents...")

    indexing_pipeline.run(
        {
            "converter": {
                "sources": pdf_files
            }
        }
    )

    # --------------------------------------------------------
    # Query Embedder
    # --------------------------------------------------------

    print("\nPreparing query embedding model...")

    text_embedder = SentenceTransformersTextEmbedder(
        model=MODEL_NAME
    )

    text_embedder.warm_up()

    # --------------------------------------------------------
    # Dense Retriever
    # --------------------------------------------------------

    retriever = InMemoryEmbeddingRetriever(
        document_store=document_store
    )

    print("\nDense Retrieval Pipeline Ready")

    print(
        "Documents indexed:",
        document_store.count_documents()
    )

    return (
        document_store,
        text_embedder,
        retriever
    )


# ============================================================
# RUN QUESTIONS
# ============================================================

def run_questions(
    text_embedder,
    retriever
):

    print("\n")
    print("=" * 70)
    print("DENSE RETRIEVAL RESULTS")
    print("=" * 70)

    all_results = {}

    for number, question in enumerate(
        QUESTIONS,
        start=1
    ):

        # ----------------------------------------------------
        # Convert query into embedding
        # ----------------------------------------------------

        embedding_result = text_embedder.run(
            text=question
        )

        query_embedding = embedding_result[
            "embedding"
        ]

        # ----------------------------------------------------
        # Retrieve top 3 documents
        # ----------------------------------------------------

        result = retriever.run(
            query_embedding=query_embedding,
            top_k=TOP_K
        )

        documents = result["documents"]

        all_results[question] = documents

        # ----------------------------------------------------
        # Display results
        # ----------------------------------------------------

        print("\n")
        print(
            f"Question {number}: {question}"
        )

        print("-" * 70)

        for rank, document in enumerate(
            documents,
            start=1
        ):

            score = document.score

            print(
                f"\nRank {rank}"
            )

            print(
                f"Similarity Score: {score:.4f}"
            )

            print(
                "Document:",
                document.meta.get(
                    "file_path",
                    "Unknown"
                )
            )

            content = document.content

            if len(content) > 400:
                content = content[:400] + "..."

            print(
                "Content:",
                content.replace(
                    "\n",
                    " "
                )
            )

    return all_results


# ============================================================
# MAIN
# ============================================================

def main():

    try:

        (
            document_store,
            text_embedder,
            retriever
        ) = build_dense_pipeline()

        run_questions(
            text_embedder,
            retriever
        )

        print("\n")
        print("=" * 70)
        print("DENSE RETRIEVAL COMPLETED SUCCESSFULLY")
        print("=" * 70)

        print(
            "\nTotal documents:",
            document_store.count_documents()
        )

        print(
            "Total questions:",
            len(QUESTIONS)
        )

        print(
            "Top-K:",
            TOP_K
        )

    except Exception as error:

        print("\n")
        print("=" * 70)
        print("ERROR")
        print("=" * 70)

        print(
            f"\n{type(error).__name__}: {error}"
        )

        raise


if __name__ == "__main__":
    main()