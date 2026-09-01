from pathlib import Path

from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.writers import DocumentWriter


DATA_DIR = Path("data")


def build_bm25_pipeline():

    document_store = InMemoryDocumentStore()

    converter = PyPDFToDocument()

    splitter = DocumentSplitter(
        split_by="sentence",
        split_length=5,
        split_overlap=1
    )

    writer = DocumentWriter(
        document_store=document_store
    )

    pipeline = Pipeline()

    pipeline.add_component("converter", converter)
    pipeline.add_component("splitter", splitter)
    pipeline.add_component("writer", writer)

    pipeline.connect(
        "converter.documents",
        "splitter.documents"
    )

    pipeline.connect(
        "splitter.documents",
        "writer.documents"
    )

    pdf_files = list(DATA_DIR.glob("*.pdf"))

    if len(pdf_files) != 5:
        raise ValueError(
            f"Expected 5 PDF files, found {len(pdf_files)}"
        )

    pipeline.run(
        {
            "converter": {
                "sources": pdf_files
            }
        }
    )

    retriever = InMemoryBM25Retriever(
        document_store=document_store
    )

    return document_store, retriever


if __name__ == "__main__":

    document_store, retriever = build_bm25_pipeline()

    print("BM25 Pipeline Ready")
    print(
        "Documents indexed:",
        document_store.count_documents()
    )

    questions = [
        "What is machine learning?",
        "What is artificial intelligence?",
        "What is deep learning?",
        "What is natural language processing?",
        "What is a neural network?",
        "What is supervised learning?",
        "What is unsupervised learning?",
        "What is reinforcement learning?",
        "What is model evaluation?",
        "What is retrieval augmented generation?"
    ]

    for question in questions:

        result = retriever.run(
            query=question,
            top_k=3
        )

        print("\nQuestion:", question)

        for i, doc in enumerate(
            result["documents"],
            start=1
        ):
            print(
                f"{i}.",
                doc.content[:250].replace("\n", " ")
            )