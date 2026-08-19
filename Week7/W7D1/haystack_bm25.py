from pathlib import Path

from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.writers import DocumentWriter
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore


# --------------------------------------------------
# Paths
# --------------------------------------------------

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")

OUTPUT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# Questions
# --------------------------------------------------

questions = [
    "What is artificial intelligence?",
    "What are the types of machine learning?",
    "What is supervised learning?",
    "What are convolutional neural networks used for?",
    "What is deep learning?",
    "What is natural language processing?",
    "What are embeddings in NLP?",
    "What is generative AI?",
    "What is retrieval augmented generation?",
    "What are large language models?",
]


# --------------------------------------------------
# 1. Create Document Store
# --------------------------------------------------

document_store = InMemoryDocumentStore()


# --------------------------------------------------
# 2. Convert PDFs
# --------------------------------------------------

converter = PyPDFToDocument()

pdf_files = list(DATA_DIR.glob("*.pdf"))

if len(pdf_files) != 5:
    raise ValueError(
        f"Expected 5 PDF files, but found {len(pdf_files)}"
    )

conversion_result = converter.run(
    sources=pdf_files
)

documents = conversion_result["documents"]

print(f"PDF files found: {len(pdf_files)}")
print(f"Documents extracted: {len(documents)}")


# --------------------------------------------------
# 3. Store documents
# --------------------------------------------------

writer = DocumentWriter(
    document_store=document_store
)

writer.run(
    documents=documents
)

print(
    f"Documents stored: "
    f"{document_store.count_documents()}"
)


# --------------------------------------------------
# 4. BM25 Retriever
# --------------------------------------------------

retriever = InMemoryBM25Retriever(
    document_store=document_store,
    top_k=3
)


# --------------------------------------------------
# 5. Build Pipeline
# --------------------------------------------------

pipeline = Pipeline()

pipeline.add_component(
    "retriever",
    retriever
)


# --------------------------------------------------
# 6. Run 10 Questions
# --------------------------------------------------

output_file = OUTPUT_DIR / "bm25_results.txt"

with open(output_file, "w", encoding="utf-8") as file:

    file.write("BM25 RETRIEVAL RESULTS\n")
    file.write("=" * 70 + "\n\n")

    for number, question in enumerate(
        questions,
        start=1
    ):

        result = pipeline.run(
            {
                "retriever": {
                    "query": question
                }
            }
        )

        retrieved_documents = (
            result["retriever"]["documents"]
        )

        print("\n" + "=" * 70)
        print(f"Question {number}: {question}")
        print("=" * 70)

        file.write(
            f"Question {number}: {question}\n"
        )
        file.write("-" * 70 + "\n")

        for rank, document in enumerate(
            retrieved_documents,
            start=1
        ):

            score = document.score or 0

            content = document.content.replace(
                "\n",
                " "
            )

            content = content[:500]

            print(
                f"\nRank {rank} | "
                f"Score: {score:.4f}"
            )

            print(content)

            file.write(
                f"\nRank {rank}\n"
            )

            file.write(
                f"Score: {score:.4f}\n"
            )

            file.write(
                f"Content: {content}\n"
            )

        file.write("\n\n")


print("\nBM25 evaluation completed.")
print(f"Results saved to: {output_file}")