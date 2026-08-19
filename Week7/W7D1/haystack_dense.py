from pathlib import Path

from haystack import Document, Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Paths
# --------------------------------------------------

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")

OUTPUT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# Same 10 questions used for BM25
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
# Expected documents
# --------------------------------------------------

expected_documents = [
    "document1.pdf",
    "document2.pdf",
    "document2.pdf",
    "document3.pdf",
    "document3.pdf",
    "document4.pdf",
    "document4.pdf",
    "document5.pdf",
    "document5.pdf",
    "document5.pdf",
]


# --------------------------------------------------
# 1. Create document store
# --------------------------------------------------

document_store = InMemoryDocumentStore()


# --------------------------------------------------
# 2. Convert PDFs
# --------------------------------------------------

converter = PyPDFToDocument()

pdf_files = list(DATA_DIR.glob("*.pdf"))

if len(pdf_files) != 5:
    raise ValueError(
        f"Expected 5 PDF files, found {len(pdf_files)}"
    )

conversion_result = converter.run(
    sources=pdf_files
)

documents = conversion_result["documents"]

print(f"PDF files found: {len(pdf_files)}")
print(f"Documents extracted: {len(documents)}")


# --------------------------------------------------
# 3. Load embedding model
# --------------------------------------------------

print("\nLoading embedding model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# --------------------------------------------------
# 4. Generate document embeddings
# --------------------------------------------------

haystack_documents = []

for document in documents:

    embedding = model.encode(
        document.content,
        normalize_embeddings=True
    ).tolist()

    source = document.meta.get(
        "file_path",
        "Unknown"
    )

    haystack_document = Document(
        content=document.content,
        embedding=embedding,
        meta={
            "file_path": source
        }
    )

    haystack_documents.append(
        haystack_document
    )


# --------------------------------------------------
# 5. Store embedded documents
# --------------------------------------------------

document_store.write_documents(
    haystack_documents
)

print(
    f"Documents stored: "
    f"{document_store.count_documents()}"
)


# --------------------------------------------------
# 6. Create dense retriever
# --------------------------------------------------

retriever = InMemoryEmbeddingRetriever(
    document_store=document_store,
    top_k=3
)


# --------------------------------------------------
# 7. Create pipeline
# --------------------------------------------------

pipeline = Pipeline()

pipeline.add_component(
    "retriever",
    retriever
)


# --------------------------------------------------
# 8. Run 10 questions
# --------------------------------------------------

output_file = OUTPUT_DIR / "dense_results.txt"

correct = 0

with open(
    output_file,
    "w",
    encoding="utf-8"
) as file:

    file.write("DENSE RETRIEVAL RESULTS\n")
    file.write("=" * 70 + "\n\n")

    for number, (question, expected) in enumerate(
        zip(questions, expected_documents),
        start=1
    ):

        query_embedding = model.encode(
            question,
            normalize_embeddings=True
        ).tolist()

        result = pipeline.run(
            {
                "retriever": {
                    "query_embedding": query_embedding
                }
            }
        )

        retrieved_documents = (
            result["retriever"]["documents"]
        )

        print("\n" + "=" * 70)
        print(f"Question {number}: {question}")
        print(f"Expected: {expected}")
        print("=" * 70)

        file.write(
            f"Question {number}: {question}\n"
        )

        file.write(
            f"Expected: {expected}\n"
        )

        found = False

        for rank, document in enumerate(
            retrieved_documents,
            start=1
        ):

            score = document.score or 0

            source = document.meta.get(
                "file_path",
                "Unknown"
            )

            content = document.content.replace(
                "\n",
                " "
            )

            content = content[:400]

            print(
                f"Rank {rank} | "
                f"Score: {score:.4f} | "
                f"Source: {source}"
            )

            file.write(
                f"Rank {rank} | "
                f"Score: {score:.4f} | "
                f"Source: {source}\n"
            )

            file.write(
                f"Content: {content}\n"
            )

            if expected in str(source):
                found = True

        if found:
            correct += 1
            print("Correct retrieval: YES")
            file.write("Correct retrieval: YES\n")
        else:
            print("Correct retrieval: NO")
            file.write("Correct retrieval: NO\n")

        file.write("\n")


# --------------------------------------------------
# 9. Calculate precision
# --------------------------------------------------

precision = (
    correct / len(questions)
) * 100

with open(
    output_file,
    "a",
    encoding="utf-8"
) as file:

    file.write("\n")
    file.write("=" * 70 + "\n")
    file.write("DENSE RETRIEVAL EVALUATION\n")
    file.write("=" * 70 + "\n")
    file.write(
        f"Correct retrievals: {correct}\n"
    )
    file.write(
        f"Total questions: {len(questions)}\n"
    )
    file.write(
        f"Precision: {precision:.2f}%\n"
    )


print("\n" + "=" * 70)
print("DENSE RETRIEVAL EVALUATION")
print("=" * 70)
print(f"Correct retrievals: {correct}")
print(f"Total questions: {len(questions)}")
print(f"Precision: {precision:.2f}%")
print(f"\nResults saved to: {output_file}")