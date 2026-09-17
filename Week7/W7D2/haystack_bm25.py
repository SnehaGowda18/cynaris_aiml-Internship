from pathlib import Path
from pypdf import PdfReader

from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# Test questions and expected documents
# ---------------------------------------------------------

questions = [
    ("What is artificial intelligence?", "document1.pdf"),
    ("What are the types of machine learning?", "document2.pdf"),
    ("What is supervised learning?", "document2.pdf"),
    ("What are convolutional neural networks used for?", "document3.pdf"),
    ("What is deep learning?", "document3.pdf"),
    ("What is natural language processing?", "document4.pdf"),
    ("What are embeddings in NLP?", "document4.pdf"),
    ("What is generative AI?", "document5.pdf"),
    ("What is retrieval augmented generation?", "document5.pdf"),
    ("What are large language models?", "document5.pdf"),
]


# ---------------------------------------------------------
# 1. Load PDF documents
# ---------------------------------------------------------

def load_pdfs():
    documents = []

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))

    print(f"PDF files found: {len(pdf_files)}")

    for pdf_path in pdf_files:
        reader = PdfReader(str(pdf_path))

        text = ""

        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"

        documents.append(
            Document(
                content=text,
                meta={"source": pdf_path.name}
            )
        )

    print(f"Documents extracted: {len(documents)}")

    return documents


# ---------------------------------------------------------
# 2. Load documents
# ---------------------------------------------------------

documents = load_pdfs()


# ---------------------------------------------------------
# 3. Create Haystack DocumentStore
# ---------------------------------------------------------

document_store = InMemoryDocumentStore()

document_store.write_documents(documents)

print(f"Documents stored: {document_store.count_documents()}")


# ---------------------------------------------------------
# 4. Create BM25 Retriever
# ---------------------------------------------------------

retriever = InMemoryBM25Retriever(
    document_store=document_store,
    top_k=3
)


# ---------------------------------------------------------
# 5. Run 10 questions
# ---------------------------------------------------------

correct = 0
results = []

print("\n" + "=" * 70)
print("BM25 RETRIEVAL EVALUATION")
print("=" * 70)

for question_number, (question, expected) in enumerate(
    questions, start=1
):

    print("\n" + "=" * 70)
    print(f"Question {question_number}: {question}")
    print(f"Expected: {expected}")
    print("=" * 70)

    result = retriever.run(query=question)

    retrieved_documents = result["documents"]

    found_expected = False

    for rank, doc in enumerate(retrieved_documents, start=1):

        source = doc.meta.get("source", "unknown")

        score = doc.score if doc.score is not None else 0.0

        print(
            f"Rank {rank} | "
            f"Score: {score:.4f} | "
            f"Source: {source}"
        )

        if source == expected:
            found_expected = True

    if found_expected:
        correct += 1
        print("Correct retrieval: YES")
    else:
        print("Correct retrieval: NO")

    results.append({
        "question": question,
        "expected": expected,
        "correct": found_expected
    })


# ---------------------------------------------------------
# 6. Calculate precision
# ---------------------------------------------------------

total_questions = len(questions)

precision = (correct / total_questions) * 100


print("\n" + "=" * 70)
print("BM25 RETRIEVAL EVALUATION")
print("=" * 70)

print(f"Correct retrievals: {correct}")
print(f"Total questions: {total_questions}")
print(f"Precision: {precision:.2f}%")

print("=" * 70)


# ---------------------------------------------------------
# 7. Save output evidence
# ---------------------------------------------------------

output_file = OUTPUT_DIR / "bm25_results.txt"

with open(output_file, "w", encoding="utf-8") as file:

    file.write("BM25 RETRIEVAL EVALUATION\n")
    file.write("=" * 70 + "\n\n")

    for number, result in enumerate(results, start=1):

        file.write(
            f"Question {number}: "
            f"{result['question']}\n"
        )

        file.write(
            f"Expected: {result['expected']}\n"
        )

        file.write(
            f"Correct retrieval: "
            f"{'YES' if result['correct'] else 'NO'}\n\n"
        )

    file.write("=" * 70 + "\n")

    file.write(
        f"Correct retrievals: {correct}\n"
    )

    file.write(
        f"Total questions: {total_questions}\n"
    )

    file.write(
        f"Precision: {precision:.2f}%\n"
    )


print(f"\nResults saved to: {output_file}")