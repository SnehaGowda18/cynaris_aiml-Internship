import os
import ollama
import chromadb
from pypdf import PdfReader


# ============================================================
# W6D4 - RAG Pipeline
# PDF + ChromaDB + Ollama
# ============================================================

PDF_FILE = "sample.pdf"
OUTPUT_DIR = "outputs"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "rag_output.txt")


# Create outputs folder
os.makedirs(OUTPUT_DIR, exist_ok=True)


print("=" * 60)
print("W6D4 - RAG PIPELINE")
print("PDF + ChromaDB + Ollama")
print("=" * 60)


# ============================================================
# 1. Check PDF
# ============================================================

if not os.path.exists(PDF_FILE):
    print("\nERROR: sample.pdf not found.")
    exit()

print("\n[1] PDF found.")


# ============================================================
# 2. Read PDF
# ============================================================

print("\n[2] Reading PDF...")

reader = PdfReader(PDF_FILE)

text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text + "\n"

print("Pages:", len(reader.pages))
print("Characters extracted:", len(text))


# ============================================================
# 3. Split PDF into chunks
# ============================================================

print("\n[3] Creating chunks...")

chunk_size = 500
overlap = 50

chunks = []

start = 0

while start < len(text):

    end = start + chunk_size

    chunk = text[start:end].strip()

    if chunk:
        chunks.append(chunk)

    start = end - overlap

print("Chunks created:", len(chunks))


# ============================================================
# 4. Create ChromaDB
# ============================================================

print("\n[4] Creating ChromaDB...")

client = chromadb.PersistentClient(
    path="./pdf_chroma"
)

collection = client.get_or_create_collection(
    name="w6d4_pdf_collection",
    metadata={"hnsw:space": "cosine"}
)

print("ChromaDB collection created.")


# ============================================================
# 5. Create embeddings
# ============================================================

print("\n[5] Creating embeddings...")

for i, chunk in enumerate(chunks):

    response = ollama.embeddings(
        model="nomic-embed-text",
        prompt=chunk
    )

    embedding = response["embedding"]

    collection.upsert(
        ids=[f"chunk_{i}"],
        documents=[chunk],
        embeddings=[embedding],
        metadatas=[
            {"chunk_number": i + 1}
        ]
    )

    print(f"Embedded chunk {i + 1}/{len(chunks)}")


print("\nAll embeddings stored successfully.")


# ============================================================
# 6. Ask question
# ============================================================

question = input(
    "\nEnter your question about the PDF: "
)


# ============================================================
# 7. Create question embedding
# ============================================================

print("\n[6] Searching ChromaDB...")

question_response = ollama.embeddings(
    model="nomic-embed-text",
    prompt=question
)

question_embedding = question_response["embedding"]


# ============================================================
# 8. Retrieve top 3 chunks
# ============================================================

results = collection.query(
    query_embeddings=[question_embedding],
    n_results=3
)

retrieved_chunks = results["documents"][0]


# ============================================================
# 9. Display retrieved chunks
# ============================================================

print("\n" + "=" * 60)
print("TOP 3 RETRIEVED CHUNKS")
print("=" * 60)

for i, chunk in enumerate(retrieved_chunks):

    print(f"\nCHUNK {i + 1}")
    print("-" * 60)
    print(chunk)


# ============================================================
# 10. Create context
# ============================================================

context = "\n\n".join(retrieved_chunks)


# ============================================================
# 11. Generate answer using Llama
# ============================================================

print("\n[7] Generating answer using Llama 3.2...")

prompt = f"""
You are a helpful RAG assistant.

Answer the question using ONLY the context below.

If the answer is not available in the context,
say:

"The answer was not found in the document."

Do not invent information.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""


response = ollama.chat(
    model="llama3.2:3b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)


answer = response["message"]["content"]


# ============================================================
# 12. Display answer
# ============================================================

print("\n" + "=" * 60)
print("FINAL ANSWER")
print("=" * 60)

print(answer)


# ============================================================
# 13. Save output
# ============================================================

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

    file.write("W6D4 - RAG PIPELINE OUTPUT\n")
    file.write("=" * 60 + "\n\n")

    file.write("QUESTION:\n")
    file.write(question + "\n\n")

    file.write("TOP 3 RETRIEVED CHUNKS:\n")
    file.write("=" * 60 + "\n\n")

    for i, chunk in enumerate(retrieved_chunks):

        file.write(f"CHUNK {i + 1}\n")
        file.write("-" * 60 + "\n")
        file.write(chunk + "\n\n")

    file.write("\nFINAL ANSWER:\n")
    file.write("=" * 60 + "\n")
    file.write(answer + "\n")


print("\nOutput saved to:")
print(OUTPUT_FILE)

print("\n" + "=" * 60)
print("RAG PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 60)