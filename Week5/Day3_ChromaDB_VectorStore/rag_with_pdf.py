import fitz
import chromadb
from sentence_transformers import SentenceTransformer
import ollama
import os

os.makedirs("outputs", exist_ok=True)

print("Reading PDF...")

doc = fitz.open("sample.pdf")

text = ""

for page in doc:
    text += page.get_text()

chunks = [text[i:i+300] for i in range(0, len(text), 300)]

print("Chunks created:", len(chunks))

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="rag_database")

try:
    client.delete_collection("pdf_collection")
except:
    pass

collection = client.create_collection("pdf_collection")

embeddings = model.encode(chunks).tolist()

collection.add(
    documents=chunks,
    embeddings=embeddings,
    ids=[str(i) for i in range(len(chunks))]
)

query = "What is RAG?"

query_embedding = model.encode(query).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

context = "\n".join(results["documents"][0])

prompt = f"""
Use only the context below.

Context:
{context}

Question:
{query}
"""

print("\nSending to Ollama...\n")

response = ollama.chat(
    model="llama3.2",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

answer = response["message"]["content"]

print(answer)

with open("outputs/rag_output.txt", "w") as f:
    f.write(answer)

print("\nAnswer saved!")