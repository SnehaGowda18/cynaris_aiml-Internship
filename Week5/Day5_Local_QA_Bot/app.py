import os
import requests
import chromadb
from sentence_transformers import SentenceTransformer

# -----------------------------
# Create folders
# -----------------------------
os.makedirs("outputs", exist_ok=True)
os.makedirs("chroma_db", exist_ok=True)

# -----------------------------
# Load embedding model
# -----------------------------
print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Create ChromaDB client
# -----------------------------
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("knowledge_base")

# -----------------------------
# Read documents
# -----------------------------
with open("documents.txt", "r", encoding="utf-8") as file:
    documents = [line.strip() for line in file if line.strip()]

# -----------------------------
# Store documents only once
# -----------------------------
if collection.count() == 0:
    embeddings = embedding_model.encode(documents).tolist()

    collection.add(
        ids=[str(i) for i in range(len(documents))],
        documents=documents,
        embeddings=embeddings
    )

print("Knowledge Base Ready!")

# -----------------------------
# Ask Question
# -----------------------------
question = input("\nEnter your question: ")

query_embedding = embedding_model.encode([question]).tolist()

results = collection.query(
    query_embeddings=query_embedding,
    n_results=2
)

context = "\n".join(results["documents"][0])

prompt = f"""
You are a helpful AI assistant.

Answer the question only using the context below.

Context:
{context}

Question:
{question}

Answer:
"""

# -----------------------------
# Function to call Ollama
# -----------------------------
def ask_model(model_name):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model_name,
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"]


# -----------------------------
# Get Responses
# -----------------------------
print("\nGenerating response using Llama...")
llama_response = ask_model("llama3.2:3b")

print("Generating response using Qwen...")
qwen_response = ask_model("qwen2.5:3b")

# -----------------------------
# Save Outputs Automatically
# -----------------------------
with open("outputs/llama_output.txt", "w", encoding="utf-8") as file:
    file.write("MODEL : llama3.2:3b\n\n")
    file.write("QUESTION:\n")
    file.write(question)
    file.write("\n\nANSWER:\n")
    file.write(llama_response)

with open("outputs/qwen_output.txt", "w", encoding="utf-8") as file:
    file.write("MODEL : qwen2.5:3b\n\n")
    file.write("QUESTION:\n")
    file.write(question)
    file.write("\n\nANSWER:\n")
    file.write(qwen_response)

comparison = f"""
=============================
MODEL COMPARISON
=============================

Question:
{question}

-----------------------------------
Llama3.2:3b
-----------------------------------
{llama_response}

-----------------------------------
Qwen2.5:3b
-----------------------------------
{qwen_response}

-----------------------------------
Observations
-----------------------------------
1. Compare response clarity.
2. Compare accuracy.
3. Compare detail level.
4. Compare readability.
"""

with open("outputs/comparison.txt", "w", encoding="utf-8") as file:
    file.write(comparison)

# -----------------------------
# Display Responses
# -----------------------------
print("\n==============================")
print("Llama3.2:3b Response")
print("==============================")
print(llama_response)

print("\n==============================")
print("Qwen2.5:3b Response")
print("==============================")
print(qwen_response)

print("\nAll output files created successfully!")

print("\nGenerated Files:")
print("outputs/llama_output.txt")
print("outputs/qwen_output.txt")
print("outputs/comparison.txt")