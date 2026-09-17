import chromadb
from sentence_transformers import SentenceTransformer
import os

# Create output folder
os.makedirs("outputs", exist_ok=True)

# Load embedding model
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create ChromaDB client
client = chromadb.PersistentClient(path="chroma_db")

# Delete existing collection if present
try:
    client.delete_collection("documents")
except:
    pass

collection = client.create_collection("documents")

# 20 documents
documents = [
    "Artificial Intelligence is transforming healthcare.",
    "Machine Learning enables computers to learn from data.",
    "Deep Learning uses neural networks.",
    "Python is popular for AI development.",
    "Natural Language Processing works with text.",
    "Computer Vision understands images.",
    "Data Science combines statistics and programming.",
    "Cloud Computing provides scalable resources.",
    "Cybersecurity protects computer systems.",
    "DevOps automates software deployment.",
    "Docker creates lightweight containers.",
    "Kubernetes manages containers.",
    "Generative AI creates new content.",
    "Vector databases store embeddings.",
    "ChromaDB is an open-source vector database.",
    "Embeddings represent text as vectors.",
    "Cosine similarity measures semantic similarity.",
    "Large Language Models understand language.",
    "RAG combines retrieval with generation.",
    "Ollama runs LLMs locally."
]

metadata = [
    {"category": "AI"},
    {"category": "ML"},
    {"category": "DL"},
    {"category": "Programming"},
    {"category": "NLP"},
    {"category": "CV"},
    {"category": "DS"},
    {"category": "Cloud"},
    {"category": "Security"},
    {"category": "DevOps"},
    {"category": "Docker"},
    {"category": "Kubernetes"},
    {"category": "GenAI"},
    {"category": "VectorDB"},
    {"category": "VectorDB"},
    {"category": "Embedding"},
    {"category": "Math"},
    {"category": "LLM"},
    {"category": "RAG"},
    {"category": "LLM"}
]

ids = [str(i) for i in range(len(documents))]

print("Creating embeddings...")
embeddings = model.encode(documents).tolist()

collection.add(
    documents=documents,
    embeddings=embeddings,
    metadatas=metadata,
    ids=ids
)

print("\n20 documents inserted successfully!")

query = "What is Artificial Intelligence?"

query_embedding = model.encode(query).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

print("\nTop 3 Similar Documents\n")

similarity_output = ""

for i, doc in enumerate(results["documents"][0], start=1):
    print(f"{i}. {doc}")
    similarity_output += f"{i}. {doc}\n"

with open("outputs/similarity_search.txt", "w") as f:
    f.write(similarity_output)

print("\nMetadata Filtering\n")

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5,
    where={"category": "VectorDB"}
)

metadata_output = ""

for doc in results["documents"][0]:
    print(doc)
    metadata_output += doc + "\n"

with open("outputs/metadata_filter.txt", "w") as f:
    f.write(metadata_output)

print("\nOutputs saved successfully!")