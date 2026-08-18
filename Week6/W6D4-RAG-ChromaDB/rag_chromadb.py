import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

# ---------------------------------------------------------
# 1. Connect to Ollama Embedding Model
# ---------------------------------------------------------

embedding_function = OllamaEmbeddingFunction(
    model_name="nomic-embed-text",
    url="http://localhost:11434"
)

# ---------------------------------------------------------
# 2. Create ChromaDB Client
# ---------------------------------------------------------

client = chromadb.PersistentClient(
    path="./chroma_db"
)

# ---------------------------------------------------------
# 3. Create Collection
# ---------------------------------------------------------

collection = client.get_or_create_collection(
    name="w6d4_collection",
    embedding_function=embedding_function,
    metadata={"hnsw:space": "cosine"}
)

# ---------------------------------------------------------
# 4. Create 20 Documents
# ---------------------------------------------------------

documents = [
    "Python is a programming language used for software development.",
    "Machine learning allows computers to learn from data.",
    "Deep learning uses neural networks with multiple layers.",
    "Artificial intelligence enables machines to perform intelligent tasks.",
    "Cybersecurity protects computers and networks from attacks.",
    "A firewall controls incoming and outgoing network traffic.",
    "Phishing is an attack that attempts to steal sensitive information.",
    "Encryption protects data by converting it into an encoded format.",
    "Cloud computing provides computing resources through the internet.",
    "AWS provides cloud computing services.",
    "Docker is a platform used for containerization.",
    "Kubernetes manages containerized applications.",
    "Git is a version control system.",
    "GitHub is a platform for hosting Git repositories.",
    "DevOps combines development and operations.",
    "RAG stands for Retrieval-Augmented Generation.",
    "Embeddings represent text as numerical vectors.",
    "ChromaDB is a vector database used for storing embeddings.",
    "Ollama allows language models to run locally.",
    "LangChain is a framework for building LLM applications."
]

# ---------------------------------------------------------
# 5. Create IDs
# ---------------------------------------------------------

ids = [
    f"doc_{i + 1}"
    for i in range(20)
]

# ---------------------------------------------------------
# 6. Create Metadata
# ---------------------------------------------------------

metadatas = [
    {"topic": "programming"},
    {"topic": "machine_learning"},
    {"topic": "deep_learning"},
    {"topic": "ai"},
    {"topic": "cybersecurity"},
    {"topic": "cybersecurity"},
    {"topic": "cybersecurity"},
    {"topic": "security"},
    {"topic": "cloud"},
    {"topic": "cloud"},
    {"topic": "devops"},
    {"topic": "devops"},
    {"topic": "git"},
    {"topic": "git"},
    {"topic": "devops"},
    {"topic": "rag"},
    {"topic": "embeddings"},
    {"topic": "chromadb"},
    {"topic": "ollama"},
    {"topic": "langchain"}
]

# ---------------------------------------------------------
# 7. Add Documents to ChromaDB
# ---------------------------------------------------------

collection.upsert(
    ids=ids,
    documents=documents,
    metadatas=metadatas
)

print("=" * 60)
print("W6D4 - ChromaDB")
print("=" * 60)

print("\nTotal documents:", collection.count())

# ---------------------------------------------------------
# 8. Similarity Search
# ---------------------------------------------------------

query = "What is a vector database?"

results = collection.query(
    query_texts=[query],
    n_results=3
)

print("\nQuery:")
print(query)

print("\nTop 3 Similar Documents:")

for i, document in enumerate(results["documents"][0]):
    print(f"\n{i + 1}. {document}")

# ---------------------------------------------------------
# 9. Metadata Filtering
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("Metadata Filtering")
print("=" * 60)

filtered = collection.get(
    where={"topic": "cybersecurity"}
)

print("\nCybersecurity Documents:")

for document in filtered["documents"]:
    print("-", document)

print("\nDone!")