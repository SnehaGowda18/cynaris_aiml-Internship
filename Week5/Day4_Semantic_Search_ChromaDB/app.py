import chromadb
import ollama
from pypdf import PdfReader
import os


# -----------------------------
# Create ChromaDB Client
# -----------------------------

client = chromadb.PersistentClient(
    path="./chroma_db"
)


collection = client.get_or_create_collection(
    name="semantic_search"
)


# -----------------------------
# Add 20 Documents
# -----------------------------

documents = [
    "Artificial Intelligence enables machines to think and solve problems.",
    "Machine Learning is a subset of Artificial Intelligence.",
    "Deep Learning uses neural networks to learn complex patterns.",
    "Python is widely used for Artificial Intelligence development.",
    "ChromaDB stores vector embeddings for search applications.",
    "FAISS is a library for efficient similarity search.",
    "Semantic search finds documents with similar meanings.",
    "Embeddings convert text into numerical vectors.",
    "Cosine similarity measures similarity between vectors.",
    "Large Language Models generate human-like text.",
    "Ollama allows running AI models locally.",
    "RAG combines retrieval with language generation.",
    "LangGraph helps create AI workflows.",
    "CrewAI helps build AI agent systems.",
    "MLflow manages machine learning experiments.",
    "Ragas evaluates Retrieval Augmented Generation systems.",
    "Data preprocessing improves machine learning models.",
    "Feature engineering improves model performance.",
    "Vector databases store and retrieve embeddings.",
    "Generative AI creates new content using AI models."
]


metadata = [
    {"topic": "AI"},
    {"topic": "ML"},
    {"topic": "DL"},
    {"topic": "Python"},
    {"topic": "Database"},
    {"topic": "Database"},
    {"topic": "Search"},
    {"topic": "Embedding"},
    {"topic": "Math"},
    {"topic": "LLM"},
    {"topic": "LLM"},
    {"topic": "RAG"},
    {"topic": "Workflow"},
    {"topic": "Agent"},
    {"topic": "MLOps"},
    {"topic": "Evaluation"},
    {"topic": "Data"},
    {"topic": "Features"},
    {"topic": "Database"},
    {"topic": "GenAI"}
]


print("Generating embeddings...")


# Avoid duplicate IDs error
try:
    collection.delete(
        ids=[str(i) for i in range(20)]
    )
except:
    pass


for i, doc in enumerate(documents):

    embedding = ollama.embeddings(
        model="nomic-embed-text",
        prompt=doc
    )["embedding"]


    collection.add(
        ids=[str(i)],
        documents=[doc],
        embeddings=[embedding],
        metadatas=[metadata[i]]
    )


print("20 documents stored successfully!")


# -----------------------------
# Similarity Search
# -----------------------------

query = "What is semantic search?"


query_embedding = ollama.embeddings(
    model="nomic-embed-text",
    prompt=query
)["embedding"]


results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)


print("\nTop 3 Similar Documents")


for doc in results["documents"][0]:
    print(doc)



# Save output

os.makedirs("outputs", exist_ok=True)


with open(
    "outputs/similarity_output.txt",
    "w"
) as file:

    for doc in results["documents"][0]:
        file.write(doc + "\n")



# -----------------------------
# Metadata Filtering
# -----------------------------

print("\nMetadata Filter (Database)")


filtered = collection.get(
    where={
        "topic": "Database"
    }
)


for doc in filtered["documents"]:
    print(doc)



with open(
    "outputs/metadata_output.txt",
    "w"
) as file:

    for doc in filtered["documents"]:
        file.write(doc + "\n")



# -----------------------------
# PDF RAG Pipeline
# -----------------------------

if os.path.exists("sample.pdf"):


    print("\nProcessing PDF...")


    reader = PdfReader(
        "sample.pdf"
    )


    text = ""


    for page in reader.pages:
        text += page.extract_text()



    # Split into chunks

    chunks = []

    chunk_size = 300


    for i in range(
        0,
        len(text),
        chunk_size
    ):
        chunks.append(
            text[i:i+chunk_size]
        )



    pdf_collection = client.get_or_create_collection(
        name="pdf_collection"
    )


    # Clear old PDF data

    try:
        pdf_collection.delete(
            ids=[
                str(i)
                for i in range(len(chunks))
            ]
        )
    except:
        pass



    print("Embedding PDF chunks...")


    for i, chunk in enumerate(chunks):

        embedding = ollama.embeddings(
            model="nomic-embed-text",
            prompt=chunk
        )["embedding"]


        pdf_collection.add(
            ids=[str(i)],
            documents=[chunk],
            embeddings=[embedding]
        )



    # Retrieve top 3 chunks

    question = "What is Artificial Intelligence?"


    question_embedding = ollama.embeddings(
        model="nomic-embed-text",
        prompt=question
    )["embedding"]



    retrieved = pdf_collection.query(
        query_embeddings=[
            question_embedding
        ],
        n_results=3
    )


    context = "\n".join(
        retrieved["documents"][0]
    )



    # Generate Answer

    response = ollama.generate(
        model="llama3.2",
        prompt=f"""

You are a helpful AI assistant.

Use the following context to answer the question.

Context:
{context}

Question:
{question}

Give a short simple answer.

"""
    )


    print("\nRAG Answer\n")

    print(
        response["response"]
    )



    with open(
        "outputs/rag_output.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            response["response"]
        )



else:

    print("\nNo sample.pdf found.")