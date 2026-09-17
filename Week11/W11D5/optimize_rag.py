from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import mlflow


# Load healthcare documents
loader = TextLoader("healthcare.txt")
documents = loader.load()

# Split documents into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

# Create embeddings
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# Create vector database
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="healthcare_optimization_v2"
)

question = "What are the benefits of AI in healthcare?"

with mlflow.start_run(run_name="W11D5-RAG-Optimization"):

    mlflow.log_param("chunk_size", 200)

    with open("optimization_results.txt", "w", encoding="utf-8") as file:

        for k in [2, 3]:

            results = vectorstore.similarity_search(
                question,
                k=k
            )

            file.write(f"Retrieval k={k}\n")
            file.write(f"Number of retrieved chunks: {len(results)}\n")

            for document in results:
                file.write(document.page_content + "\n")

            file.write("\n" + "-" * 50 + "\n\n")

            mlflow.log_param(f"retrieval_k_{k}", k)

    mlflow.log_artifact("optimization_results.txt")

print("Optimization comparison completed.")