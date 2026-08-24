import time
from pathlib import Path

import chromadb

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
)

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore


# ============================================================
# W7D3 - LlamaIndex + ChromaDB RAG
# ============================================================

DATA_DIR = "data"
CHROMA_DIR = "./chroma_db"
OUTPUT_DIR = Path("outputs")
OUTPUT_FILE = OUTPUT_DIR / "chroma_results.txt"

OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 1. Configure Ollama
# ============================================================

print("=" * 70)
print("W7D3 - LlamaIndex + ChromaDB")
print("=" * 70)

print("\nConfiguring Ollama...")

Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text"
)

Settings.llm = Ollama(
    model="llama3.2:3b",
    request_timeout=300.0,
    temperature=0.0,
)

print("Ollama configuration completed.")


# ============================================================
# 2. Load Documents
# ============================================================

print("\nLoading documents...")

documents = SimpleDirectoryReader(
    DATA_DIR,
    recursive=True
).load_data()

print(f"Documents loaded: {len(documents)}")

if not documents:
    raise RuntimeError(
        "No documents found inside the data folder."
    )


# ============================================================
# 3. Connect to Persistent ChromaDB
# ============================================================

print("\nConnecting to ChromaDB...")

chroma_client = chromadb.PersistentClient(
    path=CHROMA_DIR
)

collection = chroma_client.get_or_create_collection(
    name="w7d3_documents"
)

print("ChromaDB collection ready.")
print(f"Existing vectors: {collection.count()}")


# ============================================================
# 4. Create Chroma Vector Store
# ============================================================

vector_store = ChromaVectorStore(
    chroma_collection=collection
)

storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)


# ============================================================
# 5. Create LlamaIndex + ChromaDB Index
# ============================================================

print("\nCreating LlamaIndex + ChromaDB index...")

index_start = time.perf_counter()

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)

index_time = time.perf_counter() - index_start

print(
    f"Index created in {index_time:.4f} seconds"
)

print(
    f"Vectors stored in ChromaDB: "
    f"{collection.count()}"
)


# ============================================================
# 6. Create Query Engine
# ============================================================

query_engine = index.as_query_engine(
    similarity_top_k=2
)

print("QueryEngine created successfully.")


# ============================================================
# 7. Ten Evaluation Queries
# ============================================================

queries = [
    "What is artificial intelligence?",
    "What is cybersecurity?",
    "What is cloud computing?",
    "What is MLOps?",
    "What is Retrieval Augmented Generation?",
    "What are common cybersecurity threats?",
    "What are the main cloud service models?",
    "What does MLOps focus on?",
    "How does RAG work?",
    "What are common applications of artificial intelligence?",
]


# ============================================================
# 8. Run Queries
# ============================================================

successful = 0
failed = 0
latencies = []

print("\nRunning 10 queries...")
print("=" * 70)


with open(OUTPUT_FILE, "w", encoding="utf-8") as output:

    output.write(
        "W7D3 - LlamaIndex + ChromaDB Results\n"
    )

    output.write("=" * 70 + "\n")

    output.write(
        f"Documents loaded: {len(documents)}\n"
    )

    output.write(
        f"Index creation time: "
        f"{index_time:.4f} seconds\n"
    )

    output.write(
        f"ChromaDB vectors: "
        f"{collection.count()}\n"
    )

    output.write("=" * 70 + "\n")


    for number, question in enumerate(
        queries,
        start=1
    ):

        print(f"\nQuery {number}/10")
        print(f"Question: {question}")

        start_time = time.perf_counter()

        try:

            response = query_engine.query(
                question
            )

            latency = (
                time.perf_counter()
                - start_time
            )

            successful += 1
            latencies.append(latency)

            print(f"Answer: {response}")
            print(
                f"Latency: "
                f"{latency:.4f} seconds"
            )

            source_nodes = getattr(
                response,
                "source_nodes",
                []
            )

            print(
                f"Sources retrieved: "
                f"{len(source_nodes)}"
            )


            output.write(
                "\n" + "=" * 70 + "\n"
            )

            output.write(
                f"Query {number}\n"
            )

            output.write(
                f"Question: {question}\n"
            )

            output.write(
                f"Answer: {response}\n"
            )

            output.write(
                f"Latency: "
                f"{latency:.4f} seconds\n"
            )

            output.write(
                f"Sources retrieved: "
                f"{len(source_nodes)}\n"
            )


            if source_nodes:

                output.write(
                    "\nSource information:\n"
                )

                for source_number, node in enumerate(
                    source_nodes,
                    start=1
                ):

                    metadata = node.node.metadata

                    file_name = metadata.get(
                        "file_name",
                        "Unknown"
                    )

                    score = node.score

                    output.write(
                        f"  Source {source_number}: "
                        f"{file_name}"
                    )

                    if score is not None:
                        output.write(
                            f" | Score: {score:.4f}"
                        )

                    output.write("\n")


        except Exception as error:

            failed += 1

            print(f"ERROR: {error}")

            output.write(
                "\n" + "=" * 70 + "\n"
            )

            output.write(
                f"Query {number}\n"
            )

            output.write(
                f"Question: {question}\n"
            )

            output.write(
                f"ERROR: {error}\n"
            )


# ============================================================
# 9. Summary
# ============================================================

print("\n" + "=" * 70)
print("CHROMADB EXECUTION SUMMARY")
print("=" * 70)

print(
    f"Documents loaded  : {len(documents)}"
)

print(
    f"Index time        : "
    f"{index_time:.4f} seconds"
)

print(
    f"ChromaDB vectors  : "
    f"{collection.count()}"
)

print(
    f"Successful queries: {successful}"
)

print(
    f"Failed queries    : {failed}"
)


if latencies:

    average_latency = (
        sum(latencies)
        / len(latencies)
    )

    print(
        f"Average latency   : "
        f"{average_latency:.4f} seconds"
    )

    print(
        f"Fastest query     : "
        f"{min(latencies):.4f} seconds"
    )

    print(
        f"Slowest query     : "
        f"{max(latencies):.4f} seconds"
    )

    with open(
        OUTPUT_FILE,
        "a",
        encoding="utf-8"
    ) as output:

        output.write(
            "\n" + "=" * 70 + "\n"
        )

        output.write(
            "CHROMADB EXECUTION SUMMARY\n"
        )

        output.write(
            "=" * 70 + "\n"
        )

        output.write(
            f"Documents loaded  : "
            f"{len(documents)}\n"
        )

        output.write(
            f"Index time        : "
            f"{index_time:.4f} seconds\n"
        )

        output.write(
            f"ChromaDB vectors  : "
            f"{collection.count()}\n"
        )

        output.write(
            f"Successful queries: "
            f"{successful}\n"
        )

        output.write(
            f"Failed queries    : "
            f"{failed}\n"
        )

        output.write(
            f"Average latency   : "
            f"{average_latency:.4f} seconds\n"
        )

        output.write(
            f"Fastest query     : "
            f"{min(latencies):.4f} seconds\n"
        )

        output.write(
            f"Slowest query     : "
            f"{max(latencies):.4f} seconds\n"
        )


print(
    f"\nResults saved to: "
    f"{OUTPUT_FILE}"
)

if successful == len(queries):

    print(
        "\nSUCCESS: All 10 ChromaDB "
        "queries completed successfully."
    )

else:

    print(
        f"\nCompleted with "
        f"{failed} failed query(s)."
    )