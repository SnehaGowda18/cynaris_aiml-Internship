import time
from pathlib import Path

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
)
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama


# ============================================================
# W7D3 - LlamaIndex Document Indexing & Querying
# ============================================================

DATA_DIR = "data"
OUTPUT_DIR = Path("outputs")
OUTPUT_FILE = OUTPUT_DIR / "basic_results.txt"


# ============================================================
# 1. Ollama Configuration
# ============================================================

print("=" * 70)
print("W7D3 - LlamaIndex Document Indexing & Querying")
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

if len(documents) == 0:
    raise RuntimeError("No documents found inside the data folder.")


# ============================================================
# 3. Create VectorStoreIndex
# ============================================================

print("\nCreating VectorStoreIndex...")

index_start = time.perf_counter()

index = VectorStoreIndex.from_documents(
    documents
)

index_time = time.perf_counter() - index_start

print(f"Index created in {index_time:.4f} seconds")


# ============================================================
# 4. Create QueryEngine
# ============================================================

print("\nCreating QueryEngine...")

query_engine = index.as_query_engine(
    similarity_top_k=2
)

print("QueryEngine created successfully.")


# ============================================================
# 5. Test Queries
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
# 6. Create Output Folder
# ============================================================

OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 7. Run Queries
# ============================================================

print("\nRunning 10 queries...")
print("=" * 70)

successful = 0
failed = 0
latencies = []

with open(OUTPUT_FILE, "w", encoding="utf-8") as output:

    output.write("W7D3 - LlamaIndex Basic RAG Results\n")
    output.write("=" * 70 + "\n")
    output.write(f"Documents: {len(documents)}\n")
    output.write(f"Index creation time: {index_time:.4f} seconds\n")
    output.write("=" * 70 + "\n")

    for number, question in enumerate(queries, start=1):

        print(f"\nQuery {number}/10")
        print(f"Question: {question}")

        start_time = time.perf_counter()

        try:

            response = query_engine.query(question)

            latency = time.perf_counter() - start_time

            successful += 1
            latencies.append(latency)

            print(f"Answer: {response}")
            print(f"Latency: {latency:.4f} seconds")

            # ------------------------------------------------
            # Source verification
            # ------------------------------------------------

            source_nodes = getattr(response, "source_nodes", [])

            print(f"Sources retrieved: {len(source_nodes)}")

            output.write("\n" + "=" * 70 + "\n")
            output.write(f"Query {number}\n")
            output.write(f"Question: {question}\n")
            output.write(f"Answer: {response}\n")
            output.write(f"Latency: {latency:.4f} seconds\n")
            output.write(
                f"Sources retrieved: {len(source_nodes)}\n"
            )

            if source_nodes:

                output.write("\nSource information:\n")

                for source_number, node in enumerate(
                    source_nodes,
                    start=1
                ):

                    metadata = node.node.metadata

                    file_name = (
                        metadata.get(
                            "file_name",
                            "Unknown"
                        )
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

            else:

                output.write(
                    "No source nodes returned.\n"
                )

        except Exception as error:

            failed += 1

            latency = time.perf_counter() - start_time

            print(f"ERROR: {error}")

            output.write("\n" + "=" * 70 + "\n")
            output.write(f"Query {number}\n")
            output.write(f"Question: {question}\n")
            output.write(f"ERROR: {error}\n")
            output.write(f"Time before failure: {latency:.4f} seconds\n")


# ============================================================
# 8. Summary
# ============================================================

print("\n" + "=" * 70)
print("EXECUTION SUMMARY")
print("=" * 70)

print(f"Documents loaded : {len(documents)}")
print(f"Index time       : {index_time:.4f} seconds")
print(f"Successful queries: {successful}")
print(f"Failed queries    : {failed}")

if latencies:

    average_latency = sum(latencies) / len(latencies)

    print(
        f"Average latency  : "
        f"{average_latency:.4f} seconds"
    )

    print(
        f"Fastest query    : "
        f"{min(latencies):.4f} seconds"
    )

    print(
        f"Slowest query    : "
        f"{max(latencies):.4f} seconds"
    )

    with open(OUTPUT_FILE, "a", encoding="utf-8") as output:

        output.write("\n" + "=" * 70 + "\n")
        output.write("EXECUTION SUMMARY\n")
        output.write("=" * 70 + "\n")
        output.write(
            f"Documents loaded : {len(documents)}\n"
        )
        output.write(
            f"Index time       : {index_time:.4f} seconds\n"
        )
        output.write(
            f"Successful queries: {successful}\n"
        )
        output.write(
            f"Failed queries    : {failed}\n"
        )
        output.write(
            f"Average latency  : {average_latency:.4f} seconds\n"
        )
        output.write(
            f"Fastest query    : {min(latencies):.4f} seconds\n"
        )
        output.write(
            f"Slowest query    : {max(latencies):.4f} seconds\n"
        )

print(f"\nResults saved to: {OUTPUT_FILE}")

if successful == len(queries):
    print("\nSUCCESS: All 10 queries completed successfully.")
else:
    print(
        f"\nCompleted with {failed} failed query(s)."
    )