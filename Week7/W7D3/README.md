# W7D3 - LlamaIndex Document Indexing and RAG

## Objective

Build a document-based Retrieval Augmented Generation (RAG) system using LlamaIndex, Ollama embeddings, and ChromaDB.

## Technologies Used

- Python 3.11
- LlamaIndex
- Ollama
- llama3.2:3b
- nomic-embed-text
- ChromaDB
- VectorStoreIndex
- QueryEngine

## Project Structure

```text
W7D3/
├── data/
│   ├── ai.txt
│   ├── cybersecurity.txt
│   ├── cloud.txt
│   ├── mlops.txt
│   └── rag.txt
│
├── outputs/
│   ├── basic_results.txt
│   └── chroma_results.txt
│
├── chroma_db/
├── llamaindex_basic.py
├── llamaindex_chroma.py
├── comparison.py
└── README.md