# W5D4 Semantic Search with ChromaDB

## Objective

Implement semantic search using ChromaDB and Ollama embeddings.

## Features

- Installed ChromaDB
- Created persistent vector database
- Added 20 documents with embeddings
- Similarity Search using cosine similarity
- Metadata Filtering
- Embedded PDF into ChromaDB
- Retrieved Top-3 chunks
- Generated answer using Ollama (RAG)

## Run

```bash
pip install -r requirements.txt
```

Start Ollama

```bash
ollama serve
```

Run

```bash
python app.py
```

Outputs are stored inside

```
outputs/
```