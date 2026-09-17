# Week 5 Day 3 – ChromaDB Vector Store Setup & Embedding Documents

## Objective

This project demonstrates how to use ChromaDB as a vector database, generate embeddings with Sentence Transformers, perform similarity search, filter using metadata, and build a simple Retrieval-Augmented Generation (RAG) pipeline using Ollama.

## Features

- Install and use ChromaDB
- Store 20 embedded documents
- Perform cosine similarity search
- Metadata filtering
- PDF retrieval
- Local LLM integration using Ollama

## Technologies Used

- Python
- ChromaDB
- Sentence Transformers
- PyMuPDF
- Ollama

## Files

- chromadb_setup.py
- rag_with_pdf.py
- sample.pdf
- requirements.txt

## Outputs

- similarity_search.txt
- metadata_filter.txt
- rag_output.txt

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python chromadb_setup.py
```

Then:

```bash
python rag_with_pdf.py
```