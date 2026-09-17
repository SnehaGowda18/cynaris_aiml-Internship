# W6D4 - RAG Pipeline with ChromaDB

## Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline using Python, ChromaDB, and Ollama.

The system reads a PDF document, divides the text into smaller chunks, creates embeddings, stores them in ChromaDB, retrieves the most relevant chunks for a user question, and uses an Ollama language model to generate the final answer.

## Technologies Used

- Python
- ChromaDB
- Ollama
- nomic-embed-text
- llama3.2:3b
- PyPDF

## Project Structure

```text
W6D4-RAG-ChromaDB/
│
├── data/
│   └── sample.txt
│
├── outputs/
│   └── rag_output.txt
│
├── pdf_chroma/
│
├── pdf_rag.py
├── rag_chromadb.py
├── sample.pdf
├── requirements.txt
└── README.md