# W5D5: Local Q&A Bot – Ollama + ChromaDB

## Objective

Build a Local Question & Answer Bot using Ollama and ChromaDB. The application stores documents in a vector database, retrieves relevant information using semantic search, and generates answers with a local Large Language Model (LLM).

---

## Technologies Used

- Python
- Ollama
- ChromaDB
- Sentence Transformers
- Requests

---

## Project Structure

```
Day5_Local_QA_Bot/
│── app.py
│── documents.txt
│── requirements.txt
│── README.md
│
├── chroma_db/
│
└── outputs/
    ├── llama_output.txt
    ├── qwen_output.txt
    ├── comparison.txt
```

---

## Installation

### 1. Install Python packages

```bash
pip install -r requirements.txt
```

### 2. Install Ollama

Download and install Ollama from:

https://ollama.com

### 3. Pull the required models

```bash
ollama pull llama3.2:3b
ollama pull qwen2.5:3b
```

### 4. Start Ollama

```bash
ollama serve
```

If Ollama is already running, you can skip this step.

### 5. Run the application

```bash
python app.py
```

---

## Features

- Stores documents in ChromaDB.
- Uses sentence embeddings for semantic search.
- Retrieves the most relevant context.
- Generates answers using Ollama.
- Compares responses from **llama3.2:3b** and **qwen2.5:3b**.
- Automatically creates:
  - `outputs/llama_output.txt`
  - `outputs/qwen_output.txt`
  - `outputs/comparison.txt`

---

## Sample Questions

- What is Artificial Intelligence?
- What is Machine Learning?
- What is ChromaDB?
- Why use local LLMs?
- What is Quantisation?

---

## Output

The program generates:

- Answer from **llama3.2:3b**
- Answer from **qwen2.5:3b**
- Model comparison report

These files are saved automatically inside the **outputs** folder.

---

## Learning Outcomes

- Understand Retrieval-Augmented Generation (RAG)
- Build a local AI-powered Q&A system
- Use ChromaDB for vector storage
- Compare responses from different LLMs
- Work with Ollama for local inference

---

## Author

**Sneha G R**

Week 5 Day 5 – AI/ML Internship Project