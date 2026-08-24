# W7D5: Multi-Document RAG System

## Objective

Build a multi-document Retrieval Augmented Generation (RAG) system using LlamaIndex and local Ollama models.

## Technologies Used

- Python
- LlamaIndex
- Ollama
- llama3.2:3b
- nomic-embed-text
- Git & GitHub

## Implementation

The system loads multiple text documents from the `documents` folder and creates vector embeddings using the `nomic-embed-text` model.

LlamaIndex performs similarity-based retrieval and sends the relevant context to the local `llama3.2:3b` model to generate answers.

## RAG Pipeline

Documents  
↓  
LlamaIndex Document Loader  
↓  
nomic-embed-text  
↓  
Vector Embeddings  
↓  
VectorStoreIndex  
↓  
Similarity Retrieval  
↓  
llama3.2:3b  
↓  
Generated Answer

## Documents

The project uses four documents:

1. Python
2. Machine Learning
3. Retrieval Augmented Generation
4. Cybersecurity

## Test Questions

The system was tested with five questions:

1. What is Python used for?
2. What are the main types of machine learning?
3. What is Retrieval Augmented Generation?
4. What are common cybersecurity threats?
5. How does a RAG system generate answers?

## Output Evidence

The generated responses are saved in:

`outputs/rag_output.txt`

## Self-Review Checklist

- [x] Multiple documents loaded
- [x] Ollama LLM configured
- [x] Ollama embedding model configured
- [x] Vector index created
- [x] Similarity retrieval implemented
- [x] Multiple queries tested
- [x] Output evidence generated
- [x] Code tested successfully
- [x] Documentation completed
- [ ] CIA review completed
- [ ] Second CIA review completed
- [ ] Git commits completed
- [ ] Changes pushed
- [ ] PR raised

## Viva Answers

### 1. What did you build and why?

I built a multi-document RAG system using LlamaIndex and Ollama. It retrieves relevant information from multiple documents and uses a local LLM to generate answers. This approach improves responses by providing the model with relevant external context.

### 2. What was the hardest part?

The main challenge was configuring the embedding model and local LLM correctly. I solved this by using `nomic-embed-text` for embeddings and `llama3.2:3b` for answer generation.

### 3. What would you improve?

I would add persistent ChromaDB storage, source citations, Ragas evaluation, better document chunking, and a simple user interface.