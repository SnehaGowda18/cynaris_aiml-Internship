# RAG Evaluation and Optimization

## Description

This project implements a Retrieval-Augmented Generation pipeline using
LangChain, ChromaDB, and Ollama. It generates healthcare-related answers
and evaluates baseline and optimized retrieval configurations using Ragas.

The optimized configuration retrieves fewer document chunks to reduce
unnecessary context.

## Tools Used

- Python
- LangChain
- ChromaDB
- Ollama
- Ragas
- Hugging Face Embeddings
- Pandas

## Implementation

1. Created a healthcare knowledge document.
2. Split the document into smaller chunks.
3. Stored the chunks in ChromaDB.
4. Retrieved relevant context for each question.
5. Generated answers using Ollama.
6. Created baseline and optimized datasets.
7. Evaluated the datasets using Ragas.
8. Saved evaluation results as CSV files.

## Configuration

- Model: llama3.2:3b
- Chunk size: 400
- Chunk overlap: 50
- Baseline retrieval k: 2
- Optimized retrieval k: 1
- Evaluated questions: 2

## Results

Answer relevancy improved from approximately 0.6692 in the baseline
configuration to approximately 0.7057 in the optimized configuration.

Some Ragas metrics returned NaN because the local Ollama evaluation
model produced invalid JSON responses or timed out.

## Output Files

- `baseline_results.json`
- `optimized_results.json`
- `baseline_ragas_results.csv`
- `optimized_ragas_results.csv`
- `experiment_summary.txt`