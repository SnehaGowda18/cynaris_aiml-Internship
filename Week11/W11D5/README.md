# W11D5: Tracked & Evaluated RAG Pipeline

## Description

Implemented a local Retrieval-Augmented Generation (RAG) pipeline for healthcare question answering. The pipeline uses LangGraph for workflow orchestration, ChromaDB for vector retrieval, and Ollama for local language model generation.

The project also tracks experiments using MLflow and evaluates generated answers using Ragas.

## Tools Used

* Python
* LangGraph
* ChromaDB
* Ollama
* MLflow
* Ragas
* LangChain

## Features

* Healthcare document loading and chunking.
* Vector embeddings and similarity search.
* LangGraph-based retrieval and answer generation.
* MLflow experiment tracking.
* Ragas evaluation using faithfulness and answer relevancy.
* Retrieval optimization using different `k` values.

## Results

### Ragas Evaluation

* Faithfulness: 1.0000
* Answer Relevancy: 1.0000

### Retrieval Optimization

* Chunk size: 200
* `k=2`: Retrieved 2 chunks.
* `k=3`: Retrieved 3 chunks.

The optimization comparison showed that increasing `k` retrieved additional relevant healthcare information.

## Files

* `rag_pipeline.py` — Main RAG pipeline.
* `rag_evaluation.py` — Ragas evaluation.
* `optimize_rag.py` — Retrieval optimization.
* `healthcare.txt` — Healthcare knowledge source.
* `output.txt` — Generated answers.
* `experiment_results.txt` — Evaluation results.
* `optimization_results.txt` — Optimization evidence.

## Conclusion

The project demonstrates a tracked and evaluated RAG pipeline using local AI models. MLflow records experiment parameters and artifacts, while Ragas provides evaluation metrics for generated answers.
