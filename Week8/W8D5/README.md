# Local AI Research Assistant

## Overview

The Local AI Research Assistant is a local Retrieval-Augmented Generation (RAG) system that retrieves relevant information from a local document collection and uses a local Large Language Model (LLM) to generate research-based answers.

The project combines vector search, LangGraph workflow orchestration, CrewAI concepts, MLflow experiment tracking, Ragas evaluation, and MLOps practices.

The system is designed to run locally without depending on a cloud-hosted LLM for answer generation.

---

## Objectives

- Build a local AI research assistant.
- Retrieve relevant information from local documents.
- Generate answers using a local LLM.
- Orchestrate the RAG workflow using LangGraph.
- Store and search documents using ChromaDB.
- Track experiments and metrics using MLflow.
- Evaluate RAG responses using Ragas.
- Add automated testing.
- Follow Git and MLOps practices.

---

## Architecture

```text
                    User Question
                          |
                          v
                 +------------------+
                 |    LangGraph     |
                 |     Workflow     |
                 +------------------+
                          |
                          v
                 +------------------+
                 | Document         |
                 | Retrieval        |
                 +------------------+
                          |
                          v
                 +------------------+
                 |    ChromaDB      |
                 | Vector Search    |
                 +------------------+
                          |
                          v
                 +------------------+
                 | Retrieved        |
                 | Documents        |
                 +------------------+
                          |
                          v
                 +------------------+
                 | Local LLM        |
                 | llama3.2:3b      |
                 +------------------+
                          |
                          v
                    Final Answer
                          |
                          v
                 +------------------+
                 |      MLflow      |
                 | Tracking & Logs  |
                 +------------------+