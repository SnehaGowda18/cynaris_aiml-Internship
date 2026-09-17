# W8D1: Production RAG — Evaluation & Optimisation

## Objective

The objective of W8D1 is to prepare an ML/RAG API for production by
containerising the application, implementing automated CI checks, and
defining a production monitoring strategy.

## Technologies

- Python 3.11
- FastAPI
- Docker
- Pytest
- Ruff
- GitHub Actions
- RAG
- ChromaDB

## Project Structure

```text
W8D1/
├── app.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── README.md
├── tests/
│   └── test_api.py
└── .github/
    └── workflows/
        └── ci.yml