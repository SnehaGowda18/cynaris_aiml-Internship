# Production RAG API — W8D4

## Overview

This project implements a Production RAG API with a focus on documentation, automated testing, code quality, and code review.

The system accepts a user question, retrieves relevant information from the knowledge base, and generates an answer using the retrieved context.

## Objectives

* Build a clean RAG API
* Implement input validation
* Add automated API tests
* Document the project
* Perform a code review
* Follow a proper Git workflow
* Improve production readiness

## Technologies

* Python
* FastAPI
* Pydantic
* Pytest
* RAG
* ChromaDB concepts
* Git and GitHub
* Docker

## Project Structure

```text
W8D4/
│
├── app.py
├── rag_pipeline.py
├── requirements.txt
├── pytest.ini
├── README.md
│
├── tests/
│   ├── __init__.py
│   └── test_api.py
│
└── docs/
    └── code_review.md
```

## Installation

Create and activate a virtual environment if required.

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

Start the FastAPI server:

```bash
uvicorn app:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Swagger API documentation:

```text
http://127.0.0.1:8000/docs
```

## API Endpoints

### GET /

Returns basic application information.

### GET /health

Checks whether the API is running.

Example response:

```json
{
  "status": "healthy",
  "service": "rag-api"
}
```

### POST /query

Accepts a question and returns an answer with retrieved sources.

Request:

```json
{
  "question": "What is RAG?"
}
```

Response:

```json
{
  "question": "What is RAG?",
  "answer": "Based on the retrieved information: ...",
  "sources": [
    "Retrieved document..."
  ]
}
```

## Testing

Run all automated tests using:

```bash
pytest -v
```

The test suite validates:

* Root endpoint
* Health endpoint
* RAG query endpoint
* Retrieval
* Empty input
* Missing input
* Maximum input length

## Code Review

A detailed code review is available in:

```text
docs/code_review.md
```

The review covers:

* Code quality
* Organization
* Error handling
* Input validation
* Testing
* Documentation
* Security
* Performance
* Maintainability

## Self-Review Checklist

* [x] Code is clean and readable
* [x] Functions have clear responsibilities
* [x] Input validation implemented
* [x] Error handling implemented
* [x] Automated tests added
* [x] Tests executed
* [x] README updated
* [x] Code review completed
* [x] Git commits use descriptive messages
* [x] Changes pushed to feature branch

## Future Improvements

If more development time is available, the project can be improved by adding:

* Real ChromaDB vector retrieval
* LLM-based answer generation
* Ragas evaluation
* MLflow experiment tracking
* Authentication
* Rate limiting
* Logging and monitoring
* CI/CD automated testing
* Docker deployment

## Conclusion

W8D4 improves the Production RAG API by adding structured documentation, automated testing, input validation, error handling, and a formal code review process. These improvements make the project easier to maintain, test, and prepare for production deployment.
