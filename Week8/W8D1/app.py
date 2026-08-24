from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Production RAG API",
    description="Dockerised ML API for RAG applications",
    version="1.0.0"
)


class QueryRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {
        "message": "Production RAG API is running",
        "status": "healthy"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/query")
def query(request: QueryRequest):
    question = request.question.strip()

    if not question:
        return {
            "error": "Question cannot be empty"
        }

    # Placeholder for RAG retrieval + generation
    answer = f"RAG response generated for: {question}"

    return {
        "question": question,
        "answer": answer
    }