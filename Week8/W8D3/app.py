from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="W8D3 Haystack Retrieval API",
    description="Haystack BM25 document retrieval API",
    version="1.0.0",
)


# ============================================================
# REQUEST MODEL
# ============================================================

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3


# ============================================================
# BUILD HAYSTACK BM25 INDEX
# ============================================================

def build_retrieval_system():

    document_store = InMemoryDocumentStore()

    converter = PyPDFToDocument()

    splitter = DocumentSplitter(
        split_by="sentence",
        split_length=5,
        split_overlap=1,
    )

    writer = DocumentWriter(
        document_store=document_store
    )

    indexing_pipeline = Pipeline()

    indexing_pipeline.add_component(
        "converter",
        converter
    )

    indexing_pipeline.add_component(
        "splitter",
        splitter
    )

    indexing_pipeline.add_component(
        "writer",
        writer
    )

    indexing_pipeline.connect(
        "converter.documents",
        "splitter.documents"
    )

    indexing_pipeline.connect(
        "splitter.documents",
        "writer.documents"
    )

    pdf_files = list(DATA_DIR.glob("*.pdf"))

    if len(pdf_files) != 5:
        raise ValueError(
            f"Expected 5 PDF files, found {len(pdf_files)}"
        )

    print("\nIndexing PDFs:")

    for pdf in pdf_files:
        print(" -", pdf.name)

    indexing_pipeline.run(
        {
            "converter": {
                "sources": pdf_files
            }
        }
    )

    retriever = InMemoryBM25Retriever(
        document_store=document_store
    )

    print(
        f"\nSuccessfully indexed "
        f"{document_store.count_documents()} documents/chunks."
    )

    return document_store, retriever


# ============================================================
# INITIALIZE RETRIEVAL SYSTEM
# ============================================================

document_store, retriever = build_retrieval_system()


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "W8D3 Haystack Retrieval API is running",
        "retrieval_method": "BM25",
        "documents": document_store.count_documents(),
    }


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "retrieval_method": "BM25",
        "documents": document_store.count_documents(),
    }


# ============================================================
# RETRIEVAL ENDPOINT
# ============================================================

@app.post("/retrieve")
def retrieve(request: QueryRequest):

    if not request.query.strip():
        return {
            "error": "Query cannot be empty"
        }

    if request.top_k < 1:
        return {
            "error": "top_k must be at least 1"
        }

    result = retriever.run(
        query=request.query,
        top_k=request.top_k
    )

    documents = []

    for rank, document in enumerate(
        result["documents"],
        start=1
    ):

        documents.append(
            {
                "rank": rank,
                "score": document.score,
                "content": document.content,
                "source": document.meta.get(
                    "file_path",
                    "unknown"
                ),
            }
        )

    return {
        "query": request.query,
        "top_k": request.top_k,
        "results": documents,
    }