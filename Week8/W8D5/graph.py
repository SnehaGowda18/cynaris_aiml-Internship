"""
LangGraph workflow for the Local AI Research Assistant.

Workflow:

START
  ↓
Retrieve Documents
  ↓
Generate Answer
  ↓
END
"""

from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama

from vector_store import search_documents


class ResearchState(TypedDict):
    """State shared between LangGraph nodes."""

    question: str
    documents: list[str]
    answer: str


# ---------------------------------------------------------
# Local Ollama LLM
# ---------------------------------------------------------

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
)


# ---------------------------------------------------------
# Node 1: Retrieve documents
# ---------------------------------------------------------

def retrieve_documents(state: ResearchState):
    """Retrieve relevant documents from ChromaDB."""

    documents = search_documents(
        state["question"],
        top_k=3,
    )

    print(
        f"Retrieved {len(documents)} document(s)."
    )

    return {
        "documents": documents,
    }


# ---------------------------------------------------------
# Node 2: Generate answer
# ---------------------------------------------------------

def generate_answer(state: ResearchState):
    """Generate an answer using retrieved research context."""

    context = "\n\n".join(
        state["documents"]
    )

    prompt = f"""
You are a local AI research assistant.

Answer the user's research question using ONLY
the provided research context.

If the context does not contain enough information,
say that the information is not available in the
local research documents.

Research Context:
{context}

Research Question:
{state["question"]}

Provide a clear and concise answer.
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
    }


# ---------------------------------------------------------
# Build LangGraph workflow
# ---------------------------------------------------------

workflow = StateGraph(ResearchState)

workflow.add_node(
    "retrieve_documents",
    retrieve_documents,
)

workflow.add_node(
    "generate_answer",
    generate_answer,
)

workflow.add_edge(
    START,
    "retrieve_documents",
)

workflow.add_edge(
    "retrieve_documents",
    "generate_answer",
)

workflow.add_edge(
    "generate_answer",
    END,
)

# Compile graph
research_graph = workflow.compile()


# ---------------------------------------------------------
# Test LangGraph directly
# ---------------------------------------------------------

if __name__ == "__main__":

    question = (
        "What is Retrieval Augmented Generation?"
    )

    result = research_graph.invoke(
        {
            "question": question,
            "documents": [],
            "answer": "",
        }
    )

    print("\n" + "=" * 60)
    print("LOCAL AI RESEARCH ASSISTANT")
    print("=" * 60)

    print("\nQuestion:")
    print(question)

    print("\nAnswer:")
    print(result["answer"])

    print("\nSources Retrieved:")

    for index, document in enumerate(
        result["documents"],
        start=1
    ):

        print("-" * 40)
        print(f"Document {index}")
        print(document[:300])