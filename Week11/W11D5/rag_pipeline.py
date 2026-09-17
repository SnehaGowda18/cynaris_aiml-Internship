from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
import mlflow


# 1. Load healthcare knowledge
loader = TextLoader("healthcare.txt")
documents = loader.load()


# 2. Split documents into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)


# 3. Create embeddings and vector database
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="healthcare_rag"
)


# 4. Initialize local LLM
llm = Ollama(model="llama3.2:3b")


# 5. Define LangGraph state
class RAGState(TypedDict):
    question: str
    context: str
    answer: str


# 6. Retrieve relevant documents
def retrieve(state: RAGState):
    results = vectorstore.similarity_search(
        state["question"],
        k=3
    )

    context = "\n".join(
        document.page_content for document in results
    )

    return {"context": context}


# 7. Generate answer
def generate(state: RAGState):
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question using only the provided context.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
    )

    chain = prompt | llm

    answer = chain.invoke({
        "context": state["context"],
        "question": state["question"]
    })

    return {"answer": answer}


# 8. Build LangGraph workflow
graph = StateGraph(RAGState)

graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

rag_app = graph.compile()


# 9. Run experiment with MLflow
mlflow.set_experiment("W11D5-Tracked-Evaluated-RAG")

questions = [
    "What are the benefits of AI in healthcare?",
    "How is AI used in medical image analysis?",
    "What is the role of NLP in healthcare?",
    "Does AI replace healthcare professionals?",
    "What are the challenges of AI in healthcare?"
]

with mlflow.start_run():
    mlflow.log_param("chunk_size", 500)
    mlflow.log_param("retrieval_k", 3)

    with open("output.txt", "w", encoding="utf-8") as file:
        for question in questions:
            result = rag_app.invoke({
                "question": question,
                "context": "",
                "answer": ""
            })

            print(f"\nQuestion: {question}")
            print(f"Answer: {result['answer']}")

            file.write(f"Question: {question}\n")
            file.write(f"Answer: {result['answer']}\n\n")

    mlflow.log_artifact("output.txt")

print("\nRAG pipeline completed successfully.")