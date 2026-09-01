from datasets import Dataset

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from rag_pipeline import search


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "llama3.2:3b"

BASELINE_CHUNK_SIZE = 500
BASELINE_K = 3

OPTIMIZED_CHUNK_SIZE = 300
OPTIMIZED_K = 5


# ============================================================
# OLLAMA LLM
# ============================================================

ollama_llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0,
)

ragas_llm = LangchainLLMWrapper(
    ollama_llm
)


# ============================================================
# EMBEDDINGS
# ============================================================

hf_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

ragas_embeddings = LangchainEmbeddingsWrapper(
    hf_embeddings
)


# ============================================================
# 10 QUESTIONS + REFERENCE ANSWERS
# ============================================================

QA_PAIRS = [
    {
        "question": "What is Artificial Intelligence?",
        "reference": (
            "Artificial Intelligence is the field of computer "
            "science focused on creating systems that perform "
            "tasks requiring human-like intelligence."
        ),
    },

    {
        "question": "What is Machine Learning?",
        "reference": (
            "Machine Learning is a subset of Artificial Intelligence "
            "that enables computers to learn patterns from data."
        ),
    },

    {
        "question": "What is Deep Learning?",
        "reference": (
            "Deep Learning is a subset of machine learning that "
            "uses neural networks with multiple layers."
        ),
    },

    {
        "question": "What is Natural Language Processing?",
        "reference": (
            "Natural Language Processing enables computers to "
            "understand, process, and generate human language."
        ),
    },

    {
        "question": "What is Retrieval Augmented Generation?",
        "reference": (
            "Retrieval Augmented Generation combines information "
            "retrieval with large language models by retrieving "
            "relevant documents before generating an answer."
        ),
    },

    {
        "question": "What is a vector database?",
        "reference": (
            "A vector database stores numerical representations "
            "called embeddings and supports similarity searches."
        ),
    },

    {
        "question": "What is ChromaDB?",
        "reference": (
            "ChromaDB is a vector database used to store embeddings "
            "and retrieve similar documents."
        ),
    },

    {
        "question": "What are embeddings?",
        "reference": (
            "Embeddings are numerical representations of data that "
            "can be used for semantic similarity searches."
        ),
    },

    {
        "question": "What are Large Language Models?",
        "reference": (
            "Large Language Models are neural network models trained "
            "on large amounts of text data."
        ),
    },

    {
        "question": "What is RAG evaluation?",
        "reference": (
            "RAG evaluation measures the quality of a retrieval "
            "augmented generation system using metrics such as "
            "faithfulness, answer relevancy, context precision, "
            "and context recall."
        ),
    },
]


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(question, contexts):

    context_text = "\n\n".join(contexts)

    prompt = f"""
You are a RAG question answering assistant.

Answer the question using ONLY the provided context.

Context:
{context_text}

Question:
{question}

Rules:
- Do not invent information.
- Use only information from the context.
- Give a short factual answer.
"""

    response = ollama_llm.invoke(prompt)

    return response.content


# ============================================================
# CREATE RAG EVALUATION DATASET
# ============================================================

def create_dataset(
    k=3,
    chunk_size=500,
):

    dataset_rows = []

    for item in QA_PAIRS:

        question = item["question"]

        contexts = search(
            question,
            k=k,
            chunk_size=chunk_size,
        )

        answer = generate_answer(
            question,
            contexts,
        )

        dataset_rows.append(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": item["reference"],
            }
        )

        print("\n" + "-" * 60)
        print("QUESTION:")
        print(question)

        print("\nANSWER:")
        print(answer)

        print("\nCONTEXTS:")
        print(len(contexts))

    return dataset_rows


# ============================================================
# RAGAS EVALUATION
# ============================================================

def evaluate_rag(
    k=3,
    chunk_size=500,
):

    print("\n")
    print("=" * 70)
    print("RAGAS EVALUATION")
    print("=" * 70)

    print(f"Chunk size: {chunk_size}")
    print(f"Retrieval k: {k}")

    rows = create_dataset(
        k=k,
        chunk_size=chunk_size,
    )

    dataset = Dataset.from_list(rows)

    result = evaluate(
        dataset=dataset,

        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],

        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    print("\n")
    print("=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print(result)

    return result


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("W8D2 - RAG EVALUATION WITH RAGAS")
    print("=" * 70)

    print("\nRunning baseline evaluation...")

    baseline_result = evaluate_rag(
        k=BASELINE_K,
        chunk_size=BASELINE_CHUNK_SIZE,
    )

    print("\n")
    print("=" * 70)
    print("BASELINE COMPLETED")
    print("=" * 70)

    print("\nNow running optimized configuration...")

    optimized_result = evaluate_rag(
        k=OPTIMIZED_K,
        chunk_size=OPTIMIZED_CHUNK_SIZE,
    )

    print("\n")
    print("=" * 70)
    print("OPTIMIZATION COMPLETED")
    print("=" * 70)

    print("\nBaseline:")
    print(baseline_result)

    print("\nOptimized:")
    print(optimized_result)