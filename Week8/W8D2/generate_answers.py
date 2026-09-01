from langchain_ollama import ChatOllama

from evaluation import QA_PAIRS
from rag_pipeline import retrieve


llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
)


def generate_answer(question, contexts):
    context_text = "\n\n".join(contexts)

    prompt = f"""
Answer the question using only the provided context.

Context:
{context_text}

Question:
{question}

Give a concise and factual answer.
"""

    response = llm.invoke(prompt)

    return response.content


def build_dataset():
    results = []

    for item in QA_PAIRS:
        documents = retrieve(
            item["question"],
            k=3,
            chunk_size=500,
        )

        contexts = [doc.page_content for doc in documents]

        answer = generate_answer(
            item["question"],
            contexts,
        )

        results.append(
            {
                "question": item["question"],
                "answer": answer,
                "contexts": contexts,
                "reference": item["reference"],
            }
        )

        print("\nQuestion:", item["question"])
        print("Answer:", answer)

    return results


if __name__ == "__main__":
    build_dataset()