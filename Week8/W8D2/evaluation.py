from rag_pipeline import retrieve


QA_PAIRS = [
    {
        "question": "What is Artificial Intelligence?",
        "reference": (
            "Artificial Intelligence is the field of computer science "
            "focused on creating systems that perform tasks requiring "
            "human-like intelligence."
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
            "Deep Learning is a subset of machine learning that uses "
            "neural networks with multiple layers."
        ),
    },
    {
        "question": "What is NLP?",
        "reference": (
            "Natural Language Processing enables computers to understand, "
            "process, and generate human language."
        ),
    },
    {
        "question": "What is RAG?",
        "reference": (
            "RAG combines information retrieval with large language "
            "models by retrieving relevant documents before generation."
        ),
    },
    {
        "question": "What is a vector database?",
        "reference": (
            "A vector database stores numerical representations called "
            "embeddings and supports similarity search."
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
            "Embeddings are numerical representations of data used "
            "for semantic similarity searches."
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
            "RAG evaluation measures retrieval and generation quality "
            "using metrics such as faithfulness and context recall."
        ),
    },
]


def create_evaluation_data(k=3, chunk_size=500):
    data = []

    for item in QA_PAIRS:
        retrieved = retrieve(
            item["question"],
            k=k,
            chunk_size=chunk_size,
        )

        contexts = [doc.page_content for doc in retrieved]

        # Temporary answer based on retrieved context.
        answer = contexts[0] if contexts else ""

        data.append(
            {
                "question": item["question"],
                "answer": answer,
                "contexts": contexts,
                "reference": item["reference"],
            }
        )

    return data


if __name__ == "__main__":
    results = create_evaluation_data()

    print(f"Generated {len(results)} evaluation samples.")

    for i, item in enumerate(results, 1):
        print(f"\nQ{i}: {item['question']}")
        print(f"Context count: {len(item['contexts'])}")