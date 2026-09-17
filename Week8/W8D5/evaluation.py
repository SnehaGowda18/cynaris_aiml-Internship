"""
Ragas evaluation for the Local AI Research Assistant.

Evaluates generated answers using:
- Faithfulness
- Answer Relevancy
"""

from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
)


def run_evaluation():

    # Test question
    question = "What is Retrieval Augmented Generation?"

    # Expected answer based on the local research documents
    answer = (
        "Retrieval Augmented Generation (RAG) combines "
        "information retrieval with a generative language model. "
        "The system first retrieves relevant documents from a "
        "knowledge base and then provides them to the language "
        "model as context for generating an answer."
    )

    # Retrieved context from the local documents
    contexts = [
        (
            "Retrieval Augmented Generation, or RAG, combines "
            "information retrieval with generative language models. "
            "A RAG system first retrieves relevant documents from "
            "a knowledge base and then provides those documents "
            "to an LLM as context for generating an answer."
        )
    ]

    # Create Ragas dataset
    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
    }

    dataset = Dataset.from_dict(data)

    print("\n" + "=" * 60)
    print("RAGAS EVALUATION")
    print("=" * 60)

    print("\nQuestion:")
    print(question)

    print("\nRunning evaluation...")

    try:

        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
            ],
        )

        print("\nEvaluation completed.")

        print("\nResults:")
        print(result)

        # Save results
        with open(
            "ragas_results.txt",
            "w",
            encoding="utf-8",
        ) as file:

            file.write(
                "RAGAS Evaluation Results\n"
            )

            file.write(
                "========================\n\n"
            )

            file.write(
                str(result)
            )

        print(
            "\nResults saved to ragas_results.txt"
        )

    except Exception as error:

        print(
            "\nRagas evaluation failed."
        )

        print(
            f"Error: {error}"
        )


if __name__ == "__main__":
    run_evaluation()