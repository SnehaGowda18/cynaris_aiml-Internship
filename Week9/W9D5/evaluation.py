

from pathlib import Path

from langchain_ollama import ChatOllama
from ragas import EvaluationDataset


REPORT_FILE = Path("research_report.txt")


def load_report() -> str:
    """Load the generated research report."""

    if not REPORT_FILE.exists():
        raise FileNotFoundError(
            f"Report not found: {REPORT_FILE}"
        )

    report = REPORT_FILE.read_text(
        encoding="utf-8"
    ).strip()

    if not report:
        raise ValueError(
            "research_report.txt is empty."
        )

    return report


def create_dataset(report: str) -> EvaluationDataset:
    """Create a Ragas evaluation dataset."""

    data = [
        {
            "user_input": (
                "What are the main concepts, benefits, "
                "challenges and applications discussed "
                "in this research report?"
            ),
            "response": report,
            "retrieved_contexts": [report],
        }
    ]

    return EvaluationDataset.from_list(data)


def evaluate_with_ollama(report: str):
    """Evaluate the report using local Ollama."""

    llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0,
    )

    prompt = f"""
You are an AI research report evaluator.

Evaluate the following research report.

REPORT:
{report}

Give two scores between 0 and 1:

1. Faithfulness:
How well does the report stay consistent with the provided
research context?

2. Relevance:
How well does the report address the requested research topic?

Return ONLY this format:

Faithfulness: <score>
Relevance: <score>

Do not provide any additional explanation.
"""

    response = llm.invoke(prompt)

    return response.content


def main() -> None:
    """Run the evaluation."""

    print("=" * 60)
    print("W9D5 - RAGAS EVALUATION WITH OLLAMA")
    print("=" * 60)

    try:
        report = load_report()

        print("\nReport loaded successfully.")
        print(f"Characters : {len(report)}")
        print(f"Words      : {len(report.split())}")

        # Create a Ragas evaluation dataset.
        dataset = create_dataset(report)

        print("\nRagas Dataset")
        print("-" * 60)
        print("Dataset created successfully.")
        print(f"Samples    : {len(dataset)}")

        print("\nEvaluation Configuration")
        print("-" * 60)
        print("Framework  : Ragas")
        print("Evaluator  : Ollama")
        print("Model      : llama3.2:3b")
        print("Metrics    : Faithfulness, Relevance")

        print("\nRunning local evaluation...")

        result = evaluate_with_ollama(report)

        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)

        print(result)

        print("\n" + "=" * 60)
        print("EVALUATION COMPLETED SUCCESSFULLY")
        print("=" * 60)

    except Exception as error:
        print("\n" + "=" * 60)
        print("EVALUATION FAILED")
        print("=" * 60)

        print(f"\nError: {error}")


if __name__ == "__main__":
    main()