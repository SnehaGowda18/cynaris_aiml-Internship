from pathlib import Path
import csv


QUESTIONS = [
    "What is machine learning?",
    "What is artificial intelligence?",
    "What is deep learning?",
    "What is natural language processing?",
    "What is a neural network?",
    "What is supervised learning?",
    "What is unsupervised learning?",
    "What is reinforcement learning?",
    "What is model evaluation?",
    "What is retrieval augmented generation?",
]


BASE_DIR = Path(__file__).resolve().parent
EVIDENCE_DIR = BASE_DIR / "evidence"

EVIDENCE_DIR.mkdir(exist_ok=True)


def get_valid_score(method, question_number):

    while True:

        try:

            value = int(
                input(
                    f"{method} relevant results (0-3): "
                )
            )

            if 0 <= value <= 3:
                return value

            print("Enter a number between 0 and 3.")

        except ValueError:

            print("Please enter a valid number.")


def main():

    print("=" * 70)
    print("W8D3 — BM25 vs DENSE RETRIEVAL EVALUATION")
    print("=" * 70)

    print("\nEvaluation metric: Precision@3")

    print(
        "\nPrecision@3 = Relevant retrieved documents / 3"
    )

    print(
        "\nInspect the top 3 results from both "
        "pipeline_bm25.py and pipeline_dense.py."
    )

    print(
        "Enter how many of the 3 results are relevant."
    )

    print("\n" + "-" * 70)

    results = []

    for number, question in enumerate(
        QUESTIONS,
        start=1
    ):

        print(f"\nQuestion {number}")
        print(question)

        print()

        bm25_relevant = get_valid_score(
            "BM25",
            number
        )

        dense_relevant = get_valid_score(
            "Dense",
            number
        )

        bm25_precision = bm25_relevant / 3
        dense_precision = dense_relevant / 3

        results.append(
            {
                "Question Number": number,
                "Question": question,
                "BM25 Relevant": bm25_relevant,
                "BM25 Precision@3": round(
                    bm25_precision,
                    2
                ),
                "Dense Relevant": dense_relevant,
                "Dense Precision@3": round(
                    dense_precision,
                    2
                ),
            }
        )

    # --------------------------------------------------------
    # Calculate averages
    # --------------------------------------------------------

    average_bm25 = sum(
        item["BM25 Precision@3"]
        for item in results
    ) / len(results)

    average_dense = sum(
        item["Dense Precision@3"]
        for item in results
    ) / len(results)

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n")
    print("=" * 90)
    print("FINAL BM25 vs DENSE COMPARISON")
    print("=" * 90)

    print(
        f"{'Q':<4}"
        f"{'BM25 P@3':<12}"
        f"{'Dense P@3':<12}"
        f"{'Question'}"
    )

    print("-" * 90)

    for item in results:

        print(
            f"{item['Question Number']:<4}"
            f"{item['BM25 Precision@3']:<12.2f}"
            f"{item['Dense Precision@3']:<12.2f}"
            f"{item['Question']}"
        )

    print("-" * 90)

    print(
        f"\nAverage BM25 Precision@3: "
        f"{average_bm25:.2f}"
    )

    print(
        f"Average Dense Precision@3: "
        f"{average_dense:.2f}"
    )

    # --------------------------------------------------------
    # Determine winner
    # --------------------------------------------------------

    if average_dense > average_bm25:

        winner = "Dense Retrieval"

    elif average_bm25 > average_dense:

        winner = "BM25"

    else:

        winner = "Tie"

    print(
        f"\nBetter retrieval method: {winner}"
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    csv_file = EVIDENCE_DIR / "retrieval_comparison.csv"

    with open(
        csv_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = [
            "Question Number",
            "Question",
            "BM25 Relevant",
            "BM25 Precision@3",
            "Dense Relevant",
            "Dense Precision@3",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(results)

    # --------------------------------------------------------
    # Save text report
    # --------------------------------------------------------

    report_file = EVIDENCE_DIR / "evaluation_results.txt"

    with open(
        report_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "W8D3 — BM25 vs Dense Retrieval Evaluation\n"
        )

        file.write("=" * 60 + "\n\n")

        for item in results:

            file.write(
                f"Question {item['Question Number']}: "
                f"{item['Question']}\n"
            )

            file.write(
                f"BM25 Precision@3: "
                f"{item['BM25 Precision@3']:.2f}\n"
            )

            file.write(
                f"Dense Precision@3: "
                f"{item['Dense Precision@3']:.2f}\n\n"
            )

        file.write(
            f"Average BM25 Precision@3: "
            f"{average_bm25:.2f}\n"
        )

        file.write(
            f"Average Dense Precision@3: "
            f"{average_dense:.2f}\n"
        )

        file.write(
            f"Better retrieval method: {winner}\n"
        )

    print("\nEvidence files created:")

    print(csv_file)

    print(report_file)


if __name__ == "__main__":
    main()