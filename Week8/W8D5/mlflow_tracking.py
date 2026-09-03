"""
MLflow tracking for the Local AI Research Assistant.

Tracks:
- Research question
- Local LLM model
- LangGraph framework
- ChromaDB vector database
- Retrieved documents
- Answer length
- Execution time
- Question and answer artifacts
"""

import os
import time

import mlflow

from graph import research_graph


# ---------------------------------------------------------
# MLflow configuration
# ---------------------------------------------------------

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)


# ---------------------------------------------------------
# Research + MLflow tracking
# ---------------------------------------------------------

def run_research_with_tracking(question: str):
    """
    Run the LangGraph research assistant and track
    execution using MLflow.
    """

    print("\n" + "=" * 60)
    print("LOCAL AI RESEARCH ASSISTANT")
    print("=" * 60)

    print(f"\nQuestion: {question}")

    # -----------------------------------------------------
    # Create or select MLflow experiment
    # -----------------------------------------------------

    print(
        "\n[1/5] Setting up MLflow experiment..."
    )

    mlflow.set_experiment(
        "Local-AI-Research-Assistant"
    )

    # Start timer
    start_time = time.time()

    # -----------------------------------------------------
    # Start MLflow run
    # -----------------------------------------------------

    with mlflow.start_run() as run:

        print(
            "[2/5] MLflow run started."
        )

        print(
            f"Run ID: {run.info.run_id}"
        )

        # -------------------------------------------------
        # Run LangGraph workflow
        # -------------------------------------------------

        print(
            "\n[3/5] Running LangGraph research workflow..."
        )

        print(
            "Retrieving documents from ChromaDB..."
        )

        result = research_graph.invoke(
            {
                "question": question,
                "documents": [],
                "answer": "",
            }
        )

        execution_time = (
            time.time() - start_time
        )

        # -------------------------------------------------
        # Extract results
        # -------------------------------------------------

        documents = result.get(
            "documents",
            []
        )

        answer = result.get(
            "answer",
            ""
        )

        document_count = len(
            documents
        )

        answer_length = len(
            answer
        )

        print(
            "[4/5] Research completed."
        )

        # -------------------------------------------------
        # Log MLflow parameters
        # -------------------------------------------------

        mlflow.log_param(
            "model",
            "llama3.2:3b"
        )

        mlflow.log_param(
            "framework",
            "LangGraph"
        )

        mlflow.log_param(
            "vector_database",
            "ChromaDB"
        )

        mlflow.log_param(
            "embedding_method",
            "TF-IDF"
        )

        mlflow.log_param(
            "retrieval_top_k",
            3
        )

        # -------------------------------------------------
        # Log MLflow metrics
        # -------------------------------------------------

        mlflow.log_metric(
            "retrieved_documents",
            document_count
        )

        mlflow.log_metric(
            "answer_length",
            answer_length
        )

        mlflow.log_metric(
            "execution_time_seconds",
            execution_time
        )

        # -------------------------------------------------
        # Create artifact files
        # -------------------------------------------------

        question_file = (
            "research_question.txt"
        )

        answer_file = (
            "research_answer.txt"
        )

        documents_file = (
            "retrieved_documents.txt"
        )

        # Save question
        with open(
            question_file,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(question)

        # Save answer
        with open(
            answer_file,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(answer)

        # Save retrieved documents
        with open(
            documents_file,
            "w",
            encoding="utf-8"
        ) as file:

            for index, document in enumerate(
                documents,
                start=1
            ):

                file.write(
                    f"\n--- Document {index} ---\n\n"
                )

                file.write(document)

                file.write("\n")

        # -------------------------------------------------
        # Log artifacts
        # -------------------------------------------------

        mlflow.log_artifact(
            question_file
        )

        mlflow.log_artifact(
            answer_file
        )

        mlflow.log_artifact(
            documents_file
        )

        print(
            "[5/5] MLflow metrics and artifacts logged."
        )

        # -------------------------------------------------
        # Display final result
        # -------------------------------------------------

        print(
            "\n" + "=" * 60
        )

        print(
            "RESEARCH RESULT"
        )

        print(
            "=" * 60
        )

        print("\nQuestion:")
        print(question)

        print("\nAnswer:")
        print(answer)

        print(
            f"\nDocuments Retrieved: "
            f"{document_count}"
        )

        print(
            f"Answer Length: "
            f"{answer_length}"
        )

        print(
            f"Execution Time: "
            f"{execution_time:.2f} seconds"
        )

        print(
            f"\nMLflow Run ID: "
            f"{run.info.run_id}"
        )

        print(
            "\n" + "=" * 60
        )

        print(
            "MLFLOW TRACKING COMPLETED SUCCESSFULLY"
        )

        print(
            "=" * 60
        )

        return result


# ---------------------------------------------------------
# Main program
# ---------------------------------------------------------

if __name__ == "__main__":

    print(
        "MLFLOW TRACKING SCRIPT STARTED"
    )

    question = (
        "What is Retrieval Augmented Generation?"
    )

    run_research_with_tracking(
        question
    )