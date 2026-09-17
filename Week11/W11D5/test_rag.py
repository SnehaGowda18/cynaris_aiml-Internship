import os


def test_required_files():
    files = [
        "rag_pipeline.py",
        "rag_evaluation.py",
        "optimize_rag.py",
        "healthcare.txt",
        "output.txt",
        "experiment_results.txt",
        "optimization_results.txt",
        "README.md",
    ]

    for file in files:
        assert os.path.exists(file), f"Missing file: {file}"


def test_evaluation_results():
    with open("experiment_results.txt", "r", encoding="utf-8") as file:
        content = file.read()

    assert "faithfulness" in content
    assert "answer_relevancy" in content


def test_optimization_results():
    with open("optimization_results.txt", "r", encoding="utf-8") as file:
        content = file.read()

    assert "Retrieval k=2" in content
    assert "Retrieval k=3" in content


if __name__ == "__main__":
    test_required_files()
    test_evaluation_results()
    test_optimization_results()

    print("All tests passed successfully.")