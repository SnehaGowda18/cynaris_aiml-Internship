def test_assistant_project_files_exist():
    """Verify the main W8D5 project files are present."""
    from pathlib import Path

    project_root = Path(__file__).parent.parent

    required_files = [
        "crew.py",
        "graph.py",
        "evaluation.py",
        "mlflow_tracking.py",
        "vector_store.py",
    ]

    for filename in required_files:
        assert (project_root / filename).exists(), f"Missing {filename}"