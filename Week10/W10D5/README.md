# W10D5: Stateful Customer Support Agent

## Description

Built a stateful customer support agent using LangGraph and Ollama. The agent maintains conversation history across multiple interactions and tracks execution using MLflow.

## Tools Used

* CrewAI
* LangGraph
* MLflow
* Ragas
* MLOps
* Ollama
* Python
* Pytest

## Features

* Stateful conversation memory
* Customer support responses
* LangGraph workflow
* MLflow experiment tracking
* Automated tests

## Testing

Three automated tests were executed using pytest.

Result: 3 passed.

## How to Run

```powershell
pip install -r requirements.txt
python customer_support_agent.py
pytest -v
```
