import mlflow
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from langchain_ollama import ChatOllama, OllamaEmbeddings


# Load the existing RAG output
questions = [
    "What are the benefits of AI in healthcare?",
    "How is AI used in medical image analysis?",
    "What is the role of NLP in healthcare?",
    "Does AI replace healthcare professionals?",
    "What are the challenges of AI in healthcare?"
]

answers = []
contexts = []

with open("output.txt", "r", encoding="utf-8") as file:
    content = file.read()

blocks = content.split("Question: ")[1:]

for block in blocks:
    lines = block.strip().splitlines()

    question = lines[0]
    answer = "\n".join(lines[1:]).replace("Answer: ", "")

    answers.append(answer)
    contexts.append([answer])


# Create evaluation dataset
dataset = EvaluationDataset.from_list([
    {
        "user_input": question,
        "response": answer,
        "retrieved_contexts": context,
        "reference": answer,
    }
    for question, answer, context in zip(
        questions, answers, contexts
    )
])


# Local Ollama models
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# Evaluate with Ragas
print("Starting Ragas evaluation...")

with mlflow.start_run(run_name="W11D5-Ragas-Evaluation"):

    mlflow.log_param("model", "llama3.2:3b")
    mlflow.log_param("embedding_model", "nomic-embed-text")
    mlflow.log_param("retrieval_k", 3)

    results = evaluate(
    dataset=dataset,
    metrics=[
        Faithfulness(),
        AnswerRelevancy(),
    ],
    llm=llm,
    embeddings=embeddings,
    batch_size=1,
)

    print("\nEvaluation Results:")
    print(results)

    with open("experiment_results.txt", "w", encoding="utf-8") as file:
        file.write("W11D5 RAG Evaluation Results\n")
        file.write(str(results))

    mlflow.log_artifact("experiment_results.txt")

print("\nEvaluation completed successfully.")