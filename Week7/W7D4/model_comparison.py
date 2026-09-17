import ollama

questions = [
    "Explain machine learning in simple words.",
    "What is Retrieval Augmented Generation (RAG)?",
    "Why is data preprocessing important?"
]

models = ["llama3.2:3b", "qwen2.5:3b"]


def ask_model(model, question):
    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a technical mentor. Give clear and concise answers."
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )

    return response["message"]["content"]


for question in questions:
    print("\n" + "=" * 80)
    print("QUESTION:", question)

    for model in models:
        print("\n" + "-" * 80)
        print("MODEL:", model)
        print(ask_model(model, question))