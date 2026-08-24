import ollama

SYSTEM_PROMPT = """
You are a helpful AI/ML mentor.
Explain technical concepts clearly and simply.
Use short examples whenever useful.
"""

def ask_ollama(question):
    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]
    )
    return response["message"]["content"]


if __name__ == "__main__":
    prompts = [
        "What is machine learning?",
        "What is RAG?",
        "What is an embedding?",
        "What is a vector database?",
        "Why are local LLMs useful?"
    ]

    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'=' * 60}")
        print(f"Prompt {i}: {prompt}")
        print("-" * 60)
        print(ask_ollama(prompt))