import os
import requests

URL = "http://localhost:11434/api/generate"

SYSTEM_PROMPT = """
You are a helpful AI assistant.
Answer clearly, accurately, and briefly.
"""

prompts = [
    "What is Artificial Intelligence?",
    "Explain Machine Learning in simple words.",
    "What is Deep Learning?",
    "Why is Python popular for AI?",
    "Give five benefits of local LLMs."
]

os.makedirs("outputs", exist_ok=True)

with open("outputs/llama_output.txt", "w", encoding="utf-8") as f:

    for prompt in prompts:

        response = requests.post(
            URL,
            json={
                "model": "llama3.2:3b",
                "prompt": f"System: {SYSTEM_PROMPT}\nUser: {prompt}",
                "stream": False
            },
            timeout=180
        )

        answer = response.json()["response"]

        print("=" * 60)
        print("Prompt:")
        print(prompt)
        print("\nResponse:")
        print(answer)

        f.write("=" * 60 + "\n")
        f.write("Prompt:\n")
        f.write(prompt + "\n\n")
        f.write("Response:\n")
        f.write(answer + "\n\n")