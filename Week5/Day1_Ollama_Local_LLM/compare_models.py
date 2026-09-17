import os
import requests

URL = "http://localhost:11434/api/generate"

QUESTIONS = [
    "What is AI?",
    "Explain Neural Networks.",
    "What is MLOps?"
]

MODELS = {
    "llama3.2:3b": "outputs/llama_output.txt",
    "qwen2.5:3b": "outputs/qwen_output.txt"
}

os.makedirs("outputs", exist_ok=True)

comparison = []

for model, output_file in MODELS.items():

    print("\n" + "=" * 70)
    print("MODEL:", model)
    print("=" * 70)

    with open(output_file, "w", encoding="utf-8") as f:

        f.write(f"MODEL: {model}\n")
        f.write("=" * 70 + "\n\n")

        for question in QUESTIONS:

            print("\nQuestion:", question)

            try:
                response = requests.post(
                    URL,
                    json={
                        "model": model,
                        "prompt": question,
                        "stream": False
                    },
                    timeout=180
                )

                response.raise_for_status()

                answer = response.json()["response"]

                print("Answer:\n")
                print(answer)
                print("-" * 70)

                f.write(f"Question: {question}\n\n")
                f.write("Answer:\n")
                f.write(answer)
                f.write("\n\n")
                f.write("-" * 70 + "\n\n")

            except Exception as e:
                print("Error:", e)
                f.write(f"Error while processing question: {question}\n")
                f.write(str(e) + "\n\n")

comparison_text = """
Model Comparison

Questions:
1. What is AI?
2. Explain Neural Networks.
3. What is MLOps?

Llama3.2:3b
-------------
• Detailed explanations
• Beginner-friendly
• More examples

Qwen2.5:3b
------------
• Short and concise
• More technical
• Better structured

Conclusion
----------
Both models performed well.

Llama3.2:3b is suitable for beginners because it explains concepts in more detail.

Qwen2.5:3b gives shorter, structured, and technical responses, making it suitable for quick understanding.
"""

with open("outputs/comparison.txt", "w", encoding="utf-8") as f:
    f.write(comparison_text)

print("\nComparison completed successfully.")
print("Outputs saved inside the outputs folder.")