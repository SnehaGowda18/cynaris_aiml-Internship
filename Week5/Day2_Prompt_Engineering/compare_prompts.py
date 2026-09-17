import requests

URL="http://localhost:11434/api/generate"

question="Explain Artificial Intelligence."

without_system=requests.post(

    URL,

    json={
        "model":"llama3.2:3b",
        "prompt":question,
        "stream":False
    }

).json()["response"]

with_system=requests.post(

    URL,

    json={

        "model":"llama3.2:3b",

        "prompt":
        "System: You are an AI teacher. "
        "Explain using simple examples.\n"
        f"User:{question}",

        "stream":False
    }

).json()["response"]

print("="*70)
print("WITHOUT SYSTEM PROMPT\n")
print(without_system)

print("="*70)
print("WITH SYSTEM PROMPT\n")
print(with_system)

comparison=f"""
Comparison

Question:
Explain Artificial Intelligence

Without System Prompt

{without_system}

------------------------------------------------------

With System Prompt

{with_system}

------------------------------------------------------

Observation

System prompts control:

1. Tone
2. Detail
3. Response quality
4. Audience
5. Style
"""

with open(

    "outputs/comparison.txt",

    "w",

    encoding="utf-8"

) as f:

    f.write(comparison)

print("\nComparison saved.")