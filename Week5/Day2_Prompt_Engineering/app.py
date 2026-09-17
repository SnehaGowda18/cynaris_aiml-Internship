import os
import requests
from prompt_templates import SYSTEM_PROMPTS

URL = "http://localhost:11434/api/generate"

os.makedirs("outputs", exist_ok=True)

PROMPTS = {
    "Teacher":
        "Explain Machine Learning for a 10-year-old.",

    "Python Expert":
        "Write a Python function to reverse a string.",

    "Cybersecurity Mentor":
        "Explain phishing attacks.",

    "Interviewer":
        "Ask me five Python interview questions.",

    "Travel Guide":
        "Plan a two-day trip to Mysuru."
}

for role in SYSTEM_PROMPTS:

    print("="*60)
    print(role)
    print("="*60)

    response = requests.post(
        URL,
        json={
            "model":"llama3.2:3b",
            "prompt":
                f"System: {SYSTEM_PROMPTS[role]}\n"
                f"User: {PROMPTS[role]}",
            "stream":False
        },
        timeout=180
    )

    answer = response.json()["response"]

    print(answer)

    filename = role.lower().replace(" ","_") + "_output.txt"

    with open(
        f"outputs/{filename}",
        "w",
        encoding="utf-8"
    ) as f:

        f.write("ROLE:\n")
        f.write(role)
        f.write("\n\n")

        f.write("PROMPT:\n")
        f.write(PROMPTS[role])
        f.write("\n\n")

        f.write("RESPONSE:\n")
        f.write(answer)