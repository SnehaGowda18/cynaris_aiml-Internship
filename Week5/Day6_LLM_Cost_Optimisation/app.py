import os
import json
import pickle
import numpy as np
import pandas as pd
import tiktoken

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================================
# CREATE FOLDERS
# ==========================================================

os.makedirs("outputs", exist_ok=True)
os.makedirs("cache", exist_ok=True)

# ==========================================================
# MODEL PRICING
# ==========================================================

MODELS = {
    "GPT-4o": {"input": 5.0, "output": 15.0},
    "Claude Sonnet": {"input": 3.0, "output": 15.0},
    "Gemini Flash": {"input": 0.10, "output": 0.40},
    "GPT-4o Mini": {"input": 0.15, "output": 0.60},
}

USD_TO_INR = 87

# ==========================================================
# TOKEN ENCODER
# ==========================================================

enc = tiktoken.get_encoding("cl100k_base")

# ==========================================================
# READ PROMPTS
# ==========================================================

with open("data/prompts.txt", "r", encoding="utf-8") as f:
    prompts = [line.strip() for line in f if line.strip()]

records = []

# ==========================================================
# TOKEN AUDIT
# ==========================================================

for i, prompt in enumerate(prompts):

    input_tokens = len(enc.encode(prompt))

    output_tokens = max(10, int(input_tokens * 0.5))

    for model_name, price in MODELS.items():

        input_cost = input_tokens * price["input"] / 1_000_000

        output_cost = output_tokens * price["output"] / 1_000_000

        total_cost = input_cost + output_cost

        records.append({
            "Prompt No": i + 1,
            "Model": model_name,
            "Input Tokens": input_tokens,
            "Output Tokens": output_tokens,
            "Cost (USD)": round(total_cost, 8),
            "Cost (INR)": round(total_cost * USD_TO_INR, 6)
        })

df = pd.DataFrame(records)

df.to_csv("outputs/token_audit.csv", index=False)

# ==========================================================
# COST REPORT
# ==========================================================

summary = df.groupby("Model")["Cost (INR)"].sum()

with open("outputs/cost_report.txt", "w", encoding="utf-8") as f:

    f.write("=" * 50 + "\n")
    f.write("LLM COST REPORT\n")
    f.write("=" * 50 + "\n\n")

    for model, cost in summary.items():

        f.write(f"{model:<20} ₹ {cost:.6f}\n")

    f.write("\n")

    total = summary.sum()

    f.write(f"Total Estimated Cost : ₹ {total:.6f}\n")

# ==========================================================
# LOAD QA DATA
# ==========================================================

with open("data/qa_pairs.json", "r", encoding="utf-8") as f:
    qa = json.load(f)

questions = [item["question"] for item in qa]

answers = [item["answer"] for item in qa]

# ==========================================================
# TF-IDF SEMANTIC CACHE
# ==========================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=1000
)

question_vectors = vectorizer.fit_transform(questions)

pickle.dump(vectorizer,
            open("cache/tfidf_vectorizer.pkl", "wb"))

pickle.dump(question_vectors,
            open("cache/document_vectors.pkl", "wb"))

query = "Define Artificial Intelligence"

query_vector = vectorizer.transform([query])

similarity = cosine_similarity(
    query_vector,
    question_vectors
)[0]

best_index = np.argmax(similarity)

best_question = questions[best_index]

best_answer = answers[best_index]

best_score = similarity[best_index]
# ==========================================================
# SAVE SEMANTIC CACHE RESULT
# ==========================================================

with open("outputs/semantic_cache_results.txt", "w", encoding="utf-8") as f:

    f.write("=" * 50 + "\n")
    f.write("SEMANTIC CACHE RESULT\n")
    f.write("=" * 50 + "\n\n")

    f.write(f"User Query          : {query}\n\n")

    f.write(f"Matched Question    : {best_question}\n\n")

    f.write(f"Similarity Score    : {best_score:.4f}\n\n")

    f.write("Cached Response\n")
    f.write("-" * 40 + "\n")

    f.write(best_answer)

# ==========================================================
# ROUTING STRATEGY
# ==========================================================

routing = """
==================================================
LLM ROUTING STRATEGY
==================================================

LOCAL MODEL
-----------
• Greetings
• Definitions
• Simple Q&A
• Small code examples
• Basic calculations
• FAQs

API MODEL
---------
• Long document summarization
• RAG applications
• Complex reasoning
• Report generation
• Multi-step coding
• Large context (>5K tokens)

Expected Query Distribution
---------------------------
Local Model : 70%
API Model   : 30%

Estimated Cost Reduction
------------------------
60% - 80%
"""

with open("outputs/routing_strategy.txt", "w", encoding="utf-8") as f:
    f.write(routing)

# ==========================================================
# DISPLAY SUMMARY
# ==========================================================

print("\n" + "=" * 60)
print("      W5D6 - LLM COST OPTIMISATION COMPLETED")
print("=" * 60)

print("\nGenerated Files")

print("----------------------------")
print("outputs/token_audit.csv")
print("outputs/cost_report.txt")
print("outputs/semantic_cache_results.txt")
print("outputs/routing_strategy.txt")
print("cache/tfidf_vectorizer.pkl")
print("cache/document_vectors.pkl")

print("\nToken Audit Preview\n")

print(df.head())

print("\nTotal Estimated Cost (INR)\n")

print(summary)

print("\nSemantic Cache Match")

print("----------------------------")
print("Query           :", query)
print("Matched Question:", best_question)
print("Similarity      :", round(best_score, 4))
print("Answer          :", best_answer)

print("\nProject Completed Successfully!")
print("=" * 60)