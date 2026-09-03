from pathlib import Path
import matplotlib.pyplot as plt


question = Path("research_question.txt").read_text(encoding="utf-8").strip()
answer = Path("research_answer.txt").read_text(encoding="utf-8").strip()

retrieved_file = Path("retrieved_documents.txt")
retrieved_text = retrieved_file.read_text(encoding="utf-8").strip()

document_count = 0
if retrieved_text:
    document_count = len(
        [line for line in retrieved_text.splitlines() if line.strip()]
    )


fig = plt.figure(figsize=(12, 7))
ax = fig.add_axes([0, 0, 1, 1])
ax.axis("off")

ax.text(
    0.05,
    0.92,
    "Local AI Research Assistant",
    fontsize=24,
    fontweight="bold",
)

ax.text(
    0.05,
    0.84,
    "Research Question",
    fontsize=14,
    fontweight="bold",
)

ax.text(
    0.05,
    0.79,
    question,
    fontsize=12,
    wrap=True,
)

ax.text(
    0.05,
    0.68,
    "Generated Answer",
    fontsize=14,
    fontweight="bold",
)

ax.text(
    0.05,
    0.62,
    answer,
    fontsize=12,
    wrap=True,
    verticalalignment="top",
)

ax.text(
    0.05,
    0.32,
    "Pipeline Evidence",
    fontsize=14,
    fontweight="bold",
)

evidence = (
    "LLM Model: llama3.2:3b\n"
    "Framework: LangGraph\n"
    "Vector Database: ChromaDB\n"
    "Embedding Method: TF-IDF\n"
    f"Retrieved Documents: {document_count}"
)

ax.text(
    0.05,
    0.26,
    evidence,
    fontsize=12,
    linespacing=1.6,
)

plt.savefig("output.png", dpi=150, bbox_inches="tight")
plt.close()

print("output.png created successfully.")