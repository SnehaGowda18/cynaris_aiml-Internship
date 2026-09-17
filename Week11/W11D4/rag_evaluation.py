import json
import time
from pathlib import Path

from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from ragas import evaluate, EvaluationDataset
from ragas.embeddings import LangchainEmbeddingsWrapper

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
CHROMA_DIR = BASE_DIR / "chroma_db"

OUTPUT_DIR.mkdir(exist_ok=True)

# Evaluate only two questions for faster execution.
# Change to 10 after the script works successfully.
EVALUATION_LIMIT = 2

OLLAMA_MODEL = "llama3.2:3b"

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

BASELINE_K = 2
OPTIMIZED_K = 1


# ============================================================
# QUESTIONS AND REFERENCES
# ============================================================

qa_pairs = [
    {
        "question": "What are the benefits of AI in healthcare?",
        "reference": (
            "AI can support diagnosis, medical image analysis, "
            "patient monitoring, drug discovery, and administrative work."
        ),
    },
    {
        "question": "How is AI used in medical image analysis?",
        "reference": (
            "AI analyses X-rays, CT scans, MRI scans, and ultrasound images "
            "to identify patterns that may be difficult for humans to notice."
        ),
    },
    {
        "question": "What is the role of NLP in healthcare?",
        "reference": (
            "NLP processes clinical notes, medical reports, patient messages, "
            "and research papers to extract useful information."
        ),
    },
    {
        "question": "Does AI replace healthcare professionals?",
        "reference": (
            "AI supports healthcare professionals but does not completely "
            "replace them. Final clinical decisions should remain with qualified professionals."
        ),
    },
    {
        "question": "What are the challenges of AI in healthcare?",
        "reference": (
            "Challenges include privacy risks, security concerns, biased data, "
            "lack of explainability, incorrect predictions, and regulatory requirements."
        ),
    },
    {
        "question": "How can AI improve patient monitoring?",
        "reference": (
            "AI can analyse vital signs, patient records, and healthcare information "
            "to help identify possible problems early."
        ),
    },
    {
        "question": "How is AI used in drug discovery?",
        "reference": (
            "AI analyses biological data, predicts molecular properties, and identifies "
            "candidate compounds during early drug discovery."
        ),
    },
    {
        "question": "How can AI help hospital administration?",
        "reference": (
            "AI can support appointment scheduling, billing, document processing, "
            "and hospital resource management."
        ),
    },
    {
        "question": "How can patient information be protected in AI systems?",
        "reference": (
            "Patient information can be protected using access controls, encryption, "
            "auditing, and privacy policies."
        ),
    },
    {
        "question": "Why is human oversight important in healthcare AI?",
        "reference": (
            "Human oversight is important because AI may produce incorrect or biased "
            "results. Professionals should verify important recommendations."
        ),
    },
]


# ============================================================
# MODEL INITIALIZATION
# ============================================================

def load_models():
    print("Loading Ollama model...")

    llm = Ollama(
        model=OLLAMA_MODEL,
        temperature=0,
        num_predict=150,
    )

    print("Loading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    return llm, embeddings, ragas_embeddings


# ============================================================
# VECTOR DATABASE
# ============================================================

def create_vectorstore(embeddings):
    print("Creating vector database...")

    healthcare_text = """
    Artificial intelligence is increasingly used in healthcare.
    It can support disease diagnosis, medical image analysis,
    drug discovery, patient monitoring, and administrative work.

    AI systems can examine medical images such as X-rays, CT scans,
    MRI scans, and ultrasound images. These systems can identify
    patterns that may be difficult for humans to notice. However,
    AI predictions should be reviewed by qualified healthcare
    professionals.

    Natural language processing is used to process and understand
    clinical notes, medical reports, patient messages, and research
    papers. NLP can help extract important information from large
    amounts of medical text.

    AI does not completely replace healthcare professionals.
    Instead, it can support doctors, nurses, and other professionals
    by providing additional information and reducing repetitive work.
    The final clinical decision should remain with qualified
    professionals.

    The main challenges of AI in healthcare include privacy risks,
    security concerns, biased training data, lack of explainability,
    incorrect predictions, limited high-quality data, and the need
    for regulatory compliance.

    AI can improve patient monitoring by analysing vital signs,
    patient records, and other healthcare information. Early warning
    systems may help healthcare staff identify possible problems.

    AI can support drug discovery by analysing biological data,
    predicting molecular properties, and identifying possible
    candidate compounds. This may reduce the time required during
    the early stages of research.

    AI can improve hospital administration through appointment
    scheduling, billing support, document processing, and resource
    management.

    Healthcare AI systems must protect patient information.
    Appropriate access controls, encryption, auditing, and privacy
    policies are important for protecting sensitive medical data.

    Human oversight is necessary because AI systems may produce
    incorrect or biased results. Healthcare professionals should
    verify important recommendations before taking action.
    """

    document = Document(page_content=healthcare_text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    split_documents = splitter.split_documents([document])

    vectorstore = Chroma.from_documents(
        documents=split_documents,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    return vectorstore


# ============================================================
# ANSWER GENERATION
# ============================================================

def generate_answer(llm, question, context):
    prompt = f"""
Answer the question using only the context below.

Context:
{context}

Question:
{question}

Give a short answer in two or three sentences.
Do not use JSON.
Do not add information outside the context.

Answer:
"""

    response = llm.invoke(prompt)
    return response.strip()


# ============================================================
# DATASET GENERATION
# ============================================================

def generate_dataset(llm, vectorstore, k, output_filename):
    output_file = OUTPUT_DIR / output_filename

    # Reuse existing data to avoid another long generation process.
    if output_file.exists():
        print(f"\nUsing existing file: {output_file}")

        with open(output_file, "r", encoding="utf-8") as file:
            return json.load(file)

    print(f"\nGenerating dataset with retrieval k={k}...")

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": k}
    )

    records = []

    for index, item in enumerate(qa_pairs, start=1):
        question = item["question"]
        reference = item["reference"]

        retrieved_documents = retriever.invoke(question)

        contexts = [
            document.page_content
            for document in retrieved_documents
        ]

        combined_context = "\n\n".join(contexts)

        answer = generate_answer(
            llm=llm,
            question=question,
            context=combined_context,
        )

        record = {
            "user_input": question,
            "response": answer,
            "retrieved_contexts": contexts,
            "reference": reference,
        }

        records.append(record)

        print(f"Completed {index}/{len(qa_pairs)}")

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(
            records,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print(f"Saved: {output_file}")

    return records


# ============================================================
# RAGAS EVALUATION
# ============================================================

def run_ragas_evaluation(
    llm,
    ragas_embeddings,
    records,
    output_filename,
):
    print("\nStarting Ragas evaluation...")

    # Limit evaluation to avoid 40 slow local LLM jobs.
    selected_records = records[:EVALUATION_LIMIT]

    evaluation_dataset = EvaluationDataset.from_list(
        selected_records
    )

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    try:
        result = evaluate(
            dataset=evaluation_dataset,
            metrics=metrics,
            llm=llm,
            embeddings=ragas_embeddings,
            raise_exceptions=False,
        )

        result_dataframe = result.to_pandas()

        result_file = OUTPUT_DIR / output_filename

        result_dataframe.to_csv(
            result_file,
            index=False,
        )

        print(f"\nRagas result saved to: {result_file}")
        print(result_dataframe)

        return result_dataframe

    except Exception as error:
        print("\nRagas evaluation failed")
        print(f"Error type: {type(error).__name__}")
        print(f"Error message: {error}")

        error_file = OUTPUT_DIR / "ragas_error.txt"

        with open(error_file, "w", encoding="utf-8") as file:
            file.write(f"Error type: {type(error).__name__}\n")
            file.write(f"Error message: {error}\n")

        print(f"Error saved to: {error_file}")

        return None


# ============================================================
# SUMMARY
# ============================================================

def save_summary():
    summary_file = OUTPUT_DIR / "experiment_summary.txt"

    with open(summary_file, "w", encoding="utf-8") as file:
        file.write("RAGAS RAG EVALUATION SUMMARY\n")
        file.write("=" * 40 + "\n\n")
        file.write(f"Total Q&A pairs generated: {len(qa_pairs)}\n")
        file.write(f"Questions evaluated: {EVALUATION_LIMIT}\n")
        file.write(f"Baseline retrieval k: {BASELINE_K}\n")
        file.write(f"Optimized retrieval k: {OPTIMIZED_K}\n")
        file.write(f"Chunk size: {CHUNK_SIZE}\n")
        file.write(f"Chunk overlap: {CHUNK_OVERLAP}\n\n")
        file.write(
            "The optimized pipeline retrieves fewer chunks "
            "to reduce unnecessary context.\n"
        )
        file.write(
            "Evaluation was limited because local Ollama "
            "LLM-based Ragas evaluation is time-consuming.\n"
        )

    print(f"\nSummary saved to: {summary_file}")


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":
    start_time = time.time()

    print("=" * 60)
    print("FAST RAGAS EVALUATION")
    print("=" * 60)

    llm, embeddings, ragas_embeddings = load_models()

    vectorstore = create_vectorstore(embeddings)

    baseline_records = generate_dataset(
        llm=llm,
        vectorstore=vectorstore,
        k=BASELINE_K,
        output_filename="baseline_results.json",
    )

    optimized_records = generate_dataset(
        llm=llm,
        vectorstore=vectorstore,
        k=OPTIMIZED_K,
        output_filename="optimized_results.json",
    )

    run_ragas_evaluation(
        llm=llm,
        ragas_embeddings=ragas_embeddings,
        records=baseline_records,
        output_filename="baseline_ragas_results.csv",
    )

    run_ragas_evaluation(
        llm=llm,
        ragas_embeddings=ragas_embeddings,
        records=optimized_records,
        output_filename="optimized_ragas_results.csv",
    )

    save_summary()

    elapsed_minutes = (time.time() - start_time) / 60

    print("\n" + "=" * 60)
    print("PROCESS COMPLETED")
    print(f"Total time: {elapsed_minutes:.2f} minutes")
    print("=" * 60)