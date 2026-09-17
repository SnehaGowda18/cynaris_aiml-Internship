from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

documents = {
    "document1_machine_learning.pdf": (
        "Machine Learning",
        """
Machine learning is a branch of artificial intelligence that enables
computers to learn patterns from data and make predictions or decisions.

Supervised learning uses labelled data for tasks such as classification
and regression. Unsupervised learning discovers patterns in unlabelled
data using techniques such as clustering.

A machine learning workflow includes data collection, preprocessing,
feature engineering, model training, validation, evaluation, and deployment.

Common evaluation metrics include accuracy, precision, recall, F1-score,
mean squared error, and mean absolute error.

Machine learning is used in cybersecurity, healthcare, finance,
recommendation systems, computer vision, and natural language processing.
"""
    ),

    "document2_artificial_intelligence.pdf": (
        "Artificial Intelligence",
        """
Artificial intelligence, or AI, is a field of computer science concerned
with creating systems that can perform tasks associated with human
intelligence.

AI systems can perform learning, reasoning, planning, perception,
language understanding, and decision making.

Machine learning is an important approach to artificial intelligence.
Modern AI includes machine learning, deep learning, natural language
processing, computer vision, robotics, and generative AI.

AI applications include virtual assistants, fraud detection, medical
diagnosis, recommendation systems, cybersecurity monitoring, and robotics.

Responsible AI considers fairness, privacy, security, transparency,
accountability, and bias.
"""
    ),

    "document3_deep_learning.pdf": (
        "Deep Learning",
        """
Deep learning is a subfield of machine learning based on artificial
neural networks containing multiple layers.

A neural network normally contains an input layer, hidden layers,
and an output layer. The network learns parameters from training data.

Convolutional neural networks are widely used for computer vision.
Transformers are widely used for language and other sequential tasks.

Training involves forward propagation, calculating a loss function,
backpropagation, and optimization.

Deep learning is used for image classification, object detection,
speech recognition, natural language processing, medical imaging,
recommendation systems, and cybersecurity.
"""
    ),

    "document4_nlp.pdf": (
        "Natural Language Processing",
        """
Natural Language Processing, or NLP, enables computers to process,
understand, and generate human language.

Common NLP tasks include text classification, sentiment analysis,
named entity recognition, machine translation, question answering,
summarization, and information extraction.

Traditional NLP uses techniques such as tokenization, stemming,
lemmatization, stop-word removal, and TF-IDF.

Modern NLP commonly uses transformer architectures and embeddings.
Embeddings represent words, sentences, or documents as numerical vectors.

NLP is used in chatbots, search engines, customer support,
document analysis, spam detection, and information retrieval.
"""
    ),

    "document5_rag.pdf": (
        "Retrieval Augmented Generation",
        """
Retrieval Augmented Generation, or RAG, combines information retrieval
with large language models.

A RAG system retrieves relevant documents from an external knowledge
source and provides them as context to a language model.

A typical RAG pipeline includes document loading, text splitting,
embedding generation, vector storage, retrieval, prompt construction,
and response generation.

BM25 is a keyword-based retrieval method. Dense retrieval uses vector
embeddings to identify semantically similar documents.

BM25 works well when important query keywords occur in documents.
Dense retrieval can find relevant information even when different
words are used for the same meaning.

RAG evaluation can include precision, recall, MRR, nDCG, faithfulness,
and answer correctness.
"""
    )
}

styles = getSampleStyleSheet()

title_style = styles["Title"]
title_style.alignment = TA_CENTER

body_style = styles["BodyText"]
body_style.leading = 16


for filename, (title, content) in documents.items():

    output_path = DATA_DIR / filename

    pdf = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50
    )

    story = [
        Paragraph(title, title_style),
        Spacer(1, 20)
    ]

    for paragraph in content.strip().split("\n\n"):
        story.append(
            Paragraph(
                paragraph.strip(),
                body_style
            )
        )
        story.append(Spacer(1, 10))

    pdf.build(story)

    print(f"Created: {filename}")


print("\nSuccessfully created 5 PDF documents.")