from reportlab.pdfgen import canvas
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

documents = {
    "document1.pdf": (
        "Artificial Intelligence",
        """Artificial Intelligence (AI) is a field of computer science that focuses
on creating systems capable of performing tasks that normally require human
intelligence.

AI systems can perform tasks such as reasoning, learning, problem solving,
speech recognition, image understanding, and decision making.

Machine learning is an important part of artificial intelligence. Instead of
being explicitly programmed for every task, machine learning systems learn
patterns from data.

AI is used in healthcare, finance, transportation, education, robotics,
cybersecurity, and many other industries."""
    ),

    "document2.pdf": (
        "Machine Learning",
        """Machine Learning is a branch of artificial intelligence that enables
computers to learn patterns from data and make predictions or decisions.

The three common types of machine learning are supervised learning,
unsupervised learning, and reinforcement learning.

Supervised learning uses labeled training data. Examples include linear
regression, logistic regression, decision trees, random forests, and support
vector machines.

Unsupervised learning works with unlabeled data. Clustering and dimensionality
reduction are common examples.

Machine learning requires data preprocessing, feature engineering, model
training, evaluation, and deployment."""
    ),

    "document3.pdf": (
        "Deep Learning",
        """Deep Learning is a subset of machine learning that uses artificial
neural networks with multiple layers.

Deep learning models can automatically learn complex representations from
large amounts of data.

Convolutional Neural Networks (CNNs) are commonly used for image processing
and computer vision.

Recurrent Neural Networks (RNNs) and transformer architectures can be used
for sequential and language-related tasks.

Deep learning is widely used in image classification, object detection,
speech recognition, natural language processing, and generative AI."""
    ),

    "document4.pdf": (
        "Natural Language Processing",
        """Natural Language Processing (NLP) is a field of artificial intelligence
that enables computers to understand, process, and generate human language.

Common NLP tasks include text classification, sentiment analysis, named entity
recognition, machine translation, question answering, and summarization.

Tokenization divides text into smaller units called tokens.

Embeddings represent words, sentences, or documents as numerical vectors.
These vectors allow systems to compare semantic similarity between pieces
of text.

Modern NLP systems frequently use transformer-based models."""
    ),

    "document5.pdf": (
        "Generative AI",
        """Generative AI refers to artificial intelligence systems that can create
new content such as text, images, audio, video, and code.

Large Language Models (LLMs) are generative AI models trained on large
collections of text.

Retrieval-Augmented Generation (RAG) combines information retrieval with
language generation. A retrieval system first finds relevant documents and
then provides that information to a language model.

Vector databases and embedding models are commonly used in semantic search
and RAG systems.

Generative AI is used for chatbots, coding assistants, content generation,
document analysis, and question answering."""
    ),
}


def create_pdf(filename, title, content):
    path = DATA_DIR / filename

    pdf = canvas.Canvas(str(path))
    pdf.setTitle(title)

    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(50, 800, title)

    pdf.setFont("Helvetica", 11)

    y = 770

    for paragraph in content.split("\n"):
        words = paragraph.split()
        line = ""

        for word in words:
            test_line = line + " " + word

            if pdf.stringWidth(test_line, "Helvetica", 11) > 500:
                pdf.drawString(50, y, line.strip())
                y -= 18
                line = word
            else:
                line = test_line

            if y < 60:
                pdf.showPage()
                pdf.setFont("Helvetica", 11)
                y = 800

        if line:
            pdf.drawString(50, y, line.strip())
            y -= 25

    pdf.save()


for filename, (title, content) in documents.items():
    create_pdf(filename, title, content)

print("5 PDF documents created successfully.")

for file in DATA_DIR.glob("*.pdf"):
    print(f"{file.name}: {file.stat().st_size} bytes")