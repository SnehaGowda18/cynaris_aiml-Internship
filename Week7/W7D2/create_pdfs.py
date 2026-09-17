from pathlib import Path
from reportlab.pdfgen import canvas

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

documents = {
    "document1.pdf": """
Artificial Intelligence

Artificial Intelligence (AI) is the field of computer science concerned
with creating systems that can perform tasks that normally require human
intelligence. These tasks include reasoning, learning, problem solving,
decision making, speech recognition, and visual perception.
""",

    "document2.pdf": """
Machine Learning

Machine Learning (ML) is a subset of artificial intelligence that enables
computers to learn patterns from data and make predictions or decisions.

The main types of machine learning are supervised learning, unsupervised
learning, and reinforcement learning.

Supervised learning uses labeled training data. Examples include
classification and regression.
""",

    "document3.pdf": """
Deep Learning and Convolutional Neural Networks

Deep learning is a branch of machine learning that uses multi-layer
artificial neural networks to learn complex patterns from large datasets.

Convolutional Neural Networks (CNNs) are deep learning models commonly
used for image classification, object detection, image recognition,
and computer vision applications.
""",

    "document4.pdf": """
Natural Language Processing and Embeddings

Natural Language Processing (NLP) is a field of artificial intelligence
that enables computers to understand, process, and generate human language.

Embeddings are numerical vector representations of words, sentences,
or documents. They capture semantic relationships between pieces of text
and are widely used in semantic search and NLP applications.
""",

    "document5.pdf": """
Generative AI, RAG and Large Language Models

Generative AI refers to artificial intelligence systems that can create
new content such as text, images, audio, and code.

Retrieval Augmented Generation (RAG) combines information retrieval with
generative AI. Relevant information is retrieved from a knowledge base
and provided to a language model to generate a grounded response.

Large Language Models (LLMs) are neural network models trained on large
amounts of text data. They can understand and generate human language.
"""
}


def create_pdf(filename, text):
    path = DATA_DIR / filename
    c = canvas.Canvas(str(path))
    y = 800

    for line in text.strip().splitlines():
        if y < 50:
            c.showPage()
            y = 800

        c.drawString(50, y, line.strip())
        y -= 18

    c.save()
    print(f"Created: {path}")


for filename, text in documents.items():
    create_pdf(filename, text)

print(f"\nCreated {len(documents)} PDF documents.")