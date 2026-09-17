# W7D1 - BM25 Manual Evaluation

questions = [
    "What is artificial intelligence?",
    "What are the types of machine learning?",
    "What is supervised learning?",
    "What are convolutional neural networks used for?",
    "What is deep learning?",
    "What is natural language processing?",
    "What are embeddings in NLP?",
    "What is generative AI?",
    "What is retrieval augmented generation?",
    "What are large language models?",
]

expected_documents = [
    "document1.pdf",
    "document2.pdf",
    "document2.pdf",
    "document3.pdf",
    "document3.pdf",
    "document4.pdf",
    "document4.pdf",
    "document5.pdf",
    "document5.pdf",
    "document5.pdf",
]

print("=" * 70)
print("BM25 MANUAL RETRIEVAL EVALUATION")
print("=" * 70)

for i in range(10):
    print(f"\nQ{i + 1}: {questions[i]}")
    print(f"Expected document: {expected_documents[i]}")
    print("Retrieved correctly: YES / NO")

print("\n")
print("Enter the number of correctly retrieved questions.")