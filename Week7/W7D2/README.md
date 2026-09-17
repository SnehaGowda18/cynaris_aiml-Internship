# W7D2: Haystack Retrieval — BM25 & Dense Retrieval

## Objective

Implemented and compared BM25 and Dense Retrieval using Haystack on 5 PDF documents and 10 test questions.

## Technologies Used

- Python
- Haystack
- Sentence Transformers
- PyPDF
- InMemoryDocumentStore
- BM25 Retriever
- Dense Embedding Retriever
- Git & GitHub

## Implementation

### BM25 Retrieval

PDF documents are extracted and stored in a Haystack `InMemoryDocumentStore`. The `InMemoryBM25Retriever` retrieves relevant documents using keyword-based matching.

### Dense Retrieval

The PDF contents and user questions are converted into vector embeddings using Sentence Transformers. Haystack's `InMemoryEmbeddingRetriever` performs semantic similarity-based retrieval.

## Dataset

Five PDF documents were used:

1. document1.pdf — Artificial Intelligence
2. document2.pdf — Machine Learning
3. document3.pdf — Deep Learning and CNNs
4. document4.pdf — NLP and Embeddings
5. document5.pdf — Generative AI, RAG and LLMs

## Evaluation

Both retrieval methods were evaluated using the same 10 questions.

The expected source document for every question was manually defined and the retrieved results were compared against it.

## Results

The final precision values are recorded in:

- `outputs/bm25_results.txt`
- `outputs/dense_results.txt`
- `outputs/comparison_results.txt`

## Self-Review Checklist

- [ ] 5 PDF documents indexed
- [ ] BM25 retrieval implemented
- [ ] Dense retrieval implemented
- [ ] 10 questions tested
- [ ] BM25 precision evaluated
- [ ] Dense precision evaluated
- [ ] BM25 and Dense results compared
- [ ] Output evidence generated
- [ ] Code reviewed using CIA
- [ ] Documentation completed
- [ ] Minimum 2 Git commits completed
- [ ] Pull Request created

## Conclusion

The project demonstrates the difference between lexical BM25 retrieval and semantic dense retrieval. BM25 is effective for exact keyword matching, while dense retrieval can identify semantically related content even when the wording differs.