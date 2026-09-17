# W9D4: CrewAI + LangChain — Hybrid Agent Systems

## Objective

Build a LangChain chain with Ollama, maintain conversation history, and create a simple two-tool agent.

## Technologies Used

- Python
- LangChain
- Ollama
- LangChain Core
- LangChain Ollama
- CrewAI concepts
- AI/ML 3M stack

## Tasks Completed

### 1. LangChain Chain

Implemented the following chain:

PromptTemplate → Ollama LLM → StrOutputParser

The chain was tested with five inputs:

1. What is artificial intelligence?
2. What is machine learning?
3. What is LangChain?
4. What is an LLM?
5. What is RAG?

### 2. Conversation Memory

Implemented conversation history using a history buffer.

Five conversation turns were tested.

The system successfully maintained previous conversation information and used it to answer later questions.

### 3. Two-Tool Agent

Implemented two tools:

#### Web Search

A simulated web search stub that returns search results.

#### Calculator

A calculator tool for evaluating basic mathematical expressions.

Three tasks were tested:

- Asking about LangChain
- Calculating `25 * 4`
- Searching for information about RAG

## Project Structure

```text
W9D4/
├── chain_memory.py
├── agent_tools.py
├── requirements.txt
├── README.md
└── outputs/
    └── evidence.txt