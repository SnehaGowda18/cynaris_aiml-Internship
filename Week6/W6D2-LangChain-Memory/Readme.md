W6D2: LangChain Memory & Conversation History
📌 Overview

This project demonstrates LangChain chains, conversation history, and tool-based agents using a local Ollama LLM.

The implementation covers the three practical tasks for Week 6 Day 2:

Build a LangChain chain using PromptTemplate, Ollama LLM, and OutputParser.
Maintain conversation history across multiple turns.
Build a simple agent with a calculator tool and a web search stub.
🎯 Learning Objectives
Understand LangChain chains.
Use PromptTemplate with an LLM.
Process LLM responses using OutputParser.
Maintain conversation history.
Understand the basic concept of agents and tools.
Implement a calculator tool.
Implement a web search stub.
Generate output evidence for the completed tasks.
🛠️ Technologies Used
Python 3.11
LangChain
LangChain Core
LangChain Ollama
Ollama
Llama 3.2 3B
Git & GitHub
📁 Project Structure
W6D2-LangChain-Memory/
│
├── memory_demo.py
├── README.md
├── requirements.txt
│
├── outputs/
│   ├── chain_output.txt
│   ├── memory_output.txt
│   └── agent_output.txt
│
└── screenshots/
⚙️ Installation
1. Create a virtual environment
python -m venv .venv
2. Activate the virtual environment

Windows PowerShell:

.venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
4. Check Ollama
ollama list

The project uses:

llama3.2:3b

If the model is not available:

ollama pull llama3.2:3b
▶️ How to Run

Navigate to the project folder:

cd "C:\Users\USER\Desktop\Cynaris Intenship\Week6\W6D2-LangChain-Memory"

Run the application:

python memory_demo.py

The program displays the results in the terminal and automatically saves the results into the outputs folder.

🧪 Task 1: LangChain Chain

A LangChain chain is created using the following flow:

PromptTemplate
      ↓
Ollama LLM
      ↓
StrOutputParser
      ↓
Final Response

The chain is tested with 5 inputs:

What is artificial intelligence?
What is machine learning?
What is LangChain?
What is an LLM?
What is cloud computing?
Output

The results are saved in:

outputs/chain_output.txt
🧠 Task 2: Conversation History

Conversation history is implemented to maintain information from previous user interactions.

The program is tested with 5 conversation turns.

Example:

User: My name is Sneha.

User: I am studying Information Science.

User: What is my name?

Assistant: Your name is Sneha.

User: What am I studying?

Assistant: You are studying Information Science.
Output

The conversation results are saved in:

outputs/memory_output.txt
🤖 Task 3: Agent with Tools

A simple tool-based agent is implemented with two tools.

Tool 1: Calculator

The calculator performs mathematical calculations.

Example:

Input: 25 * 4
Result: 100
Input: 100 / 5 + 10
Result: 30.0
Tool 2: Web Search Stub

A simulated web search tool is implemented for demonstration purposes.

Example:

Input: What is LangChain?
Result: Simulated web search result
Output

The agent results are saved in:

outputs/agent_output.txt
📊 Output Evidence

The project generates three output files:

Output File	Description
chain_output.txt	Results from the LangChain chain
memory_output.txt	Conversation history results
agent_output.txt	Calculator and web search tool results