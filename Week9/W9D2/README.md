# W9D2 — CrewAI Tools: Web Search & Code Execution

## 📌 Overview

This project demonstrates a multi-agent research workflow using **CrewAI**. Three specialized AI agents work together to research, write, and review a technical article on **Retrieval Augmented Generation (RAG)**.

The project also integrates a custom **Web Search Tool** to provide current information from the web.

## 🎯 Objective

* Create multiple CrewAI agents with different roles.
* Use a Web Search tool for current information.
* Use a local Ollama LLM instead of OpenAI.
* Build a sequential multi-agent workflow.
* Compare research before and after adding web search.
* Generate and save the final research output.

## 🏗️ Architecture

```text
                    ┌─────────────────┐
                    │   Researcher    │
                    │   AI Agent      │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Web Search    │
                    │      Tool       │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │     Writer      │
                    │    AI Agent     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    Reviewer     │
                    │    AI Agent     │
                    └────────┬────────┘
                             │
                             ▼
                  research_answer.txt
```

## 🤖 Agents

### 1. Researcher

**Role:** AI Researcher

Responsible for researching RAG and collecting relevant information using the Web Search tool.

### 2. Writer

**Role:** Technical Writer

Converts the research findings into a structured and easy-to-understand technical article.

### 3. Reviewer

**Role:** Technical Reviewer

Reviews the generated article for technical correctness, clarity, completeness, and readability.

## 🔧 Tools & Technologies

* **CrewAI** — Multi-agent orchestration
* **Ollama** — Local LLM execution
* **Llama 3.2 3B** — Local language model
* **Python** — Programming language
* **Requests** — HTTP requests for web search
* **BeautifulSoup** — Web page parsing
* **CrewAI BaseTool** — Custom CrewAI web search tool

## 📂 Project Structure

```text
W9D2/
│
├── crew.py
├── tools.py
├── research_answer.txt
├── research_comparison.txt
└── README.md
```

## ⚙️ Setup

### 1. Install dependencies

```bash
pip install crewai litellm requests beautifulsoup4
```

### 2. Check Ollama

Make sure Ollama is installed and the model is available:

```bash
ollama list
```

Required model:

```text
llama3.2:3b
```

If the model is not available:

```bash
ollama pull llama3.2:3b
```

### 3. Start Ollama

If Ollama is not already running:

```bash
ollama serve
```

Ollama normally runs at:

```text
http://localhost:11434
```

## ▶️ Run the Project

Run:

```bash
python crew.py
```

The CrewAI workflow executes sequentially:

```text
Researcher
    ↓
Web Search
    ↓
Writer
    ↓
Reviewer
```

The final output is saved to:

```text
research_answer.txt
```

## 🌐 Web Search

A custom CrewAI `BaseTool` is implemented in `tools.py`.

The tool:

1. Accepts a search query.
2. Sends the query to a web search engine.
3. Parses the returned HTML using BeautifulSoup.
4. Extracts useful text.
5. Returns the results to the Researcher agent.

## 📊 Web Search Improvement

The project compares the research workflow before and after adding web search.

### Before Web Search

The model generates information using its existing knowledge.

### After Web Search

The Researcher retrieves external web information before the Writer generates the article.

### Improvements

* Access to more current information.
* Better research coverage.
* External information can be incorporated into the answer.
* Research, writing, and reviewing are separated between agents.
* The Reviewer improves the final response.

The comparison is documented in:

```text
research_comparison.txt
```

## 📄 Output

The final reviewed RAG article is stored in:

```text
research_answer.txt
```

The output covers:

* What is RAG?
* How RAG works
* Main components
* Benefits
* Limitations
* Applications
* RAG vs traditional LLM generation
* Conclusion

## 🧠 Key Learning

This project demonstrates how **CrewAI tools and multiple specialized agents** can be combined to create a research workflow.

Instead of asking a single agent to perform every task, responsibilities are divided:

```text
Research → Write → Review
```

Using a local Ollama model also allows the workflow to run without requiring an OpenAI API key.

## 🔐 Local LLM Configuration

The project uses:

```text
Ollama
    ↓
llama3.2:3b
    ↓
CrewAI Agents
```

The LLM is configured in `crew.py` using the Ollama endpoint:

```text
http://localhost:11434
```

## 🚀 Conclusion

W9D2 implements a CrewAI-based multi-agent research system with web search capabilities. The Researcher collects information, the Writer creates the article, and the Reviewer validates and improves the final output.

The project demonstrates practical use of **CrewAI agents, custom tools, web search, and local LLM execution with Ollama**.
