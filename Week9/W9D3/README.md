# W9D3: Building a Research Crew — Multi-Agent Pipeline

## Overview

This project implements a multi-agent research pipeline using CrewAI and a local Ollama LLM.

Three specialized agents work sequentially:

**Researcher → Writer → Reviewer**

The Researcher uses web search to collect current information, the Writer converts the findings into a structured technical report, and the Reviewer evaluates the quality of the final report.

## Learning Objective

* Define and configure multiple CrewAI agents.
* Assign roles, goals, and backstories.
* Create sequential CrewAI tasks.
* Integrate web search into an agent.
* Use Ollama as the local LLM.
* Generate and review research output.
* Compare research with and without web search.
* Follow Git and documentation best practices.

## Research Topic

**Impact of Generative AI on Software Development**

## Multi-Agent Architecture

```text
                 ┌─────────────────────┐
                 │    Web Search Tool  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │     Researcher      │
                 │  Collects current   │
                 │    web research     │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │       Writer        │
                 │ Creates technical   │
                 │       report        │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │      Reviewer       │
                 │ Checks quality and  │
                 │ suggests changes    │
                 └──────────┬──────────┘
                            │
                            ▼
                    Final Research
                       Output
```

## Agents

### 1. Researcher

**Role:** Technology Researcher

**Goal:** Research Generative AI and its impact on software development using current web information.

**Responsibilities:**

* Search for relevant information.
* Identify applications and use cases.
* Collect benefits and challenges.
* Identify security and privacy concerns.
* Find real-world examples.
* Identify future trends.
* Provide references where possible.

### 2. Writer

**Role:** Technical Writer

**Goal:** Convert the research findings into a clear and structured technical report.

**Responsibilities:**

* Organize the research.
* Explain technical concepts clearly.
* Create structured sections.
* Include benefits and limitations.
* Include references.
* Avoid unsupported claims.

### 3. Reviewer

**Role:** Research Quality Reviewer

**Goal:** Evaluate the report for accuracy, completeness, clarity, and overall quality.

**Responsibilities:**

* Check factual accuracy.
* Identify unsupported claims.
* Check organization and readability.
* Identify missing information.
* Check balance between benefits and risks.
* Suggest improvements.
* Provide a final quality verdict.

## Technology Stack

* Python
* CrewAI
* CrewAI Tools
* Ollama
* Llama 3.2 3B
* Serper Web Search
* python-dotenv
* Git / GitHub

## Project Structure

```text
W9D3/
│
├── .env
├── .gitignore
├── research_crew.py
├── research_output.txt
├── web_research_output.txt
├── comparison.txt
├── self_review.md
├── requirements.txt
└── README.md
```

## Setup

### 1. Create and activate virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Install and run Ollama

Check the installed model:

```powershell
ollama list
```

The project uses:

```text
llama3.2:3b
```

The CrewAI LLM configuration connects to:

```text
http://localhost:11434
```

### 4. Configure Serper API

Create a `.env` file:

```env
SERPER_API_KEY=your_actual_api_key
```

The API key is used by `SerperDevTool` for web search.

**Do not commit `.env` to GitHub.**

## Running the Project

Run:

```powershell
python research_crew.py
```

The CrewAI pipeline executes in sequential order:

```text
Researcher
    ↓
Writer
    ↓
Reviewer
```

## Output Files

### `research_output.txt`

Contains the generated research crew output.

### `web_research_output.txt`

Contains the Researcher's web-based research output.

### `comparison.txt`

Documents the improvement achieved by adding web search.

## Web Search Improvement

### Before Web Search

The initial research crew used the local Ollama model without an external web-search tool.

Limitations included:

* Information was based primarily on the model's existing knowledge.
* Current web information was not actively retrieved.
* External references were limited.

### After Web Search

The Researcher was provided with `SerperDevTool`.

Benefits include:

* Access to current web information.
* Better research grounding.
* More relevant examples.
* References from external sources.
* Improved research credibility.

## Error Handling

The application checks whether the `SERPER_API_KEY` is available before starting the crew.

It also catches execution errors and provides troubleshooting information for:

* Ollama availability.
* Missing LLM model.
* Missing Serper API key.
* Missing dependencies.
* Internet connectivity.

## Security

The Serper API key is stored in `.env`.

The `.env` file should be excluded from Git:

```text
.env
.venv/
__pycache__/
```

Never commit API keys or other secrets to the repository.

## Git Workflow

Example workflow:

```powershell
git status
git add .
git commit -m "feat: crewai — multi-agent research crew"
git push
```

If the branch does not have an upstream:

```powershell
git push --set-upstream origin feat/aiml-W9-sneha
```

## Self-Review Checklist

* [x] Three CrewAI agents created.
* [x] Roles, goals, and backstories defined.
* [x] Sequential task pipeline implemented.
* [x] Ollama local LLM integrated.
* [x] Web search integrated.
* [x] Research output generated.
* [x] Writer produces structured report.
* [x] Reviewer evaluates report quality.
* [x] Output files documented.
* [x] API key protected using `.env`.
* [ ] Final Git commit completed.
* [ ] Pull request created.
* [ ] CIA interactions documented.

## Viva Questions and Answers

### 1. Explain your build and design decisions.

I built a sequential multi-agent research pipeline using CrewAI. I used three agents: Researcher, Writer, and Reviewer. The Researcher collects information using web search, the Writer converts it into a structured report, and the Reviewer checks the quality. I used Ollama with Llama 3.2 3B as the local LLM to avoid depending on an OpenAI API key.

### 2. What was the hardest part and how did you solve it?

The main challenge was configuring the LLM and web-search integration. Initially, the project required an OpenAI API key. I changed the implementation to use Ollama locally. I then installed `crewai-tools` and configured `SerperDevTool` with a Serper API key stored securely in `.env`.

### 3. What would you improve with one more day?

I would improve the system by adding stronger source validation, automated evaluation metrics, better citation handling, and MLflow tracing. I would also add automated tests and improve the report formatting.

## Conclusion

This project demonstrates a practical multi-agent research workflow using CrewAI. Separating research, writing, and review into specialized agents makes the pipeline easier to understand, maintain, and improve. Adding web search provides access to current external information and improves the grounding of the research process.
