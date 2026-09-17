# W9D1 - CrewAI Multi-Agent Research Crew

## Description

Built a multi-agent research system using CrewAI with three roles: Researcher, Writer, and Reviewer. The project also compares research performed without web search and research enhanced with real-time web search.

## Agents

* **Researcher** – Collects and organizes information about the topic.
* **Writer** – Converts the research into a structured article.
* **Reviewer** – Checks the article for accuracy, clarity, grammar, and completeness.

## Web Search Comparison

The project performs two research runs:

1. **Without Web** – Uses the local Ollama LLM's existing knowledge.
2. **With Web** – Uses CrewAI's web search tool to retrieve current information.

The results are saved in:

* `research_without_web.txt`
* `research_with_web.txt`

## Tools Used

* Python
* CrewAI
* CrewAI Tools
* Ollama
* Llama 3.2 3B
* Serper Web Search
* PowerShell
* Git & GitHub

## Process

```text
Researcher
    ↓
Writer
    ↓
Reviewer
    ↓
Final Research Output
```

## Learning Outcomes

* Understanding CrewAI Agents
* Understanding CrewAI Tasks
* Creating and running Crews
* Sequential multi-agent processing
* Agent-to-agent task context
* Integrating web search
* Comparing research with and without web data
* Saving and documenting research results
