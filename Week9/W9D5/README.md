Project Overview

The Automated Research Report Agent is an AI-powered research workflow that automatically generates structured research reports on a given topic.

The project combines CrewAI for multi-agent collaboration, LangGraph for workflow orchestration, Ollama for local LLM inference, Ragas for evaluation, and MLflow for experiment tracking.

🎯 Objectives
Automate the research and report-generation process.
Use multiple specialized AI agents for research, analysis, and writing.
Orchestrate the workflow using LangGraph.
Run the LLM locally using Ollama.
Evaluate generated reports using a Ragas evaluation dataset and local Ollama scoring.
Track report metrics and evaluation scores using MLflow.
Add automated tests using pytest.
🏗️ Architecture
                    User
                      |
                      v
             Research Topic
                      |
                      v
             +----------------+
             |   LangGraph    |
             |   Workflow     |
             +----------------+
                      |
                      v
             +----------------+
             |    CrewAI      |
             +----------------+
                      |
          +-----------+-----------+
          |           |           |
          v           v           v
     Researcher   Analyst     Report Writer
          |           |           |
          +-----------+-----------+
                      |
                      v
             Research Report
                      |
          +-----------+-----------+
          |                       |
          v                       v
       Ragas                  MLflow
     Evaluation              Tracking
🤖 CrewAI Agents
1. Research Specialist

Collects important information about the research topic, including:

Introduction
Important concepts
Key developments
Benefits
Limitations
Applications
Challenges
Future possibilities
2. Research Analyst

Reviews the research findings and identifies:

Major insights
Advantages
Disadvantages
Practical applications
Challenges
Future opportunities
Conclusions
3. Report Writer

Transforms the research and analysis into a structured professional report.

🔄 LangGraph Workflow

The workflow follows a sequential graph:

START
  |
  v
Research
  |
  v
Analysis
  |
  v
Report
  |
  v
END

Each node receives and updates a shared ResearchState.

🧰 Technologies Used
Technology	Purpose
Python	Application development
CrewAI	Multi-agent orchestration
LangGraph	Workflow orchestration
Ollama	Local LLM execution
Llama 3.2 3B	Local language model
Ragas	Evaluation dataset and evaluation workflow
MLflow	Experiment and metric tracking
pytest	Automated testing
Git/GitHub	Version control