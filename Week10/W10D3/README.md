# W10D3: LangGraph + Memory — Persistent Conversations

## Objective

Build a three-node LangGraph workflow with conditional routing,
persistent conversation memory, and human-in-the-loop interruption.

## Features

- Classify user inputs as questions, tasks, or other.
- Route inputs using conditional edges.
- Generate responses using a local Ollama LLM.
- Maintain conversation history using LangGraph MemorySaver.
- Pause task execution for human approval.
- Resume execution after human input.

## Workflow

START → CLASSIFY → ROUTE → RESPOND → END

## Technologies

- Python
- LangGraph
- LangChain
- Ollama
- Llama 3.2
- MemorySaver
- MLOps

## Testing

Five inputs were tested:

1. User name memory.
2. Remembering the user's name.
3. Artificial intelligence question.
4. LangGraph memory explanation.
5. Python study plan task.

## Expected Evidence

- Classification output.
- Correct routing.
- Conversation memory.
- Human approval interruption.
- Successful resume.
- Five test results.

## Self-Review Checklist

- [x] Three-node graph created.
- [x] Conditional routing implemented.
- [x] Persistent conversation memory added.
- [x] Human-in-the-loop interruption tested.
- [x] Five inputs tested.
- [x] README created.
- [ ] CIA Full Stack Mentor Mode review completed.
- [ ] Minimum two Git commits completed.
- [ ] Pull request raised.