# W10D4: Human-in-the-Loop with LangGraph

## Description

Built and tested a three-node LangGraph workflow using classify, route, and respond nodes. The workflow uses conditional routing to classify questions and tasks, and a human-in-the-loop interrupt to pause and resume execution after human approval or rejection.

## Tools Used

* Python
* LangGraph
* LangChain Core
* MemorySaver
* PowerShell
* Git and GitHub

## Features

* Three-node stateful graph.
* Conditional routing based on classification.
* Human-in-the-loop interrupt.
* Pause and resume using Command.
* Five test inputs.
* Approval and rejection handling.

## Test Results

All five test inputs completed successfully.

* Three questions classified as question.
* Two tasks classified as task.
* Four inputs approved.
* One input rejected.

## Learning Outcome

Learned how to build a stateful LangGraph workflow, implement conditional edges, and use human-in-the-loop interrupts to control execution.
