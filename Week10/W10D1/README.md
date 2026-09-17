# W10D1: LangGraph — Stateful Agent Graphs

## Overview

Built a stateful agent graph using LangGraph with three main nodes:

```text
classify → route → respond
```

The graph classifies user input, conditionally routes it, and generates a response.

## Features

* Three-node LangGraph architecture
* Stateful graph using `TypedDict`
* Classification of user requests
* Conditional routing
* Human-in-the-loop interrupt
* Checkpointing with `MemorySaver`
* Pause and resume using `Command(resume=...)`
* Tested with multiple user inputs

## Graph Flow

```text
START
  ↓
classify
  ↓
route
  ↓
conditional routing
  ↓
respond
  ↓
END
```

## Classification

The agent classifies inputs into:

* `question`
* `task`
* `other`

The classification determines the routing decision.

## Human-in-the-Loop

Before the response is generated, the graph pauses using LangGraph's `interrupt()`.

The human reviewer can enter:

```text
yes
```

to approve the request or:

```text
no
```

to reject it.

The graph resumes using:

```python
Command(resume=human_input)
```

## Technologies

* Python
* LangGraph
* LangChain Core

## Testing

The initial graph was tested with five inputs:

1. What is artificial intelligence?
2. Explain how Python functions work.
3. Create a simple Python calculator.
4. Tell me today's weather.
5. Hello, how are you?

Human-in-the-loop testing was performed using both approval and rejection inputs.

## Deliverables

* `stateful_agent.py` — LangGraph implementation
* `requirements.txt` — dependencies
* `output.txt` — execution evidence
* `README.md` — project documentation

