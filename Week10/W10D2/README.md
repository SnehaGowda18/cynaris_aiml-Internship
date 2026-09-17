# W10D2: LangGraph State Machines & Conditional Edges

## Description

Built and tested a LangGraph state machine with three nodes: classify, route, and respond. The project uses conditional edges to route user inputs based on their classification and implements human-in-the-loop interruption with resume functionality.

## Objectives

- Build a LangGraph state machine with three nodes.
- Implement conditional edges based on classification.
- Test the graph with five different inputs.
- Add human-in-the-loop interruption and resume.
- Save execution results automatically to output.txt.

## Technologies Used

- Python 3.11
- LangGraph
- LangChain Core
- MemorySaver
- VS Code

## Project Structure

W10D2/
├── w10d2_state_machine.py
├── requirements.txt
├── README.md
└── output.txt

## Workflow

User Input
    ↓
Classify
    ↓
Route
    ↓
Conditional Edge
    ├── Task
    ├── Question
    └── Human Review
              ↓
        Human Approval
              ↓
           Respond

## Implementation

### Classify Node

Classifies user inputs into task, question, or human_review.

### Route Node

Uses conditional routing and pauses execution when human review is required.

### Respond Node

Generates a response based on the classification and human feedback.

### Human-in-the-Loop

The graph uses LangGraph interrupt() to pause execution and Command(resume=...) to continue after human approval.

## Test Inputs

1. Explain what DNS is
2. How does a router work?
3. Build a Python application
4. Create a login page
5. Tell me something interesting

## Results

- Five inputs tested successfully.
- Conditional routing verified.
- Human-in-the-loop interruption tested.
- Human approval and resume completed.
- Results saved automatically to output.txt.

## How to Run

Install dependencies:

pip install -r requirements.txt

Run the application:

python w10d2_state_machine.py

Enter human approval when prompted for the fifth input.

## Output Evidence

The file output.txt contains the test inputs, classifications, responses, human-review evidence, and completion status.

## Git Workflow

Branch:

feat/aiml-W10D2-sneha

Commit:

feat: langgraph state machine with conditional routing

## Self-Review Checklist

- [x] Three LangGraph nodes implemented.
- [x] Conditional edges added.
- [x] Five inputs tested.
- [x] Human-in-the-loop interruption implemented.
- [x] Human approval and resume tested.
- [x] Output saved automatically.
- [x] README documentation completed.
- [ ] Code review completed.
- [ ] Pull request raised.

## Future Improvements

- Add an LLM-based classifier.
- Improve response generation.
- Add persistent database checkpointing.
- Add automated unit tests.