from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver


# ==========================================
# 1. Define Agent State
# ==========================================

class AgentState(TypedDict):
    user_input: str
    classification: str
    response: str
    human_feedback: str


# ==========================================
# 2. Classify Node
# ==========================================

def classify(state: AgentState):
    text = state["user_input"].lower()

    if any(word in text for word in [
        "build",
        "create",
        "write",
        "develop"
    ]):
        classification = "task"

    elif any(word in text for word in [
        "what",
        "why",
        "how",
        "explain"
    ]):
        classification = "question"

    else:
        classification = "human_review"

    print(f"Classification: {classification}")

    return {
        "classification": classification
    }


# ==========================================
# 3. Route Node
# ==========================================

def route(state: AgentState):
    classification = state["classification"]

    print(f"Routing to: {classification}")

    if classification == "human_review":
        human_feedback = interrupt(
            "Human review required. "
            "Enter approval or instructions:"
        )

        return {
            "human_feedback": str(human_feedback)
        }

    return {}


# ==========================================
# 4. Respond Node
# ==========================================

def respond(state: AgentState):
    classification = state["classification"]

    if classification == "task":
        response = (
            "This is a task. "
            "I will help you complete it."
        )

    elif classification == "question":
        response = (
            "This is a question. "
            "I will explain it clearly."
        )

    else:
        response = (
            "Human review completed. "
            f"Instructions received: {state['human_feedback']}"
        )

    return {
        "response": response
    }


# ==========================================
# 5. Build LangGraph State Machine
# ==========================================

builder = StateGraph(AgentState)

builder.add_node("classify", classify)
builder.add_node("route", route)
builder.add_node("respond", respond)

builder.add_edge(START, "classify")

# Conditional edge after classification
builder.add_conditional_edges(
    "classify",
    lambda state: state["classification"],
    {
        "task": "route",
        "question": "route",
        "human_review": "route"
    }
)

# Route all classifications to the response node.
# Human review pauses inside the route node.
builder.add_edge("route", "respond")

builder.add_edge("respond", END)


# ==========================================
# 6. Compile with Checkpointer
# ==========================================

checkpointer = MemorySaver()

graph = builder.compile(
    checkpointer=checkpointer
)


# ==========================================
# 7. Test Five Inputs
# ==========================================

test_inputs = [
    "Explain what DNS is",
    "How does a router work?",
    "Build a Python application",
    "Create a login page",
    "Tell me something interesting"
]


# ==========================================
# 8. Run Tests and Save Output
# ==========================================

with open("output.txt", "w", encoding="utf-8") as file:

    for index, text in enumerate(test_inputs, start=1):

        print("\n" + "=" * 50)
        print(f"Test {index}")
        print("Input:", text)

        file.write("\n" + "=" * 50 + "\n")
        file.write(f"Test {index}\n")
        file.write(f"Input: {text}\n")

        config = {
            "configurable": {
                "thread_id": f"w10d2-test-{index}"
            }
        }

        result = graph.invoke(
            {
                "user_input": text,
                "classification": "",
                "response": "",
                "human_feedback": ""
            },
            config=config
        )

        # Check for human-in-the-loop interruption
        if "__interrupt__" in result:

            print("Human review required.")

            file.write("Status: Paused for human review\n")

            approval = input(
                "Enter human approval or instructions: "
            )

            file.write(f"Human input: {approval}\n")

            # Resume the interrupted graph
            result = graph.invoke(
                Command(resume=approval),
                config=config
            )

        print("Classification:", result["classification"])
        print("Response:", result["response"])

        file.write(
            f"Classification: {result['classification']}\n"
        )

        file.write(
            f"Response: {result['response']}\n"
        )

        file.write("Status: Completed\n")


print("\nAll five tests completed successfully.")
print("Output saved automatically to output.txt")