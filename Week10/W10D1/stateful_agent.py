from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command


# ============================================================
# 1. STATE DEFINITION
# ============================================================

class AgentState(TypedDict):
    user_input: str
    classification: str
    human_approval: str
    response: str


# ============================================================
# 2. CLASSIFY NODE
# ============================================================

def classify(state: AgentState):
    """
    Classify the user input into:
    - question
    - task
    - other
    """

    text = state["user_input"].lower()

    if any(word in text for word in [
        "create",
        "build",
        "write",
        "make",
        "calculate"
    ]):
        classification = "task"

    elif any(word in text for word in [
        "hello",
        "hi",
        "hey",
        "how are you"
    ]):
        classification = "other"

    else:
        classification = "question"

    print(f"[CLASSIFY] {classification}")

    return {
        "classification": classification
    }


# ============================================================
# 3. ROUTE NODE + HUMAN INTERRUPT
# ============================================================

def route(state: AgentState):
    """
    Route the request and pause for human approval.
    """

    classification = state["classification"]

    print(f"[ROUTE] Classification: {classification}")

    # Pause the graph and wait for human input
    approval = interrupt({
        "message": "Human approval required before responding.",
        "classification": classification,
        "user_input": state["user_input"]
    })

    print(f"[HUMAN INPUT] {approval}")

    return {
        "human_approval": str(approval)
    }


# ============================================================
# 4. RESPOND NODE
# ============================================================

def respond(state: AgentState):
    """
    Generate a response after human approval.
    """

    classification = state["classification"]
    user_input = state["user_input"]
    approval = state["human_approval"]

    approval = approval.lower().strip()

    if approval not in [
        "yes",
        "y",
        "approve",
        "approved"
    ]:
        response = "Human reviewer rejected the request."

    elif classification == "question":
        response = f"Answering your question: {user_input}"

    elif classification == "task":
        response = f"Processing your requested task: {user_input}"

    else:
        response = f"Hello! You said: {user_input}"

    print(f"[RESPOND] {response}")

    return {
        "response": response
    }


# ============================================================
# 5. CONDITIONAL ROUTING FUNCTION
# ============================================================

def route_decision(state: AgentState):
    """
    Decide which path should be followed based
    on the classification.
    """

    classification = state["classification"]

    if classification == "question":
        return "question"

    elif classification == "task":
        return "task"

    else:
        return "other"


# ============================================================
# 6. BUILD LANGGRAPH
# ============================================================

graph_builder = StateGraph(AgentState)

# Add the three nodes
graph_builder.add_node("classify", classify)
graph_builder.add_node("route", route)
graph_builder.add_node("respond", respond)

# START → CLASSIFY
graph_builder.add_edge(
    START,
    "classify"
)

# CLASSIFY → ROUTE
graph_builder.add_edge(
    "classify",
    "route"
)

# ROUTE → conditional paths
graph_builder.add_conditional_edges(
    "route",
    route_decision,
    {
        "question": "respond",
        "task": "respond",
        "other": "respond",
    }
)

# RESPOND → END
graph_builder.add_edge(
    "respond",
    END
)


# ============================================================
# 7. CHECKPOINT / MEMORY
# ============================================================

memory = MemorySaver()

graph = graph_builder.compile(
    checkpointer=memory
)


# ============================================================
# 8. TEST FUNCTION
# ============================================================

def run_test(test_number, user_input, human_input):

    print("\n" + "-" * 60)
    print(f"Test {test_number}")
    print(f"User: {user_input}")

    # Unique thread ID is required for checkpointing
    config = {
        "configurable": {
            "thread_id": f"w10d1-test-{test_number}"
        }
    }

    # --------------------------------------------------------
    # First execution
    # Graph will pause at interrupt()
    # --------------------------------------------------------

    result = graph.invoke(
        {
            "user_input": user_input,
            "classification": "",
            "human_approval": "",
            "response": "",
        },
        config=config
    )

    print("Graph paused for human approval.")

    # --------------------------------------------------------
    # Simulated human input
    # --------------------------------------------------------

    print(f"Human approval: {human_input}")

    # --------------------------------------------------------
    # Resume graph
    # --------------------------------------------------------

    result = graph.invoke(
        Command(
            resume=human_input
        ),
        config=config
    )

    print("Graph resumed.")

    print(
        f"Final Response: {result['response']}"
    )


# ============================================================
# 9. MAIN TESTS
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("LANGGRAPH STATEFUL AGENT")
    print("3-NODE GRAPH + CONDITIONAL ROUTING")
    print("HUMAN-IN-THE-LOOP INTERRUPT")
    print("=" * 60)

    # --------------------------------------------------------
    # Test 1
    # Question + approval
    # --------------------------------------------------------

    run_test(
        1,
        "What is artificial intelligence?",
        "yes"
    )

    # --------------------------------------------------------
    # Test 2
    # Question + approval
    # --------------------------------------------------------

    run_test(
        2,
        "Explain how Python functions work.",
        "yes"
    )

    # --------------------------------------------------------
    # Test 3
    # Task + approval
    # --------------------------------------------------------

    run_test(
        3,
        "Create a simple Python calculator.",
        "yes"
    )

    # --------------------------------------------------------
    # Test 4
    # Question + approval
    # --------------------------------------------------------

    run_test(
        4,
        "Tell me today's weather.",
        "yes"
    )

    # --------------------------------------------------------
    # Test 5
    # Other + rejection
    # --------------------------------------------------------

    run_test(
        5,
        "Hello, how are you?",
        "no"
    )

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)