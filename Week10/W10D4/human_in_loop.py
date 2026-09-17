from typing import TypedDict, Literal
from pathlib import Path

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command


# ==================================================
# W10D4: HUMAN-IN-THE-LOOP WITH LANGGRAPH
# ==================================================

OUTPUT_FILE = Path(__file__).parent / "output.txt"


class AgentState(TypedDict):
    user_input: str
    classification: str
    route: str
    response: str


# ==================================================
# OUTPUT LOGGER
# ==================================================

OUTPUT_FILE.write_text("", encoding="utf-8")


def log(message=""):
    """Display output and save it automatically."""
    print(message, flush=True)

    with OUTPUT_FILE.open("a", encoding="utf-8") as file:
        file.write(str(message) + "\n")


# ==================================================
# NODE 1: CLASSIFY
# ==================================================

def classify(state: AgentState):
    text = state["user_input"].lower().strip()

    question_words = [
        "what",
        "why",
        "how",
        "explain",
        "define",
        "difference",
        "when",
        "where",
    ]

    classification = "question"

    if not any(
        word in text.split()
        for word in question_words
    ):
        classification = "task"

    log(f"Classification: {classification}")

    return {
        "classification": classification
    }


# ==================================================
# NODE 2: ROUTE
# ==================================================

def route(state: AgentState):
    if state["classification"] == "question":
        selected_route = "answer_question"
    else:
        selected_route = "perform_task"

    log(f"Route: {selected_route}")

    return {
        "route": selected_route
    }


# ==================================================
# CONDITIONAL ROUTING
# ==================================================

def choose_route(
    state: AgentState,
) -> Literal["answer_question", "perform_task"]:

    if state["classification"] == "question":
        return "answer_question"

    return "perform_task"


# ==================================================
# NODE 3: RESPOND
# ==================================================

def respond(state: AgentState):

    human_decision = interrupt(
        {
            "message": "Human review required.",
            "question": "Approve this response?",
            "input": state["user_input"],
            "classification": state["classification"],
            "route": state["route"],
        }
    )

    if human_decision == "approve":

        if state["classification"] == "question":
            response = (
                "Approved response: "
                "The input is a question. "
                "The system selected the "
                "question-answering route."
            )

        else:
            response = (
                "Approved response: "
                "The input is a task. "
                "The system selected the "
                "task-execution route."
            )

    else:
        response = "Response rejected by human reviewer."

    log(f"Response: {response}")

    return {
        "response": response
    }


# ==================================================
# BUILD GRAPH
# ==================================================

builder = StateGraph(AgentState)

builder.add_node("classify", classify)
builder.add_node("route", route)
builder.add_node("respond", respond)

builder.add_edge(START, "classify")

builder.add_edge("classify", "route")

builder.add_conditional_edges(
    "route",
    choose_route,
    {
        "answer_question": "respond",
        "perform_task": "respond",
    },
)

builder.add_edge("respond", END)

memory = MemorySaver()

graph = builder.compile(checkpointer=memory)


# ==================================================
# TEST FIVE INPUTS
# ==================================================

if __name__ == "__main__":

    log("=" * 50)
    log("W10D4: HUMAN-IN-THE-LOOP WITH LANGGRAPH")
    log("=" * 50)

    test_inputs = [
        "What is artificial intelligence?",
        "Explain LangGraph.",
        "Write a Python program.",
        "Create a study plan.",
        "What is machine learning?",
    ]

    results = []

    for test_number, user_input in enumerate(
        test_inputs,
        start=1
    ):

        log("")
        log("=" * 50)
        log(f"TEST {test_number}")
        log("=" * 50)

        log(f"Input: {user_input}")

        config = {
            "configurable": {
                "thread_id": f"w10d4-test-{test_number}"
            }
        }

        initial_state = {
            "user_input": user_input,
            "classification": "",
            "route": "",
            "response": "",
        }

        # First invocation pauses at interrupt
        result = graph.invoke(
            initial_state,
            config=config,
        )

        log("")
        log("Graph paused for human review.")

        log("Interrupt information:")
        log(result["__interrupt__"])

        # Real human approval or rejection
        while True:

            human_decision = input(
                "Enter approve or reject: "
            ).strip().lower()

            if human_decision in ["approve", "reject"]:
                break

            log("Please enter only approve or reject.")

        # Resume the interrupted graph
        final_result = graph.invoke(
            Command(resume=human_decision),
            config=config,
        )

        log("")
        log("Graph resumed successfully.")

        log("Final response:")
        log(final_result["response"])

        results.append(
            {
                "test": test_number,
                "input": user_input,
                "classification": final_result["classification"],
                "route": final_result["route"],
                "human_decision": human_decision,
                "response": final_result["response"],
            }
        )

    # ==================================================
    # FINAL TEST SUMMARY
    # ==================================================

    log("")
    log("=" * 50)
    log("FINAL TEST SUMMARY")
    log("=" * 50)

    for item in results:

        log(
            f"Test {item['test']}: "
            f"{item['classification']} -> "
            f"{item['route']} -> "
            f"{item['human_decision']}"
        )

    log("")
    log("All five tests completed successfully.")

    log("")
    log(f"Output saved automatically to: {OUTPUT_FILE}")