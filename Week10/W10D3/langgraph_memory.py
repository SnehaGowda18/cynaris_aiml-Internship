

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama


# --------------------------------------------------
# 1. LLM SETUP
# --------------------------------------------------

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


# --------------------------------------------------
# 2. STATE DEFINITION
# --------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], "Conversation history"]
    category: str
    response: str


# --------------------------------------------------
# 3. CLASSIFY NODE
# --------------------------------------------------

def classify_node(state: AgentState):

    last_message = state["messages"][-1].content

    prompt = f"""
Classify the following user input into exactly one category:

QUESTION
TASK
OTHER

User input:
{last_message}

Return only the category name.
"""

    result = llm.invoke(prompt)

    category = result.content.strip().upper()

    if category not in ["QUESTION", "TASK", "OTHER"]:
        category = "OTHER"

    print(f"\n[CLASSIFY] {category}")

    return {
        "category": category
    }


# --------------------------------------------------
# 4. ROUTE NODE
# --------------------------------------------------

def route_node(state: AgentState):

    category = state["category"]

    print(f"[ROUTE] Sending input to {category} route")

    return {}


# --------------------------------------------------
# 5. RESPOND NODE
# --------------------------------------------------

def respond_node(state: AgentState):

    messages = state["messages"]
    category = state["category"]

    latest_message = messages[-1].content

    # Human approval for tasks
    if category == "TASK":

        approval = interrupt(
            {
                "type": "human_approval",
                "message": (
                    "The user submitted a task. "
                    "Should the agent continue?"
                ),
                "user_input": latest_message
            }
        )

        print(f"[HUMAN APPROVAL] {approval}")

        if isinstance(approval, dict):
            decision = approval.get("decision", "reject")
        else:
            decision = str(approval)

        if str(decision).lower() not in ["approve", "approved", "yes"]:
            response_text = "The task was stopped by human approval."

            print(f"[RESPOND] {response_text}")

            return {
                "messages": [
                    AIMessage(content=response_text)
                ],
                "response": response_text
            }

    # Include conversation history for persistent memory
    history = "\n".join(
        f"{type(message).__name__}: {message.content}"
        for message in messages
    )

    prompt = f"""
You are a helpful AI assistant.

Conversation history:
{history}

Current category:
{category}

Answer the user's latest message clearly and helpfully.
"""

    result = llm.invoke(prompt)

    response_text = result.content

    print(f"[RESPOND] {response_text}")

    return {
        "messages": [
            AIMessage(content=response_text)
        ],
        "response": response_text
    }


# --------------------------------------------------
# 6. CONDITIONAL ROUTING
# --------------------------------------------------

def choose_route(state: AgentState):

    category = state["category"]

    if category == "QUESTION":
        return "respond"

    elif category == "TASK":
        return "respond"

    else:
        return "respond"


# --------------------------------------------------
# 7. BUILD GRAPH
# --------------------------------------------------

def build_graph():

    workflow = StateGraph(AgentState)

    workflow.add_node("classify", classify_node)
    workflow.add_node("route", route_node)
    workflow.add_node("respond", respond_node)

    workflow.add_edge(START, "classify")
    workflow.add_edge("classify", "route")

    workflow.add_conditional_edges(
        "route",
        choose_route,
        {
            "respond": "respond"
        }
    )

    workflow.add_edge("respond", END)

    # Persistent conversation memory
    memory = MemorySaver()

    graph = workflow.compile(
        checkpointer=memory
    )

    return graph


# --------------------------------------------------
# 8. RUN ONE CONVERSATION
# --------------------------------------------------

def run_conversation(graph, thread_id, user_input):

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    print("\n" + "=" * 60)
    print(f"USER: {user_input}")
    print("=" * 60)

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ],
            "category": "",
            "response": ""
        },
        config=config
    )

    # Human interrupt handling
    if "__interrupt__" in result:

        print("\n[INTERRUPTED]")
        print(result["__interrupt__"])

        decision = input(
            "\nEnter human decision (approve/reject): "
        ).strip().lower()

        resume_result = graph.invoke(
            Command(
                resume={
                    "decision": decision
                }
            ),
            config=config
        )

        result = resume_result

    print("\nASSISTANT:")

    if result.get("response"):
        print(result["response"])

    else:
        for message in reversed(result["messages"]):
            if isinstance(message, AIMessage):
                print(message.content)
                break


# --------------------------------------------------
# 9. TESTING
# --------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("W10D3: LANGGRAPH + MEMORY")
    print("=" * 60)

    graph = build_graph()

    thread_id = "sneha-w10d3-conversation"

    # Test 1
    run_conversation(
        graph,
        thread_id,
        "My name is Sneha."
    )

    # Test 2
    run_conversation(
        graph,
        thread_id,
        "What is my name?"
    )

    # Test 3
    run_conversation(
        graph,
        thread_id,
        "What is artificial intelligence?"
    )

    # Test 4
    run_conversation(
        graph,
        thread_id,
        "Explain how LangGraph memory works."
    )

    # Test 5
    run_conversation(
        graph,
        thread_id,
        "Create a short study plan for learning Python."
    )

    print("\n" + "=" * 60)
    print("ALL FIVE TESTS COMPLETED")
    print("=" * 60)