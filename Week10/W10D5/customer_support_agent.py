

from typing import List, TypedDict

import mlflow
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph


# ============================================================
# 1. MLflow Configuration
# ============================================================

# SQLite is used because the latest MLflow versions
# no longer support the old filesystem tracking backend.
mlflow.set_tracking_uri("sqlite:///mlflow.db")

mlflow.set_experiment("Stateful-Customer-Support-Agent")


# ============================================================
# 2. Define Customer Support State
# ============================================================

class CustomerSupportState(TypedDict):
    customer_name: str
    messages: List[str]
    current_query: str
    response: str


# ============================================================
# 3. Initialize Local Ollama Model
# ============================================================

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


# ============================================================
# 4. Customer Support Agent Node
# ============================================================

def support_agent(state: CustomerSupportState):
    """
    Generate a customer support response.

    The agent receives previous messages and the
    current question to maintain conversation context.
    """

    conversation_history = "\n".join(state["messages"])

    prompt = f"""
You are a helpful and professional customer support agent.

Customer name: {state['customer_name']}

Previous conversation:
{conversation_history}

Current customer question:
{state['current_query']}

Instructions:
- Answer clearly and politely.
- Remember relevant previous conversation.
- Do not invent order numbers, tracking information,
  delivery dates, or real system records.
- If information is missing, ask the customer for it.
- Do not claim to have checked a real order system.
- Keep the response concise.
"""

    result = llm.invoke(prompt)

    response = result.content

    updated_messages = state["messages"] + [
        f"Customer: {state['current_query']}",
        f"Agent: {response}"
    ]

    return {
        "response": response,
        "messages": updated_messages
    }


# ============================================================
# 5. Build LangGraph Workflow
# ============================================================

workflow = StateGraph(CustomerSupportState)

workflow.add_node("support_agent", support_agent)

workflow.add_edge(START, "support_agent")
workflow.add_edge("support_agent", END)

app = workflow.compile()


# ============================================================
# 6. Run Stateful Customer Support Agent
# ============================================================

def run_support_agent(
    customer_name: str,
    query: str,
    previous_messages: List[str] | None = None
):
    """
    Run one customer support conversation turn.

    Args:
        customer_name: Name of the customer.
        query: Current customer question.
        previous_messages: Conversation history.

    Returns:
        Updated customer support state.
    """

    if previous_messages is None:
        previous_messages = []

    state: CustomerSupportState = {
        "customer_name": customer_name,
        "messages": previous_messages,
        "current_query": query,
        "response": ""
    }

    # Track each agent interaction using MLflow.
    with mlflow.start_run(run_name="customer-support-interaction"):

        mlflow.log_param("customer_name", customer_name)
        mlflow.log_param("query", query)
        mlflow.log_param(
            "previous_message_count",
            len(previous_messages)
        )

        result = app.invoke(state)

        mlflow.log_metric(
            "message_count",
            len(result["messages"])
        )

        mlflow.log_metric(
            "response_length",
            len(result["response"])
        )

        return result


# ============================================================
# 7. Main Program
# ============================================================

if __name__ == "__main__":

    print("=== Stateful Customer Support Agent ===")

    history = []

    # --------------------------------------------------------
    # Conversation 1
    # --------------------------------------------------------

    result = run_support_agent(
        "Sneha",
        "My order has not arrived yet.",
        history
    )

    print("\nCustomer: My order has not arrived yet.")
    print("Agent:", result["response"])

    history = result["messages"]

    # --------------------------------------------------------
    # Conversation 2
    # --------------------------------------------------------

    result = run_support_agent(
        "Sneha",
        "Can you help me check the order status?",
        history
    )

    print("\nCustomer: Can you help me check the order status?")
    print("Agent:", result["response"])

    history = result["messages"]

    # --------------------------------------------------------
    # Conversation 3
    # --------------------------------------------------------

    result = run_support_agent(
        "Sneha",
        "What is my name?",
        history
    )

    print("\nCustomer: What is my name?")
    print("Agent:", result["response"])

    # --------------------------------------------------------
    # Display Conversation Memory
    # --------------------------------------------------------

    print("\n=== Conversation Memory ===")

    for message in result["messages"]:
        print(message)
