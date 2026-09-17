

from customer_support_agent import run_support_agent


def test_customer_name_memory():
    """
    Verify that the agent maintains the customer's name.
    """

    result = run_support_agent(
        "Sneha",
        "What is my name?"
    )

    assert result["customer_name"] == "Sneha"


def test_conversation_memory():
    """
    Verify that previous conversation messages
    are preserved in the next interaction.
    """

    first = run_support_agent(
        "Sneha",
        "My order has not arrived yet."
    )

    second = run_support_agent(
        "Sneha",
        "Can you help me check the order status?",
        first["messages"]
    )

    assert len(second["messages"]) > len(first["messages"])

    assert (
        "My order has not arrived yet."
        in second["messages"][0]
    )


def test_agent_response():
    """
    Verify that the agent returns a non-empty response.
    """

    result = run_support_agent(
        "Sneha",
        "I need help with my order."
    )

    assert result["response"]
    assert isinstance(result["response"], str)

