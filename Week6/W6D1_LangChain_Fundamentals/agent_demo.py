from langchain.agents import create_agent
from langchain_ollama import ChatOllama


# Calculator tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception:
        return "Invalid mathematical expression."


# Web search stub
def web_search(query: str) -> str:
    """Stub for web search."""
    return (
        f"Search result for '{query}': "
        "LangChain is a framework for building applications using LLMs."
    )


# Create tools using LangChain tool decorator
from langchain_core.tools import tool


@tool
def calculator_tool(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception:
        return "Invalid mathematical expression."


@tool
def web_search_tool(query: str) -> str:
    """Search the web using a simple stub."""
    return (
        f"Search result for '{query}': "
        "LangChain is a framework for developing LLM applications."
    )


# Create tool list
tools = [
    calculator_tool,
    web_search_tool
]


# Create Ollama model
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


# Create LangChain agent
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant. Use tools when necessary."
)


# Three required tasks
tasks = [
    "Calculate 25 * 8",
    "Calculate 150 / 5 + 20",
    "Search for information about LangChain"
]


# Run the agent
for task in tasks:

    print("\n" + "=" * 60)
    print("TASK:", task)
    print("=" * 60)

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": task
                }
            ]
        }
    )

    messages = result["messages"]

    print("FINAL ANSWER:")
    print(messages[-1].content)