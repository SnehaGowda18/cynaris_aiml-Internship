from langchain_ollama import OllamaLLM
from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """Search the web for information. This is a simulated web search stub."""
    return (
        f"Search result for '{query}': "
        "LangChain is a framework for developing applications powered by "
        "language models."
    )


@tool
def calculator(expression: str) -> str:
    """Calculate a basic mathematical expression."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"


# Create tools
tools = [web_search, calculator]

# Ollama LLM
llm = OllamaLLM(model="llama3.2:3b")


def run_agent(task):
    """Simple agent that selects a tool based on the task."""
    
    task_lower = task.lower()

    if any(word in task_lower for word in ["calculate", "compute", "+", "-", "*", "/"]):
        expression = task_lower.replace("calculate", "").replace("compute", "").strip()
        return calculator.invoke({"expression": expression})

    elif any(word in task_lower for word in ["search", "langchain", "information", "what is"]):
        return web_search.invoke({"query": task})

    else:
        return llm.invoke(task)


# Run 3 agent tasks
tasks = [
    "What is LangChain?",
    "Calculate 25 * 4",
    "Search for information about RAG"
]

print("=== Two-Tool Agent Test ===")

for i, task in enumerate(tasks, start=1):
    print(f"\nTask {i}")
    print("Input:", task)
    print("Output:", run_agent(task))