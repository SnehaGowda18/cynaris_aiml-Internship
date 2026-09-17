from pathlib import Path

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM


# ============================================================
# W6D2: LANGCHAIN MEMORY & CONVERSATION HISTORY
# ============================================================

# Existing output folder
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# OLLAMA MODEL
# ============================================================

llm = OllamaLLM(
    model="llama3.2:3b"
)

parser = StrOutputParser()


# ============================================================
# TASK 1: LANGCHAIN CHAIN
# PromptTemplate -> Ollama LLM -> OutputParser
# ============================================================

prompt = PromptTemplate(
    input_variables=["question"],
    template="""
Answer the question accurately and simply.
Give the answer in 2-3 sentences.

Question: {question}
"""
)

chain = prompt | llm | parser


questions = [
    "What is artificial intelligence?",
    "What is machine learning?",
    "What is LangChain?",
    "What is an LLM?",
    "What is cloud computing?"
]


chain_results = []

print("=" * 60)
print("W6D2 - LANGCHAIN CHAIN")
print("=" * 60)

for i, question in enumerate(questions, start=1):

    response = chain.invoke({
        "question": question
    })

    output = (
        f"Input {i}: {question}\n"
        f"Output: {response}\n"
        f"{'-' * 60}\n"
    )

    print(output)

    chain_results.append(output)


# Save Task 1 output
with open(
    OUTPUT_DIR / "chain_output.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("=" * 60 + "\n")
    file.write("W6D2 - LANGCHAIN CHAIN OUTPUT\n")
    file.write("=" * 60 + "\n\n")

    file.write("\n".join(chain_results))


# ============================================================
# TASK 2: CONVERSATION MEMORY
# ============================================================

print()
print("=" * 60)
print("CONVERSATION HISTORY")
print("=" * 60)


conversation_history = []


def chat(user_message):

    # Add user message to history
    conversation_history.append(
        f"User: {user_message}"
    )

    history = "\n".join(
        conversation_history
    )

    memory_prompt = PromptTemplate(
        input_variables=[
            "history",
            "question"
        ],
        template="""
You are a helpful assistant.

Previous conversation:
{history}

Current question:
{question}

Use the previous conversation when necessary.
Answer clearly and simply.
"""
    )

    memory_chain = (
        memory_prompt
        | llm
        | parser
    )

    response = memory_chain.invoke({
        "history": history,
        "question": user_message
    })

    # Save assistant response
    conversation_history.append(
        f"Assistant: {response}"
    )

    return response


turns = [
    "My name is Sneha.",
    "I am studying Information Science.",
    "What is my name?",
    "What am I studying?",
    "What do you know about me from this conversation?"
]


memory_results = []


for i, message in enumerate(turns, start=1):

    response = chat(message)

    output = (
        f"Turn {i}\n"
        f"User: {message}\n"
        f"Assistant: {response}\n"
        f"{'-' * 60}\n"
    )

    print(output)

    memory_results.append(output)


# Save Task 2 output
with open(
    OUTPUT_DIR / "memory_output.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("=" * 60 + "\n")
    file.write("W6D2 - CONVERSATION MEMORY OUTPUT\n")
    file.write("=" * 60 + "\n\n")

    file.write("\n".join(memory_results))


# ============================================================
# TASK 3: SIMPLE AGENT WITH TWO TOOLS
# ============================================================

print()
print("=" * 60)
print("SIMPLE AGENT TOOLS")
print("=" * 60)


# ------------------------------------------------------------
# Tool 1: Calculator
# ------------------------------------------------------------

def calculator(expression):
    """
    Simple calculator tool.
    """

    try:

        result = eval(
            expression,
            {
                "__builtins__": {}
            },
            {}
        )

        return str(result)

    except Exception:

        return "Invalid calculation"


# ------------------------------------------------------------
# Tool 2: Web Search Stub
# ------------------------------------------------------------

def web_search_stub(query):
    """
    Simulated web search tool.
    """

    return (
        f"Web search result for '{query}': "
        "This is a simulated web search result."
    )


# ------------------------------------------------------------
# Three agent tasks
# ------------------------------------------------------------

tasks = [
    ("calculator", "25 * 4"),
    ("calculator", "100 / 5 + 10"),
    ("web_search", "What is LangChain?")
]


agent_results = []


for i, (tool, task) in enumerate(tasks, start=1):

    if tool == "calculator":

        result = calculator(task)

    elif tool == "web_search":

        result = web_search_stub(task)

    else:

        result = "Unknown tool"


    output = (
        f"Task {i}\n"
        f"Tool: {tool}\n"
        f"Input: {task}\n"
        f"Result: {result}\n"
        f"{'-' * 60}\n"
    )

    print(output)

    agent_results.append(output)


# Save Task 3 output
with open(
    OUTPUT_DIR / "agent_output.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("=" * 60 + "\n")
    file.write("W6D2 - AGENT TOOLS OUTPUT\n")
    file.write("=" * 60 + "\n\n")

    file.write("\n".join(agent_results))


# ============================================================
# FINAL STATUS
# ============================================================

print()
print("=" * 60)
print("W6D2 COMPLETED SUCCESSFULLY!")
print("=" * 60)

print()
print("Output files updated:")
print("1. outputs/chain_output.txt")
print("2. outputs/memory_output.txt")
print("3. outputs/agent_output.txt")