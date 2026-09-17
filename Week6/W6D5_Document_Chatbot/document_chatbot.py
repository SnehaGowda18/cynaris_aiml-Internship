from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from langchain_core.tools import tool
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "llama3.2:3b"

# Create outputs folder automatically
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "w6d5_output.txt"


# ============================================================
# OUTPUT LOGGER
# ============================================================

def log(text=""):
    """
    Print output to terminal and save it to the output file.
    """

    print(text)

    with open(OUTPUT_FILE, "a", encoding="utf-8") as file:
        file.write(str(text) + "\n")


# Clear old output file at the beginning
with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    file.write("W6D5 - DOCUMENT CHATBOT WITH LANGCHAIN\n")
    file.write("=" * 60 + "\n\n")


# ============================================================
# OLLAMA MODEL
# ============================================================

llm = OllamaLLM(
    model=MODEL_NAME
)

parser = StrOutputParser()


# ============================================================
# PART 1: LANGCHAIN CHAIN
# ============================================================

prompt = PromptTemplate(
    input_variables=["question"],
    template="""
You are a helpful AI assistant.

Answer the question clearly and briefly.

Question:
{question}

Answer:
"""
)

chain = prompt | llm | parser


def test_chain():

    log("\n")
    log("=" * 60)
    log("PART 1: LANGCHAIN CHAIN")
    log("=" * 60)

    questions = [
        "What is Python?",
        "What is Machine Learning?",
        "What is LangChain?",
        "What is a vector database?",
        "What is RAG?"
    ]

    for i, question in enumerate(questions, start=1):

        log(f"\nTest {i}")
        log(f"Question: {question}")

        try:

            answer = chain.invoke({
                "question": question
            })

            log(f"Answer: {answer}")

        except Exception as e:

            log(f"Error: {e}")


# ============================================================
# PART 2: CONVERSATION MEMORY
# ============================================================

conversation_history = []


def chat_with_memory(user_input):

    conversation_history.append(
        f"User: {user_input}"
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
You are a helpful conversational AI assistant.

Use the previous conversation to understand
the current question.

Previous conversation:
{history}

Current question:
{question}

Give a clear answer.
"""
    )

    memory_chain = memory_prompt | llm | parser

    response = memory_chain.invoke({
        "history": history,
        "question": user_input
    })

    conversation_history.append(
        f"Assistant: {response}"
    )

    return response


def test_memory():

    log("\n")
    log("=" * 60)
    log("PART 2: CONVERSATION MEMORY")
    log("=" * 60)

    questions = [
        "My name is Sneha.",
        "What is my name?",
        "I am learning Artificial Intelligence.",
        "What am I learning?",
        "What is my name and what am I learning?"
    ]

    for i, question in enumerate(questions, start=1):

        log(f"\nTurn {i}")
        log(f"User: {question}")

        try:

            response = chat_with_memory(
                question
            )

            log(f"Assistant: {response}")

        except Exception as e:

            log(f"Error: {e}")


# ============================================================
# PART 3: TOOLS
# ============================================================

@tool
def calculator(expression: str) -> str:
    """
    Calculate a mathematical expression.
    """

    try:

        result = eval(
            expression,
            {"__builtins__": {}},
            {}
        )

        return str(result)

    except Exception:

        return "Invalid mathematical expression."


@tool
def web_search_stub(query: str) -> str:
    """
    Simulated web search tool.
    """

    return (
        f"Search result for '{query}': "
        f"LangChain is a framework used to build "
        f"applications powered by language models."
    )


# ============================================================
# PART 4: AGENT
# ============================================================

def run_agent(task):

    log(f"\nTask: {task}")

    # Calculator task
    if any(
        symbol in task
        for symbol in ["+", "-", "*", "/"]
    ):

        result = calculator.invoke(task)

        log("Selected Tool: Calculator")
        log(f"Result: {result}")

        return result

    # Web search task
    else:

        result = web_search_stub.invoke(task)

        log("Selected Tool: Web Search Stub")
        log(f"Result: {result}")

        return result


def test_agent():

    log("\n")
    log("=" * 60)
    log("PART 3: AGENT WITH TWO TOOLS")
    log("=" * 60)

    tasks = [
        "25 * 4",
        "150 / 5 + 10",
        "What is LangChain?"
    ]

    for i, task in enumerate(tasks, start=1):

        log(f"\nAgent Task {i}")

        try:

            run_agent(task)

        except Exception as e:

            log(f"Error: {e}")


# ============================================================
# MAIN
# ============================================================

def main():

    log("=" * 60)
    log("W6D5 - DOCUMENT CHATBOT WITH LANGCHAIN")
    log("=" * 60)

    log(f"Model: {MODEL_NAME}")

    # Part 1
    test_chain()

    # Part 2
    test_memory()

    # Part 3
    test_agent()

    log("\n")
    log("=" * 60)
    log("W6D5 COMPLETED SUCCESSFULLY")
    log("=" * 60)

    log(f"\nOutput saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()