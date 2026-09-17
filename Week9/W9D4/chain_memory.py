from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser

# Prompt with conversation history
prompt = PromptTemplate(
    input_variables=["history", "question"],
    template="""You are a helpful AI assistant.

Conversation history:
{history}

Current question:
{question}

Answer the current question using the conversation history when useful.
Keep the answer simple.
"""
)

# Ollama model
llm = OllamaLLM(model="llama3.2:3b")

# Output parser
parser = StrOutputParser()

# Chain
chain = prompt | llm | parser

# Conversation history
history = []

# Five conversation turns
questions = [
    "My name is Sneha.",
    "What is my name?",
    "I am learning artificial intelligence.",
    "What am I learning?",
    "Can you summarize what you know about me from this conversation?"
]

print("=== Conversation Memory Test ===")

for i, question in enumerate(questions, start=1):

    history_text = "\n".join(history)

    answer = chain.invoke({
        "history": history_text,
        "question": question
    })

    print(f"\nTurn {i}")
    print("User:", question)
    print("AI:", answer)

    # Store conversation
    history.append(f"User: {question}")
    history.append(f"AI: {answer}")

print("\n=== Conversation History ===")
for item in history:
    print(item)