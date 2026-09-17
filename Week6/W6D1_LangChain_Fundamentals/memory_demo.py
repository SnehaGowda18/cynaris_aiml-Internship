from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage


# Create Ollama chat model
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

# Conversation history
history = []


# Five conversation turns
questions = [
    "My name is Sneha.",
    "What is my name?",
    "I am learning DevOps.",
    "What am I learning?",
    "Can you summarize what you know about me?"
]


# Run conversation
for question in questions:

    messages = history + [HumanMessage(content=question)]

    response = llm.invoke(messages)

    print("\nUser:", question)
    print("AI:", response.content)

    # Save conversation history
    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=response.content))