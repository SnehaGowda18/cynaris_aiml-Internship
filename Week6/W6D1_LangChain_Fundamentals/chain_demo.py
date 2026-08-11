from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser


# Create prompt
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in simple terms in 3 sentences."
)

# Create Ollama LLM
llm = OllamaLLM(model="llama3.2:3b")

# Create output parser
parser = StrOutputParser()

# Create chain
chain = prompt | llm | parser


# Test with 5 inputs
topics = [
    "Cloud Computing",
    "Cybersecurity",
    "DevOps",
    "Artificial Intelligence",
    "Machine Learning"
]

for topic in topics:
    print("\nTopic:", topic)
    print("Answer:", chain.invoke({"topic": topic}))