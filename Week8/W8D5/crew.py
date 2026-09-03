"""
CrewAI component for the Local AI Research Assistant.

Uses Ollama locally instead of OpenAI.
"""

from crewai import Agent, Task, Crew, LLM


# Configure CrewAI to use the local Ollama model.
local_llm = LLM(
    model="ollama/llama3.2:3b",
    base_url="http://localhost:11434",
    temperature=0,
)


def create_research_crew(context: str, question: str):
    """Create a CrewAI research crew using local Ollama."""

    researcher = Agent(
        role="Local Research Analyst",
        goal="Analyze local research information and answer questions accurately.",
        backstory=(
            "You are a research analyst who works with local technical "
            "documents and provides clear, factual answers."
        ),
        llm=local_llm,
        verbose=True,
        allow_delegation=False,
    )

    research_task = Task(
        description=f"""
Answer the following research question using ONLY the
provided local research context.

Research Question:
{question}

Local Research Context:
{context}

Requirements:
- Give an accurate answer.
- Do not invent information.
- Keep the explanation clear and concise.
""",
        expected_output=(
            "A concise and accurate research answer based only "
            "on the provided local research context."
        ),
        agent=researcher,
    )

    crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        verbose=True,
    )

    return crew


if __name__ == "__main__":

    question = "What is Retrieval Augmented Generation?"

    context = """
    Retrieval Augmented Generation (RAG) combines information
    retrieval with generative language models. A RAG system
    retrieves relevant documents from a knowledge base and
    provides them to an LLM as context for generating an answer.
    """

    research_crew = create_research_crew(
        context=context,
        question=question,
    )

    result = research_crew.kickoff()

    print("\n" + "=" * 60)
    print("CREWAI LOCAL RESEARCH RESULT")
    print("=" * 60)
    print(result)