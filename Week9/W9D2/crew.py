from crewai import Agent, Task, Crew, Process, LLM
from tools import WebSearchTool


# --------------------------------------------------
# LOCAL OLLAMA LLM
# --------------------------------------------------

local_llm = LLM(
    model="ollama/llama3.2:3b",
    base_url="http://localhost:11434",
    temperature=0.2
)


# --------------------------------------------------
# WEB SEARCH TOOL
# --------------------------------------------------

web_search_tool = WebSearchTool()


# --------------------------------------------------
# AGENT 1: RESEARCHER
# --------------------------------------------------

researcher = Agent(
    role="AI Researcher",

    goal=(
        "Research Retrieval Augmented Generation (RAG) "
        "using current web information and identify accurate "
        "technical facts, applications, benefits, and limitations."
    ),

    backstory=(
        "You are an experienced AI researcher who specializes "
        "in Large Language Models, Retrieval Augmented Generation, "
        "vector databases, embeddings, and modern AI systems. "
        "You carefully collect reliable information before writing."
    ),

    tools=[web_search_tool],

    llm=local_llm,

    verbose=True,

    allow_delegation=False
)


# --------------------------------------------------
# AGENT 2: WRITER
# --------------------------------------------------

writer = Agent(
    role="Technical Writer",

    goal=(
        "Convert the research findings into a clear, accurate "
        "and well-structured technical article about RAG."
    ),

    backstory=(
        "You are a technical writer experienced in explaining "
        "complex artificial intelligence concepts in simple language. "
        "You organize information logically and avoid unnecessary jargon."
    ),

    llm=local_llm,

    verbose=True,

    allow_delegation=False
)


# --------------------------------------------------
# AGENT 3: REVIEWER
# --------------------------------------------------

reviewer = Agent(
    role="Technical Reviewer",

    goal=(
        "Review the written RAG article for correctness, "
        "clarity, completeness, and consistency with the research."
    ),

    backstory=(
        "You are a senior AI reviewer who checks technical content "
        "for factual errors, missing information, unclear explanations, "
        "and unsupported claims."
    ),

    llm=local_llm,

    verbose=True,

    allow_delegation=False
)


# --------------------------------------------------
# TASK 1: WEB RESEARCH
# --------------------------------------------------

research_task = Task(
    description=(
        "Research the topic 'Retrieval Augmented Generation (RAG)'. "
        "Use the Web Search tool to find current information. "
        "Cover the following points:\n"
        "1. What is RAG?\n"
        "2. How does RAG work?\n"
        "3. Main components of a RAG system.\n"
        "4. Benefits of RAG.\n"
        "5. Limitations of RAG.\n"
        "6. Real-world applications.\n"
        "7. Difference between RAG and traditional LLM generation.\n\n"
        "Return concise but useful research notes."
    ),

    expected_output=(
        "A structured research report containing current web-based "
        "information about RAG, including definition, architecture, "
        "workflow, benefits, limitations, and applications."
    ),

    agent=researcher
)


# --------------------------------------------------
# TASK 2: WRITE ARTICLE
# --------------------------------------------------

writing_task = Task(
    description=(
        "Using the research produced by the researcher, write a "
        "clear technical article about Retrieval Augmented Generation. "
        "The article should include:\n\n"
        "- Introduction\n"
        "- What is RAG?\n"
        "- How RAG works\n"
        "- Main components\n"
        "- Advantages\n"
        "- Limitations\n"
        "- Real-world applications\n"
        "- RAG vs traditional LLM generation\n"
        "- Conclusion\n\n"
        "Use simple and professional technical language."
    ),

    expected_output=(
        "A well-structured technical article about RAG that is "
        "accurate, readable, and based on the research findings."
    ),

    agent=writer,

    context=[research_task]
)


# --------------------------------------------------
# TASK 3: REVIEW ARTICLE
# --------------------------------------------------

review_task = Task(
    description=(
        "Review the article created by the writer. "
        "Check it for:\n\n"
        "1. Technical correctness\n"
        "2. Clarity\n"
        "3. Completeness\n"
        "4. Logical structure\n"
        "5. Unsupported or inaccurate claims\n"
        "6. Grammar and readability\n\n"
        "If improvements are needed, provide the corrected version. "
        "Return the final polished article."
    ),

    expected_output=(
        "A final reviewed and corrected technical article about RAG."
    ),

    agent=reviewer,

    context=[research_task, writing_task]
)


# --------------------------------------------------
# CREATE CREW
# --------------------------------------------------

crew = Crew(
    agents=[
        researcher,
        writer,
        reviewer
    ],

    tasks=[
        research_task,
        writing_task,
        review_task
    ],

    process=Process.sequential,

    verbose=True
)


# --------------------------------------------------
# RUN CREW
# --------------------------------------------------

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("CREWAI RAG RESEARCH CREW")
    print("=" * 70)
    print("LLM: Ollama - llama3.2:3b")
    print("Agents: Researcher -> Writer -> Reviewer")
    print("Web Search: Enabled")
    print("=" * 70)
    print("\n")

    try:

        result = crew.kickoff()

        print("\n")
        print("=" * 70)
        print("FINAL RESULT")
        print("=" * 70)
        print(result)
        print("=" * 70)

        # Save final answer
        with open(
            "research_answer.txt",
            "w",
            encoding="utf-8"
        ) as file:

            file.write(str(result))

        print("\nFinal answer saved to: research_answer.txt")

    except Exception as e:

        print("\n")
        print("=" * 70)
        print("CREW EXECUTION FAILED")
        print("=" * 70)
        print(f"Error: {e}")
        print("=" * 70)