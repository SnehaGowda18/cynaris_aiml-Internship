import os
from pathlib import Path

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool


# ============================================================
# W9D3 - BUILDING A RESEARCH CREW
# Multi-Agent Pipeline with Web Search
# ============================================================

load_dotenv()

TOPIC = "Impact of Generative AI on Software Development"

BASE_DIR = Path(__file__).resolve().parent

RESEARCH_FILE = BASE_DIR / "web_research_output.txt"
FINAL_FILE = BASE_DIR / "research_output.txt"


# ============================================================
# CHECK API KEY
# ============================================================

serper_key = os.getenv("SERPER_API_KEY")

if not serper_key:
    print("\nERROR: SERPER_API_KEY is missing.")
    print("Create a .env file in the W9D3 folder:")
    print("SERPER_API_KEY=your_api_key_here")
    raise ValueError("SERPER_API_KEY is missing.")


# ============================================================
# OLLAMA LLM
# ============================================================

llm = LLM(
    model="ollama/llama3.2:3b",
    base_url="http://localhost:11434",
)


# ============================================================
# WEB SEARCH TOOL
# ============================================================

search_tool = SerperDevTool()


# ============================================================
# AGENT 1 - RESEARCHER
# ============================================================

researcher = Agent(
    role="Technology Researcher",

    goal=(
        "Research the given topic using current web information. "
        "Find reliable facts, applications, benefits, challenges, "
        "security concerns, real-world examples, and future trends."
    ),

    backstory=(
        "You are an experienced technology researcher specializing "
        "in Artificial Intelligence, Generative AI, and Software "
        "Engineering. You collect relevant information from reliable "
        "web sources and organize it into useful research notes."
    ),

    tools=[search_tool],

    llm=llm,

    verbose=True,

    allow_delegation=False,
)


# ============================================================
# AGENT 2 - WRITER
# ============================================================

writer = Agent(
    role="Technical Writer",

    goal=(
        "Transform the researcher's findings into a clear, accurate "
        "and well-structured technical report about Generative AI "
        "and software development."
    ),

    backstory=(
        "You are a professional technical writer experienced in "
        "Artificial Intelligence and Software Engineering. You "
        "convert complex research into readable and organized "
        "technical documentation."
    ),

    llm=llm,

    verbose=True,

    allow_delegation=False,
)


# ============================================================
# AGENT 3 - REVIEWER
# ============================================================

reviewer = Agent(
    role="Research Quality Reviewer",

    goal=(
        "Review the written report for accuracy, completeness, "
        "clarity, logical structure, unsupported claims, and "
        "overall research quality."
    ),

    backstory=(
        "You are a senior AI and software engineering reviewer. "
        "You evaluate technical reports and identify weaknesses, "
        "missing information, unsupported statements, and areas "
        "that need improvement."
    ),

    llm=llm,

    verbose=True,

    allow_delegation=False,
)


# ============================================================
# TASK 1 - WEB RESEARCH
# ============================================================

research_task = Task(
    description=(
        f"Research the following topic:\n\n"
        f"{TOPIC}\n\n"

        "Use the web search tool to gather current information.\n\n"

        "Cover the following areas:\n"
        "1. Introduction to Generative AI in software development\n"
        "2. Major applications and use cases\n"
        "3. Benefits for developers\n"
        "4. Productivity improvements\n"
        "5. Challenges and limitations\n"
        "6. Security and privacy concerns\n"
        "7. Impact on developer roles and skills\n"
        "8. Real-world examples\n"
        "9. Future trends\n\n"

        "Include relevant source names or references where possible.\n"
        "Do not invent sources or unsupported facts.\n\n"

        "Return detailed and structured research notes."
    ),

    expected_output=(
        "A structured research document containing current web-based "
        "findings, important facts, applications, benefits, risks, "
        "examples, future trends, and relevant references."
    ),

    agent=researcher,
)


# ============================================================
# TASK 2 - WRITING
# ============================================================

writing_task = Task(
    description=(
        f"Using the researcher's findings, write a professional "
        f"technical report on:\n\n"
        f"{TOPIC}\n\n"

        "Use the following structure:\n\n"

        "1. Introduction\n"
        "2. Applications of Generative AI\n"
        "3. Benefits for Software Development\n"
        "4. Challenges and Limitations\n"
        "5. Security and Privacy Considerations\n"
        "6. Impact on Software Developers\n"
        "7. Future Trends\n"
        "8. Conclusion\n"
        "9. References\n\n"

        "Use clear and professional technical language.\n"
        "Base the report on the research provided by the Researcher.\n"
        "Avoid unsupported claims."
    ),

    expected_output=(
        "A complete technical report with clear headings, detailed "
        "explanations, examples, balanced discussion, conclusion, "
        "and references."
    ),

    agent=writer,

    context=[research_task],
)


# ============================================================
# TASK 3 - REVIEW
# ============================================================

review_task = Task(
    description=(
        "Review the technical report produced by the Writer.\n\n"

        "Evaluate the report for:\n"
        "1. Factual accuracy\n"
        "2. Relevance to the topic\n"
        "3. Completeness\n"
        "4. Logical organization\n"
        "5. Clarity and readability\n"
        "6. Unsupported claims\n"
        "7. Repetition\n"
        "8. Balance between benefits and risks\n"
        "9. Quality of references\n\n"

        "Provide:\n"
        "- Overall quality assessment\n"
        "- Strengths\n"
        "- Weaknesses\n"
        "- Specific improvements\n"
        "- Final reviewer verdict"
    ),

    expected_output=(
        "A detailed review containing an overall quality assessment, "
        "strengths, weaknesses, recommended improvements, and a final "
        "reviewer verdict."
    ),

    agent=reviewer,

    context=[research_task, writing_task],
)


# ============================================================
# CREATE CREW
# ============================================================

crew = Crew(
    agents=[
        researcher,
        writer,
        reviewer,
    ],

    tasks=[
        research_task,
        writing_task,
        review_task,
    ],

    process=Process.sequential,

    verbose=True,
)


# ============================================================
# RUN CREW
# ============================================================

print("\n" + "=" * 70)
print("W9D3 - MULTI-AGENT RESEARCH CREW")
print("=" * 70)

print(f"\nTopic: {TOPIC}")

print("\nAgents:")
print("1. Researcher - Web Research")
print("2. Writer - Technical Report")
print("3. Reviewer - Quality Review")

print("\nPipeline:")
print("Researcher -> Writer -> Reviewer")

print("\nStarting CrewAI execution...")
print("=" * 70)


try:

    result = crew.kickoff()

    # --------------------------------------------------------
    # GET FINAL OUTPUT
    # --------------------------------------------------------

    final_output = str(result)

    print("\n")
    print("=" * 70)
    print("CREW EXECUTION COMPLETED")
    print("=" * 70)

    print("\nFINAL OUTPUT:\n")
    print(final_output)


    # ========================================================
    # SAVE FINAL OUTPUT
    # ========================================================

    with open(FINAL_FILE, "w", encoding="utf-8") as file:
        file.write(final_output)

    print("\n" + "=" * 70)
    print("FINAL OUTPUT SAVED")
    print("=" * 70)

    print(f"File: {FINAL_FILE}")
    print(f"Size: {FINAL_FILE.stat().st_size} bytes")


    # ========================================================
    # SAVE RESEARCH TASK OUTPUT
    # ========================================================

    try:

        research_output = str(research_task.output)

        with open(RESEARCH_FILE, "w", encoding="utf-8") as file:
            file.write(research_output)

        print("\n" + "=" * 70)
        print("WEB RESEARCH OUTPUT SAVED")
        print("=" * 70)

        print(f"File: {RESEARCH_FILE}")
        print(f"Size: {RESEARCH_FILE.stat().st_size} bytes")

    except Exception as research_error:

        print("\nCould not save research task output:")
        print(research_error)


    # ========================================================
    # FINAL FILE CHECK
    # ========================================================

    print("\n" + "=" * 70)
    print("OUTPUT FILE CHECK")
    print("=" * 70)

    print(
        f"research_output.txt: "
        f"{FINAL_FILE.stat().st_size} bytes"
    )

    print(
        f"web_research_output.txt: "
        f"{RESEARCH_FILE.stat().st_size} bytes"
    )

    print("\nW9D3 research crew completed successfully.")


except Exception as error:

    print("\n" + "=" * 70)
    print("CREW EXECUTION FAILED")
    print("=" * 70)

    print(f"\nError type: {type(error).__name__}")
    print(f"Error: {error}")

    print("\nCheck:")
    print("1. Ollama is running")
    print("2. llama3.2:3b is installed")
    print("3. SERPER_API_KEY is present in .env")
    print("4. Internet connection is available")
    print("5. crewai-tools is installed")