import os
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool


# ============================================================
# LOCAL OLLAMA LLM
# ============================================================

llm = LLM(
    model="ollama/llama3.2:3b",
    base_url="http://localhost:11434"
)


# ============================================================
# TOPIC
# ============================================================

TOPIC = "Impact of Artificial Intelligence on Education"


# ============================================================
# 1. RESEARCH WITHOUT WEB
# ============================================================

researcher_without_web = Agent(
    role="Researcher",
    goal="Research the given topic using your existing knowledge.",
    backstory=(
        "You are an experienced research analyst who provides "
        "accurate, structured, and easy-to-understand information."
    ),
    llm=llm,
    verbose=True
)

research_task_without_web = Task(
    description=f"""
    Research the topic: "{TOPIC}".

    Do NOT use any external web search.

    Explain:
    1. Introduction
    2. Applications of AI in education
    3. Benefits
    4. Challenges and limitations
    5. Future possibilities

    Provide a structured research summary.
    """,
    expected_output=(
        "A structured research summary covering introduction, "
        "applications, benefits, challenges, limitations, "
        "and future possibilities."
    ),
    agent=researcher_without_web
)

crew_without_web = Crew(
    agents=[researcher_without_web],
    tasks=[research_task_without_web],
    process=Process.sequential,
    verbose=True
)


# ============================================================
# 2. RUN WITHOUT WEB
# ============================================================

print("\n" + "=" * 60)
print("RESEARCH WITHOUT WEB")
print("=" * 60)

result_without_web = crew_without_web.kickoff()

with open("research_without_web.txt", "w", encoding="utf-8") as file:
    file.write(str(result_without_web))

print("\nResearch without web saved to:")
print("research_without_web.txt")


# ============================================================
# 3. WEB SEARCH TOOL
# ============================================================

search_tool = SerperDevTool()


# ============================================================
# 4. RESEARCH WITH WEB
# ============================================================

researcher_with_web = Agent(
    role="Web Researcher",
    goal=(
        "Research the given topic using current information "
        "from the web and provide reliable, relevant findings."
    ),
    backstory=(
        "You are an expert web research analyst. "
        "You search the internet for current information, "
        "compare findings, and produce structured research."
    ),
    tools=[search_tool],
    llm=llm,
    verbose=True
)

research_task_with_web = Task(
    description=f"""
    Research the topic: "{TOPIC}".

    Use the web search tool to find current and relevant information.

    Search for information about:
    1. Current applications of AI in education
    2. Benefits of AI in education
    3. Challenges and limitations
    4. Recent developments
    5. Future possibilities

    Use information found through web search and clearly organize
    the research findings.

    Do not simply rely on your existing knowledge.
    """,
    expected_output=(
        "A detailed web-based research report covering current "
        "applications, benefits, challenges, recent developments, "
        "and future possibilities of AI in education."
    ),
    agent=researcher_with_web
)

crew_with_web = Crew(
    agents=[researcher_with_web],
    tasks=[research_task_with_web],
    process=Process.sequential,
    verbose=True
)


# ============================================================
# 5. RUN WITH WEB
# ============================================================

print("\n" + "=" * 60)
print("RESEARCH WITH WEB")
print("=" * 60)

result_with_web = crew_with_web.kickoff()

with open("research_with_web.txt", "w", encoding="utf-8") as file:
    file.write(str(result_with_web))

print("\nResearch with web saved to:")
print("research_with_web.txt")


# ============================================================
# 6. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("W9D1 RESEARCH COMPLETED")
print("=" * 60)

print("\nGenerated files:")
print("1. research_without_web.txt")
print("2. research_with_web.txt")