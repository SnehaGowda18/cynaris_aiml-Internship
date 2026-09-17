"""
W9D5 - Automated Research Report Agent

Stack:
- CrewAI: Multi-agent orchestration
- Ollama: Local LLM
- Llama 3.2 3B: Local language model

Agents:
1. Research Specialist
2. Research Analyst
3. Report Writer
"""

from pathlib import Path

from crewai import Agent, Crew, Process, Task


# ============================================================
# Configuration
# ============================================================

OLLAMA_MODEL = "ollama/llama3.2:3b"

REPORT_FILE = Path("research_report.txt")


# ============================================================
# Build Research Crew
# ============================================================

def build_research_crew(topic: str) -> Crew:
    """
    Create the CrewAI research crew.

    Args:
        topic: Research topic provided by the user.

    Returns:
        Configured CrewAI Crew.
    """

    # --------------------------------------------------------
    # Agent 1: Research Specialist
    # --------------------------------------------------------

    researcher = Agent(
        role="Research Specialist",
        goal=(
            f"Research the topic '{topic}' and collect accurate, "
            "relevant and useful information."
        ),
        backstory=(
            "You are an experienced research specialist. "
            "You identify important facts, explain key concepts, "
            "and organize information clearly."
        ),
        llm=OLLAMA_MODEL,
        verbose=True,
        allow_delegation=False,
    )

    # --------------------------------------------------------
    # Agent 2: Research Analyst
    # --------------------------------------------------------

    analyst = Agent(
        role="Research Analyst",
        goal=(
            "Analyze the research findings and identify the "
            "most important insights, advantages, disadvantages "
            "and practical implications."
        ),
        backstory=(
            "You are a critical research analyst. "
            "You carefully examine research findings, compare "
            "important points and produce meaningful conclusions."
        ),
        llm=OLLAMA_MODEL,
        verbose=True,
        allow_delegation=False,
    )

    # --------------------------------------------------------
    # Agent 3: Report Writer
    # --------------------------------------------------------

    writer = Agent(
        role="Report Writer",
        goal=(
            "Create a clear, structured and professional research "
            "report using the research findings and analysis."
        ),
        backstory=(
            "You are a professional technical writer. "
            "You transform research findings and analysis into "
            "a concise and well-organized report."
        ),
        llm=OLLAMA_MODEL,
        verbose=True,
        allow_delegation=False,
    )

    # ========================================================
    # Tasks
    # ========================================================

    # --------------------------------------------------------
    # Task 1: Research
    # --------------------------------------------------------

    research_task = Task(
        description=(
            f"Research the topic '{topic}'.\n\n"
            "Cover the following areas:\n"
            "1. Introduction to the topic\n"
            "2. Important concepts\n"
            "3. Key developments\n"
            "4. Benefits\n"
            "5. Limitations\n"
            "6. Real-world applications\n"
            "7. Major challenges\n"
            "8. Future possibilities\n\n"
            "Organize the findings clearly and avoid unsupported claims."
        ),
        expected_output=(
            "A structured research summary containing important "
            "facts, concepts, benefits, limitations, applications, "
            "challenges and future possibilities."
        ),
        agent=researcher,
    )

    # --------------------------------------------------------
    # Task 2: Analysis
    # --------------------------------------------------------

    analysis_task = Task(
        description=(
            "Analyze the research findings produced by the "
            "Research Specialist.\n\n"
            "Identify:\n"
            "1. Major insights\n"
            "2. Advantages\n"
            "3. Disadvantages\n"
            "4. Practical applications\n"
            "5. Challenges\n"
            "6. Future opportunities\n"
            "7. Overall conclusions\n\n"
            "Base the analysis on the research findings."
        ),
        expected_output=(
            "A structured analytical summary containing major "
            "insights, advantages, disadvantages, challenges, "
            "future opportunities and conclusions."
        ),
        agent=analyst,
        context=[research_task],
    )

    # --------------------------------------------------------
    # Task 3: Report Writing
    # --------------------------------------------------------

    report_task = Task(
        description=(
            "Create the final professional research report using "
            "the research findings and analytical summary.\n\n"
            "Use the following structure:\n\n"
            "# Title\n"
            "## Introduction\n"
            "## Key Findings\n"
            "## Analysis\n"
            "## Benefits\n"
            "## Challenges\n"
            "## Real-World Applications\n"
            "## Future Scope\n"
            "## Conclusion\n\n"
            "The report should be clear, concise, professional and "
            "easy to understand."
        ),
        expected_output=(
            "A complete professional research report containing "
            "all requested sections."
        ),
        agent=writer,
        context=[research_task, analysis_task],
    )

    # ========================================================
    # Create Crew
    # ========================================================

    crew = Crew(
        agents=[
            researcher,
            analyst,
            writer,
        ],
        tasks=[
            research_task,
            analysis_task,
            report_task,
        ],
        process=Process.sequential,
        verbose=True,
    )

    return crew


# ============================================================
# Run Research
# ============================================================

def run_research(topic: str) -> str:
    """
    Execute the research crew.

    Args:
        topic: Research topic.

    Returns:
        Final research report as a string.
    """

    crew = build_research_crew(topic)

    result = crew.kickoff()

    return str(result)


# ============================================================
# Save Report
# ============================================================

def save_report(report: str) -> None:
    """
    Save the final report to a text file.
    """

    REPORT_FILE.write_text(
        report,
        encoding="utf-8",
    )

    print(f"\nReport saved to: {REPORT_FILE}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    """Main application entry point."""

    print("=" * 70)
    print("AUTOMATED RESEARCH REPORT AGENT")
    print("=" * 70)

    print("\nLLM: Ollama")
    print(f"Model: llama3.2:3b")

    topic = input("\nEnter research topic: ").strip()

    if not topic:
        print("\nError: Research topic cannot be empty.")
        return

    print("\n" + "-" * 70)
    print(f"Research Topic: {topic}")
    print("-" * 70)

    print("\nStarting CrewAI research workflow...\n")

    try:
        report = run_research(topic)

        print("\n" + "=" * 70)
        print("FINAL RESEARCH REPORT")
        print("=" * 70)

        print(report)

        save_report(report)

        print("\n" + "=" * 70)
        print("RESEARCH COMPLETED SUCCESSFULLY")
        print("=" * 70)

    except Exception as error:
        print("\n" + "=" * 70)
        print("RESEARCH FAILED")
        print("=" * 70)

        print(f"\nError: {error}")

        print("\nTroubleshooting:")
        print("1. Make sure Ollama is running.")
        print("2. Make sure llama3.2:3b is installed.")
        print("3. Run: ollama list")
        print("4. Run: ollama run llama3.2:3b")


# ============================================================
# Program Entry
# ============================================================

if __name__ == "__main__":
    main()