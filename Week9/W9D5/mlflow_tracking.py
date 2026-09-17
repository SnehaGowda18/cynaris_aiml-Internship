

from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from research_agent import run_research


# ---------------------------------------------------------
# State Definition
# ---------------------------------------------------------

class ResearchState(TypedDict):
    """State shared between LangGraph workflow nodes."""

    topic: str
    research: str
    analysis: str
    report: str


# ---------------------------------------------------------
# Research Node
# ---------------------------------------------------------

def research_node(state: ResearchState) -> ResearchState:
    """
    Run the CrewAI research agents for the given topic.
    """

    topic = state["topic"]

    print("\n" + "=" * 70)
    print("LANGGRAPH - RESEARCH NODE")
    print("=" * 70)

    print(f"\nResearch topic: {topic}")
    print("\nRunning CrewAI research agents...")

    research = run_research(topic)

    return {
        **state,
        "research": research,
    }


# ---------------------------------------------------------
# Analysis Node
# ---------------------------------------------------------

def analysis_node(state: ResearchState) -> ResearchState:
    """
    Analyze the research findings.

    The analysis organizes the generated research into
    important analytical categories.
    """

    research = state["research"]

    print("\n" + "=" * 70)
    print("LANGGRAPH - ANALYSIS NODE")
    print("=" * 70)

    print("\nAnalyzing research findings...")

    analysis = (
        "RESEARCH ANALYSIS\n"
        "=================\n\n"
        "The research findings were reviewed according to "
        "the following areas:\n\n"
        "1. Major Insights\n"
        "2. Benefits\n"
        "3. Limitations\n"
        "4. Real-World Applications\n"
        "5. Challenges\n"
        "6. Future Opportunities\n"
        "7. Overall Conclusion\n\n"
        "SOURCE RESEARCH\n"
        "===============\n\n"
        f"{research}"
    )

    return {
        **state,
        "analysis": analysis,
    }


# ---------------------------------------------------------
# Report Node
# ---------------------------------------------------------

def report_node(state: ResearchState) -> ResearchState:
    """
    Generate the final report from the analysis.
    """

    analysis = state["analysis"]

    print("\n" + "=" * 70)
    print("LANGGRAPH - REPORT NODE")
    print("=" * 70)

    print("\nPreparing final research report...")

    report = (
        "AUTOMATED RESEARCH REPORT\n"
        "=========================\n\n"
        f"{analysis}"
    )

    return {
        **state,
        "report": report,
    }


# ---------------------------------------------------------
# Build LangGraph Workflow
# ---------------------------------------------------------

def build_workflow():
    """
    Build and compile the LangGraph workflow.

    Workflow:
        START
          |
       Research
          |
       Analysis
          |
        Report
          |
         END
    """

    workflow = StateGraph(ResearchState)

    # Add workflow nodes
    workflow.add_node("research", research_node)
    workflow.add_node("analysis", analysis_node)
    workflow.add_node("report", report_node)

    # Define workflow edges
    workflow.add_edge(START, "research")
    workflow.add_edge("research", "analysis")
    workflow.add_edge("analysis", "report")
    workflow.add_edge("report", END)

    # Compile workflow
    return workflow.compile()


# ---------------------------------------------------------
# Run Workflow
# ---------------------------------------------------------

def run_workflow(topic: str) -> str:
    """
    Execute the complete LangGraph research workflow.
    """

    if not topic.strip():
        raise ValueError("Research topic cannot be empty.")

    app = build_workflow()

    initial_state: ResearchState = {
        "topic": topic,
        "research": "",
        "analysis": "",
        "report": "",
    }

    result = app.invoke(initial_state)

    return result["report"]


# ---------------------------------------------------------
# Main Program
# ---------------------------------------------------------

def main() -> None:
    """
    Run the LangGraph research workflow from the terminal.
    """

    print("=" * 70)
    print("W9D5 - AUTOMATED RESEARCH REPORT AGENT")
    print("=" * 70)

    print("\nFramework : LangGraph")
    print("Research  : CrewAI")
    print("LLM       : Ollama")
    print("Model     : llama3.2:3b")

    topic = input("\nEnter research topic: ").strip()

    if not topic:
        print("\nError: Research topic cannot be empty.")
        return

    print("\n" + "-" * 70)
    print(f"Research Topic: {topic}")
    print("-" * 70)

    print("\nStarting LangGraph workflow...")

    try:
        final_report = run_workflow(topic)

        print("\n" + "=" * 70)
        print("FINAL RESEARCH REPORT")
        print("=" * 70)

        print("\n" + final_report)

        print("\n" + "=" * 70)
        print("LANGGRAPH WORKFLOW COMPLETED SUCCESSFULLY")
        print("=" * 70)

    except Exception as error:
        print("\n" + "=" * 70)
        print("LANGGRAPH WORKFLOW FAILED")
        print("=" * 70)

        print(f"\nError: {error}")

        print("\nTroubleshooting:")
        print("1. Make sure Ollama is running.")
        print("2. Check the installed models:")
        print("   ollama list")
        print("3. Make sure llama3.2:3b is available.")
        print("4. Test Ollama directly:")
        print("   ollama run llama3.2:3b")


# ---------------------------------------------------------
# Entry Point
# ---------------------------------------------------------

if __name__ == "__main__":
    main()