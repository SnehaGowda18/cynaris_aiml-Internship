

from pathlib import Path

from research_agent import build_research_crew
from langgraph_workflow import build_workflow


# ---------------------------------------------------------
# Test 1: CrewAI Crew Creation
# ---------------------------------------------------------

def test_research_crew_creation():
    """Verify that the CrewAI research crew is created."""

    crew = build_research_crew("Artificial Intelligence")

    assert crew is not None
    assert len(crew.agents) == 3
    assert len(crew.tasks) == 3


# ---------------------------------------------------------
# Test 2: Agent Roles
# ---------------------------------------------------------

def test_agent_roles():
    """Verify the three expected agent roles."""

    crew = build_research_crew("Artificial Intelligence")

    roles = [agent.role for agent in crew.agents]

    assert "Research Specialist" in roles
    assert "Research Analyst" in roles
    assert "Report Writer" in roles


# ---------------------------------------------------------
# Test 3: LangGraph Workflow Creation
# ---------------------------------------------------------

def test_langgraph_workflow_creation():
    """Verify that the LangGraph workflow compiles."""

    workflow = build_workflow()

    assert workflow is not None


# ---------------------------------------------------------
# Test 4: Generated Report File
# ---------------------------------------------------------

def test_report_file_exists():
    """Verify that the generated research report exists."""

    report_file = Path("research_report.txt")

    assert report_file.exists()
    assert report_file.stat().st_size > 0


# ---------------------------------------------------------
# Test 5: Report Content
# ---------------------------------------------------------

def test_report_content():
    """Verify that the report contains meaningful content."""

    report_file = Path("research_report.txt")

    report = report_file.read_text(encoding="utf-8")

    assert len(report) > 100
    assert len(report.split()) > 20


# ---------------------------------------------------------
# Test 6: Empty Topic Validation
# ---------------------------------------------------------

def test_empty_topic_validation():
    """Verify that empty topics are rejected."""

    from langgraph_workflow import run_workflow

    try:
        run_workflow("")
        assert False, "Empty topic should raise ValueError"
    except ValueError:
        assert True