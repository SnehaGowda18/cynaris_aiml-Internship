@"
# W9D3 Self-Review Checklist

## Multi-Agent Research Crew

- [x] Created three CrewAI agents.
- [x] Defined Researcher role, goal, and backstory.
- [x] Defined Writer role, goal, and backstory.
- [x] Defined Reviewer role, goal, and backstory.
- [x] Created sequential CrewAI tasks.
- [x] Integrated Ollama with Llama 3.2 3B.
- [x] Added Serper web search to the Researcher.
- [x] Generated research output.
- [x] Generated web research output.
- [x] Compared results before and after web search.
- [x] Added error handling.
- [x] Stored API credentials in `.env`.
- [x] Added `.env` to `.gitignore`.

## Output Evidence

- `research_output.txt` contains the research crew output.
- `web_research_output.txt` contains the web-search-based research output.
- `comparison.txt` documents the improvement from web search.
- `README.md` documents the project architecture, setup, agents, tasks, and viva answers.

## Git Workflow

- [x] Created feature branch `feat/aiml-W9-sneha`.
- [x] Created multiple descriptive commits.
- [x] Checked that `.env` is not tracked.
- [ ] Push final changes to remote.
- [ ] Create pull request.

## CIA Interactions

### CIA Interaction 1 — Code Review

**Prompt used:**

"Review my CrewAI multi-agent research pipeline. Check whether the Researcher, Writer, and Reviewer agents have clear roles, goals, backstories, and responsibilities. Also check whether the sequential task design is appropriate."

**Action taken:**

Reviewed the architecture and verified that the pipeline follows:

`Researcher → Writer → Reviewer`

The Researcher is responsible for web-based research, the Writer creates the report, and the Reviewer evaluates the report.

### CIA Interaction 2 — Code Quality Review

**Prompt used:**

"Review my W9D3 research crew code for code quality, error handling, security, and Git readiness. Check that the Serper API key is protected and that the output files are generated correctly."

**Action taken:**

Checked API-key handling through `.env`, confirmed `.env` is excluded from Git, verified output files, and checked that the application handles execution errors.

## Final Assessment

The W9D3 implementation successfully demonstrates a multi-agent research workflow using CrewAI, Ollama, and web search. The specialized agents separate research, writing, and quality review responsibilities.

## Future Improvements

With additional development time, I would add automated source validation, stronger citation handling, automated evaluation metrics, MLflow tracing, unit tests, and improved report formatting.
"@ | Set-Content self_review.md