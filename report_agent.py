"""Agent 3: Report Writer (writes the summary sections; code assembles the full report)."""
from crewai import Task
from agents import make_agent


def create_agent(llm):
    return make_agent(
        role="Report Writer",
        goal="Write the executive summary, overview, and conclusion of the strategy report.",
        backstory="A professional business writer who produces clear, executive-ready summaries.",
        llm=llm,
    )


def create_task(agent, context_tasks, callback=None):
    return Task(
        name="report",
        description=(
            "Using the research and strategy provided as context, write the framing sections "
            "of the final report for this business:\n\n{business_context}\n\n"
            "Write ONLY these three sections, using EXACTLY these Markdown headings:\n"
            "## 1. Executive Summary\n(120-170 words: key findings and top recommendations)\n"
            "## 2. Business/Problem Overview\n(80-120 words)\n"
            "## 15. Conclusion\n(70-100 words)\n\n"
            "Rules: use only information from the context, do not repeat other sections, "
            "no invented numbers."
        ),
        expected_output="Three Markdown sections numbered 1, 2, 15 with the exact headings above.",
        agent=agent,
        context=context_tasks,
        callback=callback,
    )
