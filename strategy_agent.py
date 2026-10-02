"""Agent 4: Business Strategy (depends on agents 1-3)."""
from crewai import Task
from agents import make_agent


def create_agent(llm):
    return make_agent(
        role="Business Strategist",
        goal="Turn the research into a coherent, actionable business strategy.",
        backstory="A senior strategy partner who builds realistic, prioritized plans.",
        llm=llm,
    )


def create_task(agent, context_tasks, callback=None):
    return Task(
        name="strategy",
        description=(
            "Using the market, competitor, and customer analyses provided as context, "
            "build the strategy for this business:\n\n"
            "{business_context}\n\n"
            "Cover: 1) SWOT analysis, 2) Business model (value proposition, revenue streams, "
            "key resources), 3) Marketing strategy, 4) Operational strategy, "
            "5) Financial and revenue considerations (revenue streams, cost drivers, "
            "pricing logic, what to model - NO invented figures; mark any numbers as "
            "[Assumption]), 6) Risks and mitigation (table), 7) Strategic recommendations "
            "(prioritized), 8) Implementation roadmap in phases: 0-3, 3-6, 6-12, 12-24 months."
        ),
        expected_output="A Markdown strategy document with the eight sections above, claims labelled.",
        agent=agent,
        context=context_tasks,
        callback=callback,
    )
