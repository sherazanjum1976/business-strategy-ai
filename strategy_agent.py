"""Agent 2: Business Strategist (depends on the research)."""
from crewai import Task
from agents import make_agent


def create_agent(llm):
    return make_agent(
        role="Business Strategist",
        goal="Turn the research into a realistic, actionable business strategy.",
        backstory="A senior strategy partner who builds practical, prioritized plans.",
        llm=llm,
    )


def create_task(agent, context_tasks, callback=None):
    return Task(
        name="strategy",
        description=(
            "Using the research provided as context, build the strategy for this business:\n\n"
            "{business_context}\n\n"
            "Write ONLY these eight sections, using EXACTLY these Markdown headings:\n"
            "## 6. SWOT Analysis\n(2-4 bullets per quadrant)\n"
            "## 8. Business Model\n(value proposition, revenue streams, key resources)\n"
            "## 9. Marketing Strategy\n(positioning, channels, 3-4 key tactics)\n"
            "## 10. Operational Strategy\n(key processes, resources, partners, tools)\n"
            "## 11. Financial/Revenue Considerations\n"
            "(revenue streams, main cost drivers, pricing logic, what to model; "
            "NO invented figures - mark any number as [Assumption])\n"
            "## 12. Risks and Mitigation\n(4-5 risks, each with a mitigation)\n"
            "## 13. Strategic Recommendations\n(4-5 prioritized recommendations)\n"
            "## 14. Implementation Roadmap\n(phases: 0-3, 3-6, 6-12, 12-24 months)\n\n"
            "Rules: 800-1000 words in total, bullet points, concise."
        ),
        expected_output="Eight Markdown sections numbered 6, 8-14 with the exact headings above.",
        agent=agent,
        context=context_tasks,
        callback=callback,
    )
