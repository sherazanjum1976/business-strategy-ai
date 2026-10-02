"""Agent 5: Report Writer / Editor (depends on all previous tasks)."""
from crewai import Task
from agents import make_agent


def create_agent(llm):
    return make_agent(
        role="Report Writer and Editor",
        goal="Merge all analyses into one polished, consistent business strategy report.",
        backstory="A professional business writer who produces clear executive-ready reports.",
        llm=llm,
    )


def create_task(agent, context_tasks, callback=None):
    return Task(
        name="report",
        description=(
            "Write the FINAL Business Strategy Report for:\n\n"
            "{business_context}\n\n"
            "Use ONLY the content from the context analyses; do not add new facts. "
            "Remove duplication and fix inconsistencies. Use these exact numbered "
            "Markdown H2 sections in this order:\n"
            "1. Executive Summary\n2. Business/Problem Overview\n3. Market & Industry Analysis\n"
            "4. Target Customers\n5. Competitor Analysis\n6. SWOT Analysis\n"
            "7. Opportunities and Threats\n8. Business Model\n9. Marketing Strategy\n"
            "10. Operational Strategy\n11. Financial/Revenue Considerations\n"
            "12. Risks and Mitigation\n13. Strategic Recommendations\n"
            "14. Implementation Roadmap\n15. Conclusion\n\n"
            "Start with a short title and a 'Note on Data and Assumptions' paragraph "
            "explaining that no live research was performed and that claims are labelled "
            "[User-Provided], [General Knowledge], or [Assumption]. Keep those labels in the text."
        ),
        expected_output="A complete Markdown report with all 15 numbered sections.",
        agent=agent,
        context=context_tasks,
        callback=callback,
    )
