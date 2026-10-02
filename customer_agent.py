"""Agent 3: Customer / Target Market Analysis."""
from crewai import Task
from agents import make_agent


def create_agent(llm):
    return make_agent(
        role="Customer Insights Analyst",
        goal="Define who the customers are, what they need, and how to reach them.",
        backstory="A customer-research specialist who builds practical segments and personas.",
        llm=llm,
    )


def create_task(agent, callback=None, async_execution=True):
    return Task(
        name="customer",
        description=(
            "Produce a Target Customer Analysis for this business:\n\n"
            "{business_context}\n\n"
            "Cover: 1) Customer segments (prioritized), 2) Two or three personas "
            "(clearly marked as hypothetical), 3) Needs and pain points, "
            "4) Buying behavior and decision factors, 5) Best channels to reach them, "
            "6) Core value proposition, 7) Suggested ways to validate these assumptions."
        ),
        expected_output="A Markdown customer analysis with the seven sections above, claims labelled.",
        agent=agent,
        async_execution=async_execution,
        callback=callback,
    )
