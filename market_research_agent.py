"""Agent 1: Market & Industry Research."""
from crewai import Task
from agents import make_agent


def create_agent(llm):
    return make_agent(
        role="Market Research Analyst",
        goal="Analyze the industry and market conditions relevant to the business.",
        backstory="A veteran industry analyst who explains market dynamics clearly and honestly.",
        llm=llm,
    )


def create_task(agent, callback=None, async_execution=True):
    return Task(
        name="market",
        description=(
            "Produce a Market & Industry Analysis for this business:\n\n"
            "{business_context}\n\n"
            "Cover: 1) Industry overview, 2) Key trends, 3) Growth drivers and barriers, "
            "4) Regulatory/economic/technology factors for the stated location, "
            "5) Market opportunities, 6) Market threats. "
            "Do not invent numbers; describe size/growth qualitatively unless the user gave data."
        ),
        expected_output="A Markdown market analysis with the six sections above, claims labelled.",
        agent=agent,
        async_execution=async_execution,
        callback=callback,
    )
