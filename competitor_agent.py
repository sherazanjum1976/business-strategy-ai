"""Agent 2: Competitor Analysis."""
from crewai import Task
from agents import make_agent


def create_agent(llm):
    return make_agent(
        role="Competitive Analyst",
        goal="Map the competitive landscape and find realistic differentiation opportunities.",
        backstory="A strategy consultant skilled at positioning and competitive benchmarking.",
        llm=llm,
    )


def create_task(agent, callback=None, async_execution=True):
    return Task(
        name="competitor",
        description=(
            "Produce a Competitor Analysis for this business:\n\n"
            "{business_context}\n\n"
            "Cover: 1) Types of competitors (direct, indirect, substitutes), "
            "2) Competitor archetypes and their typical strengths/weaknesses "
            "(name specific real companies ONLY if the user provided them or they are "
            "very well known, and label them [General Knowledge] to verify), "
            "3) A comparison table across price, quality, reach, and differentiation, "
            "4) Gaps in the market, 5) Recommended differentiation angles."
        ),
        expected_output="A Markdown competitor analysis including a comparison table, claims labelled.",
        agent=agent,
        async_execution=async_execution,
        callback=callback,
    )
