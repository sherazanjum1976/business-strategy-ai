"""Shared agent factory and grounding rules used by all agents."""
from crewai import Agent

GROUNDING_RULES = (
    "You have NO internet access. Never invent statistics, market sizes, company "
    "financials, quotes, or sources. Label important claims as one of: "
    "[User-Provided], [General Knowledge] (broad, unverified), or [Assumption]. "
    "Be concise: short bullet points, no filler, no preamble. Write in Markdown."
)


def make_agent(role, goal, backstory, llm):
    return Agent(
        role=role,
        goal=goal,
        backstory=f"{backstory} {GROUNDING_RULES}",
        llm=llm,
        allow_delegation=False,
        verbose=False,
        max_iter=3,
    )
