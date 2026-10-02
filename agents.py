"""Shared agent factory and grounding rules used by all agents."""
from crewai import Agent

GROUNDING_RULES = (
    "You have NO internet access. Never invent statistics, market sizes, company "
    "financials, quotes, or sources. Label every claim as one of: "
    "[User-Provided], [General Knowledge] (broad, unverified, should be checked), "
    "or [Assumption]. Use qualitative reasoning when data is unavailable. "
    "Write in clear Markdown."
)


def make_agent(role, goal, backstory, llm):
    return Agent(
        role=role,
        goal=goal,
        backstory=f"{backstory} {GROUNDING_RULES}",
        llm=llm,
        allow_delegation=False,
        verbose=False,
        max_iter=5,
    )
