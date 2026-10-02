"""Agent 1: Research Analyst (market + customers + competitors + opportunities/threats)."""
from crewai import Task
from agents import make_agent


def create_agent(llm):
    return make_agent(
        role="Research Analyst",
        goal="Analyze the market, customers, and competition for the business.",
        backstory="A seasoned analyst who covers industry, customer, and competitor research clearly and honestly.",
        llm=llm,
    )


def create_task(agent, callback=None):
    return Task(
        name="research",
        description=(
            "Analyze this business:\n\n{business_context}\n\n"
            "Write ONLY these four sections, using EXACTLY these Markdown headings:\n"
            "## 3. Market & Industry Analysis\n"
            "(industry overview, 3-4 key trends, growth drivers and barriers, location-specific factors)\n"
            "## 4. Target Customers\n"
            "(2-3 prioritized segments, needs and pain points, best channels, core value proposition)\n"
            "## 5. Competitor Analysis\n"
            "(competitor types, typical strengths/weaknesses, market gaps, differentiation angles; "
            "name real companies ONLY if the user gave them or they are very well known, "
            "and label them [General Knowledge])\n"
            "## 7. Opportunities and Threats\n"
            "(3-4 bullets for each)\n\n"
            "Rules: 650-850 words in total, bullet points, no invented numbers or statistics."
        ),
        expected_output="Four Markdown sections numbered 3, 4, 5, 7 with the exact headings above.",
        agent=agent,
        callback=callback,
    )
