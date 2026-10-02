"""Assembles the CrewAI crew and runs it."""
from crewai import Crew, Process

import market_research_agent as market_mod
import competitor_agent as competitor_mod
import customer_agent as customer_mod
import strategy_agent as strategy_mod
import report_agent as report_mod

AGENT_LABELS = {
    "market": "Market Research Analyst",
    "competitor": "Competitive Analyst",
    "customer": "Customer Insights Analyst",
    "strategy": "Business Strategist",
    "report": "Report Writer",
}


def clean(text):
    """Remove curly braces so user text can't break CrewAI placeholders."""
    return (text or "").replace("{", "(").replace("}", ")").strip()


def build_context(form):
    """Turn form inputs into one text block shared by all tasks."""
    fields = [
        ("Business/Company/Idea", "name"),
        ("Description", "description"),
        ("Industry", "industry"),
        ("Target market/customers", "target"),
        ("Location/market", "location"),
        ("Business goals", "goals"),
        ("Additional requirements/context", "extra"),
    ]
    lines = ["[User-Provided business information]"]
    for label, key in fields:
        value = clean(form.get(key))
        lines.append(f"- {label}: {value if value else 'Not provided'}")
    return "\n".join(lines)


def run_crew(form, llm, writer_llm, parallel, on_task_done):
    """Run all agents. on_task_done(key) is called as each task finishes."""

    def cb(key):
        return lambda _output: on_task_done(key)

    market_agent = market_mod.create_agent(llm)
    competitor_agent = competitor_mod.create_agent(llm)
    customer_agent = customer_mod.create_agent(llm)
    strategy_agent = strategy_mod.create_agent(llm)
    report_agent = report_mod.create_agent(writer_llm)

    t_market = market_mod.create_task(market_agent, cb("market"), parallel)
    t_competitor = competitor_mod.create_task(competitor_agent, cb("competitor"), parallel)
    t_customer = customer_mod.create_task(customer_agent, cb("customer"), parallel)
    research = [t_market, t_competitor, t_customer]
    t_strategy = strategy_mod.create_task(strategy_agent, research, cb("strategy"))
    t_report = report_mod.create_task(report_agent, research + [t_strategy], cb("report"))

    crew = Crew(
        agents=[market_agent, competitor_agent, customer_agent, strategy_agent, report_agent],
        tasks=[t_market, t_competitor, t_customer, t_strategy, t_report],
        process=Process.sequential,
        verbose=False,
    )
    result = crew.kickoff(inputs={"business_context": build_context(form)})

    report = (getattr(result, "raw", None) or str(result or "")).strip()
    if not report:
        raise RuntimeError("The agents returned an empty report. Please try again.")

    sections = {}
    for key, task in [("market", t_market), ("competitor", t_competitor),
                      ("customer", t_customer), ("strategy", t_strategy)]:
        out = getattr(task, "output", None)
        sections[AGENT_LABELS[key]] = getattr(out, "raw", "") if out else ""
    return {"report": report, "sections": sections}
