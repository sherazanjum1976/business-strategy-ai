"""Assembles the 3-agent CrewAI crew, runs it, and builds the final report."""
import re

from crewai import Crew, Process

import research_agent as research_mod
import strategy_agent as strategy_mod
import report_agent as report_mod

AGENT_LABELS = {
    "research": "Research Analyst",
    "strategy": "Business Strategist",
    "report": "Report Writer",
}

SECTION_TITLES = {
    1: "Executive Summary", 2: "Business/Problem Overview",
    3: "Market & Industry Analysis", 4: "Target Customers", 5: "Competitor Analysis",
    6: "SWOT Analysis", 7: "Opportunities and Threats", 8: "Business Model",
    9: "Marketing Strategy", 10: "Operational Strategy",
    11: "Financial/Revenue Considerations", 12: "Risks and Mitigation",
    13: "Strategic Recommendations", 14: "Implementation Roadmap", 15: "Conclusion",
}

FIELD_LIMITS = {"name": 100, "description": 1200, "industry": 100,
                "target": 400, "location": 100, "goals": 500, "extra": 1000}

NOTE = (
    "> **Note on Data and Assumptions:** No live web research was performed. This report is "
    "based on the information you provided and general model knowledge. Claims are labelled "
    "[User-Provided], [General Knowledge], or [Assumption]. Verify key facts before making decisions."
)

HEADING_RE = re.compile(r"^[ \t]{0,3}#{1,4}[ \t]*(\d{1,2})[\.\):]?[ \t]*(.*)$", re.M)


def clean(text, limit):
    """Remove curly braces (CrewAI placeholders) and cap the length."""
    return (text or "").replace("{", "(").replace("}", ")").strip()[:limit]


def build_context(form):
    fields = [
        ("Business/Company/Idea", "name"), ("Description", "description"),
        ("Industry", "industry"), ("Target market/customers", "target"),
        ("Location/market", "location"), ("Business goals", "goals"),
        ("Additional requirements/context", "extra"),
    ]
    lines = ["[User-Provided business information]"]
    for label, key in fields:
        value = clean(form.get(key), FIELD_LIMITS[key])
        lines.append(f"- {label}: {value if value else 'Not provided'}")
    return "\n".join(lines)


def parse_sections(text):
    """Split agent output into {section_number: body} using numbered headings."""
    matches = [m for m in HEADING_RE.finditer(text or "") if 1 <= int(m.group(1)) <= 15]
    found = {}
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        found.setdefault(int(m.group(1)), text[m.end():end].strip())
    return found


def assemble_report(title, parts):
    """Combine agent outputs into the final 15-section report in the correct order."""
    sections, extras = {}, []
    for text in parts:
        parsed = parse_sections(text)
        if parsed:
            for num, body in parsed.items():
                sections.setdefault(num, body)
        elif text and text.strip():
            extras.append(text.strip())

    out = [f"# Business Strategy Report: {title}", "", NOTE, ""]
    for num in range(1, 16):
        body = sections.get(num) or "_This section was not generated. Please regenerate the report._"
        out += [f"## {num}. {SECTION_TITLES[num]}", "", body, ""]
    if extras:
        out += ["## Additional Notes", ""] + extras
    return "\n".join(out).strip() + "\n"


def run_crew(form, llms, on_task_done):
    """Run the 3 agents one after another. on_task_done(key) fires as each finishes."""

    def cb(key):
        return lambda _output: on_task_done(key)

    a_research = research_mod.create_agent(llms["research"])
    a_strategy = strategy_mod.create_agent(llms["strategy"])
    a_report = report_mod.create_agent(llms["report"])

    t_research = research_mod.create_task(a_research, cb("research"))
    t_strategy = strategy_mod.create_task(a_strategy, [t_research], cb("strategy"))
    t_report = report_mod.create_task(a_report, [t_research, t_strategy], cb("report"))

    crew = Crew(
        agents=[a_research, a_strategy, a_report],
        tasks=[t_research, t_strategy, t_report],
        process=Process.sequential,
        verbose=False,
    )
    crew.kickoff(inputs={"business_context": build_context(form)})

    def raw(task):
        out = getattr(task, "output", None)
        return (getattr(out, "raw", "") or "") if out else ""

    outputs = {
        AGENT_LABELS["research"]: raw(t_research),
        AGENT_LABELS["strategy"]: raw(t_strategy),
        AGENT_LABELS["report"]: raw(t_report),
    }
    if not any(v.strip() for v in outputs.values()):
        raise RuntimeError("The agents returned empty output. Please try again.")

    title = clean(form.get("name"), 100) or "Your Business"
    report = assemble_report(title, [outputs[AGENT_LABELS["report"]],
                                     outputs[AGENT_LABELS["research"]],
                                     outputs[AGENT_LABELS["strategy"]]])
    return {"report": report, "sections": outputs}
