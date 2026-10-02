"""Streamlit UI for the Multi-Agent Business Strategy Report System."""
import config  # noqa: F401  (must be imported before crewai)

import queue
import threading
import time

import streamlit as st

from config import build_llm, get_api_key, get_model_name
from crew import AGENT_LABELS, run_crew

ORDER = ["research", "strategy", "report"]

st.set_page_config(page_title="Business Strategy AI", page_icon="📊", layout="wide")


def friendly_error(exc):
    msg = str(exc)
    low = msg.lower()
    if "tokens per day" in low or "tpd" in low:
        return ("Groq's daily token limit is used up. Try again tomorrow, or set "
                "GROQ_MODEL = \"openai/gpt-oss-20b\" in Streamlit Secrets (it has its own daily quota).")
    if "429" in low or "rate limit" in low or "rate_limit" in low:
        return "Groq rate limit reached even after waiting. Please wait 1-2 minutes and try again."
    if "413" in low or "request too large" in low:
        return "The request was too large for Groq's limit. Shorten your inputs and try again."
    if "401" in low or "invalid api key" in low or "authentication" in low:
        return "Groq rejected the API key. Check GROQ_API_KEY in Streamlit Secrets."
    if "model" in low and ("not found" in low or "decommissioned" in low):
        return f"The model '{get_model_name()}' is unavailable. Set GROQ_MODEL in Secrets to another Groq model."
    return f"Generation failed: {msg[:400]}"


def render_status(done):
    lines = []
    for i, key in enumerate(ORDER):
        if key in done:
            state = "✅ Done"
        elif all(k in done for k in ORDER[:i]):
            state = "🔄 Working..."
        else:
            state = "⏳ Waiting"
        lines.append(f"**{i + 1}. {AGENT_LABELS[key]}** — {state}")
    return "\n\n".join(lines)


def validate(form):
    if not form["name"].strip():
        return "Please enter a business/company/idea name."
    if len(form["description"].strip()) < 20:
        return "Please enter a business description (at least 20 characters)."
    return None


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("About")
    st.caption(f"Model: `{get_model_name()}`")
    st.caption("API key: " + ("✅ found" if get_api_key() else "❌ missing"))
    st.markdown("---")
    st.caption("3 agents run one after another. Requests are automatically paced to stay "
               "within Groq's free-plan limit (8,000 tokens/minute), so a report takes "
               "about 2-4 minutes.")
    st.caption("No live web research is performed. Claims are labelled as user-provided, "
               "general knowledge, or assumptions.")

# ---------------- Main form ----------------
st.title("📊 Multi-Agent Business Strategy Report")
st.write("Three AI agents research, plan, and write a strategy report for your business.")

c1, c2 = st.columns(2)
with c1:
    name = st.text_input("Business / company / idea name *", max_chars=100)
    industry = st.text_input("Industry", max_chars=100)
    location = st.text_input("Location / market", max_chars=100)
with c2:
    target = st.text_area("Target market / customers", height=108, max_chars=400)
    goals = st.text_area("Business goals", height=108, max_chars=500)
description = st.text_area("Business description *", height=130, max_chars=1200)
extra = st.text_area("Additional requirements / context (budget, team size, constraints...)",
                     height=100, max_chars=1000)

generate = st.button("Generate Business Strategy Report", type="primary")

# ---------------- Run ----------------
if generate:
    form = {"name": name, "description": description, "industry": industry,
            "target": target, "location": location, "goals": goals, "extra": extra}
    error = validate(form)
    if error:
        st.error(error)
    elif not get_api_key():
        st.error("GROQ_API_KEY not found. Add it under Streamlit Cloud → App settings → Secrets.")
    else:
        try:
            llms = {"research": build_llm(3500), "strategy": build_llm(3600),
                    "report": build_llm(2000)}
        except Exception as e:  # noqa: BLE001
            st.error(friendly_error(e))
            st.stop()

        events = queue.Queue()

        def worker():
            try:
                result = run_crew(form, llms, lambda key: events.put(("done", key)))
                events.put(("result", result))
            except Exception as exc:  # noqa: BLE001
                events.put(("error", exc))

        threading.Thread(target=worker, daemon=True).start()

        done, outcome, started = set(), None, time.time()
        with st.status("Agents are working...", expanded=True) as status:
            box = st.empty()
            while outcome is None:
                try:
                    while True:
                        kind, payload = events.get_nowait()
                        if kind == "done":
                            done.add(payload)
                        else:
                            outcome = (kind, payload)
                except queue.Empty:
                    pass
                elapsed = int(time.time() - started)
                box.markdown(render_status(done) +
                             f"\n\n_Elapsed: {elapsed}s — pausing between agents is normal "
                             f"(Groq free-plan rate limit)._")
                if outcome is None:
                    time.sleep(1)

            if outcome[0] == "result":
                status.update(label="Report ready", state="complete", expanded=False)
                st.session_state["result"] = outcome[1]
                st.session_state["title"] = name.strip()
            else:
                status.update(label="Generation failed", state="error")
                st.session_state.pop("result", None)
                st.error(friendly_error(outcome[1]))

# ---------------- Display result ----------------
result = st.session_state.get("result")
if result:
    st.markdown("---")
    safe = "".join(ch if ch.isalnum() else "_" for ch in st.session_state.get("title", "report"))[:40]
    d1, d2, _ = st.columns([1, 1, 4])
    d1.download_button("⬇️ Download .md", result["report"], file_name=f"{safe}_strategy.md",
                       mime="text/markdown")
    d2.download_button("⬇️ Download .txt", result["report"], file_name=f"{safe}_strategy.txt",
                       mime="text/plain")
    st.markdown(result["report"])

    with st.expander("View individual agent outputs"):
        for label, text in result["sections"].items():
            st.subheader(label)
            st.markdown(text or "_No output captured._")
