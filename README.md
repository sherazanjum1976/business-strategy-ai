# Business Strategy AI

A Streamlit app where five CrewAI agents (powered by Groq `openai/gpt-oss-120b`) research,
analyze, and write a professional business strategy report.

## Agents
1. Market Research Analyst
2. Competitive Analyst
3. Customer Insights Analyst
   (1-3 run in parallel)
4. Business Strategist (SWOT, model, marketing, operations, finance, risks, roadmap)
5. Report Writer (final 15-section report)

## Important
- No live web research is performed. Output is based on your inputs and general model
  knowledge. Claims are labelled `[User-Provided]`, `[General Knowledge]`, or `[Assumption]`.
  Verify before making real decisions.
- Provide detailed inputs (budget, team size, constraints) for better reports.

## Deploy (Streamlit Cloud)
1. Push this repo to GitHub.
2. On share.streamlit.io create an app, select `app.py`, choose Python 3.12.
3. In Advanced settings -> Secrets add:
   ```toml
   GROQ_API_KEY = "your_groq_key_here"
   ```
4. Deploy.

Optional secret: `GROQ_MODEL = "openai/gpt-oss-20b"` to use a smaller/faster model.

## Troubleshooting
- **Rate limit (429)**: untick "Run research agents in parallel", wait a minute, retry,
  or set `GROQ_MODEL` to `openai/gpt-oss-20b`. Groq's free tier has tight token limits.
- **Build fails / dependency error**: set Python to 3.12 and reboot the app.
  If it still fails, check the build log and pin the versions it suggests in `requirements.txt`.
- **sqlite3 / chromadb error**: add `pysqlite3-binary` to `requirements.txt` and add this
  line at the very top of `app.py`:
  `__import__("pysqlite3"); import sys; sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")`
