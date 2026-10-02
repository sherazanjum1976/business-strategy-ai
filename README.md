# Business Strategy AI

A Streamlit app where three CrewAI agents (powered by Groq `openai/gpt-oss-120b`) research,
plan, and write a business strategy report.

## Agents (run in sequence)
1. **Research Analyst** - market, customers, competitors, opportunities and threats
2. **Business Strategist** - SWOT, business model, marketing, operations, finance, risks, recommendations, roadmap
3. **Report Writer** - executive summary, overview, conclusion (the app then assembles all 15 sections)

## Built for Groq's free plan
The free plan allows 30 requests/minute but only **8,000 tokens/minute** per model
(and 200,000 tokens/day). `config.py` contains a rate limiter that paces every request
so the limit is never exceeded, so a report takes about 2-4 minutes. One report uses roughly
12-15k tokens, so about 13+ reports per day fit in the daily quota.

## Important
- No live web research is performed. Output is based on your inputs and general model
  knowledge. Claims are labelled `[User-Provided]`, `[General Knowledge]`, or `[Assumption]`.
  Verify before making real decisions.
- Inputs are length-limited to keep requests within Groq's token limit.

## Deploy (Streamlit Cloud)
1. Push this repo to GitHub.
2. On share.streamlit.io create an app, select `app.py`, choose Python 3.12.
3. In Advanced settings -> Secrets add:
   ```toml
   GROQ_API_KEY = "your_groq_key_here"
   ```
4. Deploy.

Optional secrets:
- `GROQ_MODEL = "openai/gpt-oss-20b"` - separate daily quota, faster, slightly lower quality.
- `GROQ_TPM_LIMIT = "20000"` - if you upgrade your Groq plan and have a higher tokens/minute limit,
  set it here to remove the waiting between agents.

## Troubleshooting
- **Rate limit message**: wait 1-2 minutes and retry. Make sure nobody else is using the same Groq key.
- **Daily token limit**: try tomorrow or set `GROQ_MODEL` to `openai/gpt-oss-20b`.
- **`cache_breakpoint is unsupported`**: a known CrewAI/Groq bug; `config.py` already patches it.
- **Build fails**: set Python to 3.12 and reboot the app; check the build log.
