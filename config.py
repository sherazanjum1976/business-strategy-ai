"""Configuration: API key, model name, and LLM factory."""
import os

# Must be set before crewai is imported anywhere.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

import streamlit as st
from crewai import LLM

DEFAULT_MODEL = "openai/gpt-oss-120b"


def _get_setting(name):
    """Read from Streamlit Secrets first, then environment variables."""
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return value or os.environ.get(name)


def get_api_key():
    return _get_setting("GROQ_API_KEY")


def get_model_name():
    return _get_setting("GROQ_MODEL") or DEFAULT_MODEL


def build_llm(max_tokens=6000):
    """Create a CrewAI LLM that talks to Groq (via LiteLLM)."""
    api_key = get_api_key()
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing.")
    return LLM(
        model=f"groq/{get_model_name()}",
        api_key=api_key,
        temperature=0.4,
        max_completion_tokens=max_tokens,
        num_retries=4,
    )
