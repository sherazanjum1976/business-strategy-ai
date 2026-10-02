"""Configuration: API key, model name, LLM factory, and a Groq compatibility patch."""
import os

# Must be set before crewai is imported anywhere.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

import streamlit as st

# ---------------------------------------------------------------------------
# Compatibility patch: some CrewAI releases add an internal "cache_breakpoint"
# key to messages (meant for Anthropic only). Groq rejects it with:
#   "property 'cache_breakpoint' is unsupported"
# We strip that key from every message right before LiteLLM sends the request.
# Works regardless of which CrewAI version gets installed.
# ---------------------------------------------------------------------------
_UNSUPPORTED_MESSAGE_KEYS = ("cache_breakpoint",)


def _strip_unsupported_keys(messages):
    if not isinstance(messages, list):
        return messages
    cleaned = []
    for msg in messages:
        if isinstance(msg, dict) and any(k in msg for k in _UNSUPPORTED_MESSAGE_KEYS):
            msg = {k: v for k, v in msg.items() if k not in _UNSUPPORTED_MESSAGE_KEYS}
        cleaned.append(msg)
    return cleaned


def _patch_litellm():
    try:
        import litellm
    except Exception:
        return
    if getattr(litellm, "_bsai_patched", False):
        return

    original_completion = litellm.completion
    original_acompletion = getattr(litellm, "acompletion", None)

    def completion(*args, **kwargs):
        if "messages" in kwargs:
            kwargs["messages"] = _strip_unsupported_keys(kwargs["messages"])
        return original_completion(*args, **kwargs)

    litellm.completion = completion

    if original_acompletion is not None:
        async def acompletion(*args, **kwargs):
            if "messages" in kwargs:
                kwargs["messages"] = _strip_unsupported_keys(kwargs["messages"])
            return await original_acompletion(*args, **kwargs)

        litellm.acompletion = acompletion

    litellm._bsai_patched = True


_patch_litellm()  # must run BEFORE crewai is imported below

from crewai import LLM  # noqa: E402

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
