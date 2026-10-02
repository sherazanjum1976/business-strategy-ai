"""Configuration: API key, model, LLM factory, Groq compatibility patch, and rate limiter."""
import asyncio
import os
import threading
import time

# Must be set before crewai is imported anywhere.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

import streamlit as st

DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_TPM_LIMIT = 8000  # Groq free plan: tokens per minute for gpt-oss models
RPM_BUDGET = 25           # Groq free plan allows 30 requests/min; stay below


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


def _tpm_budget():
    try:
        limit = int(_get_setting("GROQ_TPM_LIMIT") or DEFAULT_TPM_LIMIT)
    except (TypeError, ValueError):
        limit = DEFAULT_TPM_LIMIT
    return int(limit * 0.9)  # keep a 10% safety margin


# ---------------------------------------------------------------------------
# Rate limiter: sliding 60-second window of tokens. Before every LLM call we
# wait until the request fits inside the per-minute token budget, so the free
# plan limit is never exceeded. Shared by all users of the running app.
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_window = []  # list of [timestamp, tokens]


def _estimate_tokens(kwargs):
    messages = kwargs.get("messages") or []
    chars = 0
    for m in messages:
        chars += len(str(m.get("content", ""))) if isinstance(m, dict) else len(str(m))
    prompt_tokens = chars / 3.5 + 50
    out_tokens = kwargs.get("max_completion_tokens") or kwargs.get("max_tokens") or 1500
    return int(prompt_tokens + out_tokens)


def _try_reserve(est):
    """Return (entry, 0) if allowed now, else (None, seconds_to_wait)."""
    with _lock:
        now = time.time()
        _window[:] = [e for e in _window if now - e[0] < 60]
        used = sum(e[1] for e in _window)
        if not _window or (used + est <= _tpm_budget() and len(_window) < RPM_BUDGET):
            entry = [now, est]
            _window.append(entry)
            return entry, 0.0
        wait = 60 - (now - _window[0][0]) + 0.5
        return None, max(wait, 1.0)


def _release(entry):
    with _lock:
        if entry in _window:
            _window.remove(entry)


def _record_actual(entry, response):
    """Replace our estimate with the real token count once known."""
    try:
        total = getattr(getattr(response, "usage", None), "total_tokens", None)
        if total:
            with _lock:
                entry[1] = int(total)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Compatibility patch for litellm (used by CrewAI to talk to Groq):
#  1. Strip CrewAI's internal "cache_breakpoint" key (Groq rejects it).
#  2. Use low reasoning effort for gpt-oss models (fewer hidden tokens).
#  3. Apply the rate limiter and retry politely on 429 errors.
# ---------------------------------------------------------------------------
_UNSUPPORTED_KEYS = ("cache_breakpoint",)


def _strip_unsupported_keys(messages):
    if not isinstance(messages, list):
        return messages
    cleaned = []
    for msg in messages:
        if isinstance(msg, dict) and any(k in msg for k in _UNSUPPORTED_KEYS):
            msg = {k: v for k, v in msg.items() if k not in _UNSUPPORTED_KEYS}
        cleaned.append(msg)
    return cleaned


def _prepare(kwargs):
    if "messages" in kwargs:
        kwargs["messages"] = _strip_unsupported_keys(kwargs["messages"])
    model = str(kwargs.get("model", ""))
    if model.startswith("groq/") and "gpt-oss" in model:
        kwargs.setdefault("reasoning_effort", "low")


def _is_rate_limit(exc):
    low = str(exc).lower()
    return "429" in low or "rate limit" in low or "rate_limit" in low


def _patch_litellm():
    try:
        import litellm
    except Exception:
        return
    if getattr(litellm, "_bsai_patched", False):
        return

    litellm.drop_params = True  # ignore params a provider doesn't support
    original_completion = litellm.completion
    original_acompletion = getattr(litellm, "acompletion", None)

    def completion(*args, **kwargs):
        _prepare(kwargs)
        est = _estimate_tokens(kwargs)
        for attempt in range(4):
            entry, wait = _try_reserve(est)
            while entry is None:
                time.sleep(min(wait, 10))
                entry, wait = _try_reserve(est)
            try:
                response = original_completion(*args, **kwargs)
            except Exception as exc:
                _release(entry)
                if _is_rate_limit(exc) and attempt < 3:
                    time.sleep(20 + 15 * attempt)
                    continue
                raise
            _record_actual(entry, response)
            return response

    litellm.completion = completion

    if original_acompletion is not None:
        async def acompletion(*args, **kwargs):
            _prepare(kwargs)
            est = _estimate_tokens(kwargs)
            for attempt in range(4):
                entry, wait = _try_reserve(est)
                while entry is None:
                    await asyncio.sleep(min(wait, 10))
                    entry, wait = _try_reserve(est)
                try:
                    response = await original_acompletion(*args, **kwargs)
                except Exception as exc:
                    _release(entry)
                    if _is_rate_limit(exc) and attempt < 3:
                        await asyncio.sleep(20 + 15 * attempt)
                        continue
                    raise
                _record_actual(entry, response)
                return response

        litellm.acompletion = acompletion

    litellm._bsai_patched = True


_patch_litellm()  # must run BEFORE crewai is imported below

from crewai import LLM  # noqa: E402


def build_llm(max_tokens=3500):
    """Create a CrewAI LLM that talks to Groq (via LiteLLM)."""
    api_key = get_api_key()
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing.")
    return LLM(
        model=f"groq/{get_model_name()}",
        api_key=api_key,
        temperature=0.4,
        max_completion_tokens=max_tokens,
    )
