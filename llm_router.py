"""
LLM router: one function, `call_llm`, that handles every LLM call in the project.
"""

import hashlib
import json
import os
import time
from collections import deque

from dotenv import load_dotenv
import diskcache
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

import google.generativeai as genai
from groq import Groq, RateLimitError as GroqRateLimitError
from openai import OpenAI

from config import (
    MODEL_ROUTING,
    RATE_LIMITS_RPM,
    MAX_RETRIES,
    RETRY_INITIAL_WAIT,
    RETRY_MAX_WAIT,
    CACHE_ENABLED,
    CACHE_DIR,
    CACHE_SIZE_LIMIT_GB,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
)

# ============================================================================
# SETUP
# ============================================================================
load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not found in .env")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found in .env")

genai.configure(api_key=GOOGLE_API_KEY)
_groq_client = Groq(api_key=GROQ_API_KEY)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY not found in .env")

_openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# ============================================================================
# DISK CACHE
# ============================================================================
_cache = diskcache.Cache(
    str(CACHE_DIR),
    size_limit=CACHE_SIZE_LIMIT_GB * 1024 * 1024 * 1024,
)


def _make_cache_key(provider, model, prompt, temperature, max_tokens):
    payload = json.dumps(
        {"provider": provider, "model": model, "prompt": prompt,
         "temp": temperature, "max_tokens": max_tokens},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


# ============================================================================
# RATE LIMITER
# ============================================================================
_call_timestamps = {
    "gemini": deque(),
    "groq": deque(),
    "openrouter": deque(),
}


def _wait_for_rate_limit(provider):
    rpm_limit = RATE_LIMITS_RPM[provider]
    timestamps = _call_timestamps[provider]
    now = time.time()

    while timestamps and now - timestamps[0] > 60:
        timestamps.popleft()

    if len(timestamps) >= rpm_limit:
        wait_seconds = 60 - (now - timestamps[0]) + 0.1
        if wait_seconds > 0:
            print(f"  [rate-limit] {provider}: sleeping {wait_seconds:.1f}s")
            time.sleep(wait_seconds)
            return _wait_for_rate_limit(provider)

    timestamps.append(time.time())


# ============================================================================
# PROVIDER FUNCTIONS
# ============================================================================
class TransientError(Exception):
    pass


def _call_gemini(model, prompt, temperature, max_tokens):
    try:
        gemini_model = genai.GenerativeModel(model)
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        return response.text
    except Exception as e:
        error_str = str(e).lower()
        if any(s in error_str for s in ["429", "quota", "rate", "resource_exhausted"]):
            raise TransientError(f"Gemini rate limit: {e}") from e
        if any(s in error_str for s in ["500", "503", "timeout", "unavailable"]):
            raise TransientError(f"Gemini transient error: {e}") from e
        raise


def _call_groq(model, prompt, temperature, max_tokens):
    try:
        completion = _groq_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return completion.choices[0].message.content
    except GroqRateLimitError as e:
        raise TransientError(f"Groq rate limit: {e}") from e
    except Exception as e:
        error_str = str(e).lower()
        if any(s in error_str for s in ["500", "503", "timeout", "unavailable"]):
            raise TransientError(f"Groq transient error: {e}") from e
        raise
def _call_openrouter(model, prompt, temperature, max_tokens):
    """Make a single OpenRouter API call. OpenRouter uses the OpenAI format."""
    try:
        completion = _openrouter_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return completion.choices[0].message.content
    except Exception as e:
        error_str = str(e).lower()
        if any(s in error_str for s in ["429", "rate", "quota"]):
            raise TransientError(f"OpenRouter rate limit: {e}") from e
        if any(s in error_str for s in ["500", "502", "503", "timeout", "unavailable"]):
            raise TransientError(f"OpenRouter transient error: {e}") from e
        raise

# ============================================================================
# RETRY DECORATOR
# ============================================================================
@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=RETRY_INITIAL_WAIT, max=RETRY_MAX_WAIT),
    retry=retry_if_exception_type(TransientError),
    reraise=True,
)
def _call_with_retry(provider, model, prompt, temperature, max_tokens):
    _wait_for_rate_limit(provider)
    if provider == "gemini":
        return _call_gemini(model, prompt, temperature, max_tokens)
    elif provider == "groq":
        return _call_groq(model, prompt, temperature, max_tokens)
    elif provider == "openrouter":
        return _call_openrouter(model, prompt, temperature, max_tokens)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ============================================================================
# PUBLIC API
# ============================================================================
def call_llm(prompt, role, temperature=DEFAULT_TEMPERATURE,
             max_tokens=DEFAULT_MAX_TOKENS, use_cache=True):
    if role not in MODEL_ROUTING:
        raise ValueError(
            f"Unknown role: {role!r}. "
            f"Valid roles: {sorted(MODEL_ROUTING.keys())}"
        )

    provider, model = MODEL_ROUTING[role]
    cache_key = _make_cache_key(provider, model, prompt, temperature, max_tokens)

    if use_cache and CACHE_ENABLED:
        cached = _cache.get(cache_key)
        if cached is not None:
            return cached

    response = _call_with_retry(provider, model, prompt, temperature, max_tokens)

    if use_cache and CACHE_ENABLED:
        _cache.set(cache_key, response)

    return response


def cache_stats():
    return {
        "entries": len(_cache),
        "size_bytes": _cache.volume(),
        "size_mb": round(_cache.volume() / 1024 / 1024, 2),
    }


def clear_cache():
    _cache.clear()