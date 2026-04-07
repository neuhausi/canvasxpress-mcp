#!/usr/bin/env python3
"""
llm_providers.py
================
Unified LLM provider abstraction for the CanvasXpress MCP server.

Supports six providers, selected via the LLM_PROVIDER environment variable:

  anthropic  — Direct Anthropic API (default)
  bedrock    — Anthropic models via Amazon Bedrock
  ollama     — Locally hosted models via Ollama
  openai     — Direct OpenAI API (api.openai.com)
  openai_corporate — OpenAI-compatible API via a corporate/custom gateway
  gemini     — Google Gemini API

Each provider exposes a single function:

    complete(system: str, user: str, model: str, temperature: float,
             max_tokens: int) -> str

which returns the raw text content of the model's response.

────────────────────────────────────────────────────────────────────────────
Provider configuration (environment variables)
────────────────────────────────────────────────────────────────────────────

ANTHROPIC  (LLM_PROVIDER=anthropic, default)
  ANTHROPIC_API_KEY   — required
  LLM_MODEL           — default: claude-sonnet-4-20250514

BEDROCK    (LLM_PROVIDER=bedrock)
  AWS_ACCESS_KEY_ID   — or use an IAM role / AWS SSO profile
  AWS_SECRET_ACCESS_KEY
  AWS_SESSION_TOKEN   — if using temporary credentials
  AWS_REGION          — default: us-east-1
  LLM_MODEL           — Bedrock model ID, default:
                          anthropic.claude-sonnet-4-5-20251001-v1:0
                        Other supported IDs:
                          anthropic.claude-opus-4-5-20251001-v1:0
                          anthropic.claude-haiku-4-5-20251001-v1:0

OLLAMA     (LLM_PROVIDER=ollama)
  OLLAMA_BASE_URL     — default: http://localhost:11434
  LLM_MODEL           — default: llama3.2
                        Any model pulled via `ollama pull <model>`

OPENAI     (LLM_PROVIDER=openai)
  OPENAI_API_KEY      — required (your OpenAI API key)
  LLM_MODEL           — default: gpt-4o
  OPENAI_ORG          — optional organisation ID

OPENAI_CORPORATE  (LLM_PROVIDER=openai_corporate)
  OPENAI_API_KEY      — required (use your gateway key / token)
  OPENAI_BASE_URL     — required, your corporate gateway URL
  LLM_MODEL           — default: gpt-4o
  OPENAI_ORG          — optional organisation ID

GEMINI     (LLM_PROVIDER=gemini)
  GEMINI_API_KEY      — required (Google AI Studio key)
  LLM_MODEL           — default: gemini-2.0-flash
                        Other options: gemini-2.5-flash, gemini-2.5-pro,
                          gemini-2.0-pro

────────────────────────────────────────────────────────────────────────────
Quick start
────────────────────────────────────────────────────────────────────────────

# Anthropic (default — no change from existing behaviour)
export ANTHROPIC_API_KEY="sk-ant-..."
python src/server.py

# Amazon Bedrock
export LLM_PROVIDER=bedrock
export AWS_REGION=us-east-1
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
python src/server.py

# Bedrock with a specific model
export LLM_PROVIDER=bedrock
export LLM_MODEL=anthropic.claude-opus-4-5-20251001-v1:0
python src/server.py

# Ollama (local)
export LLM_PROVIDER=ollama
export LLM_MODEL=llama3.2
python src/server.py

# OpenAI (direct)
export LLM_PROVIDER=openai
export OPENAI_API_KEY="sk-..."
export LLM_MODEL=gpt-4o
python src/server.py

# OpenAI via corporate gateway
export LLM_PROVIDER=openai_corporate
export OPENAI_API_KEY="your-gateway-token"
export OPENAI_BASE_URL="https://api.your-company.com/openai/v1"
export LLM_MODEL=gpt-4o
python src/server.py

# Gemini
export LLM_PROVIDER=gemini
export GEMINI_API_KEY="AIza..."
export LLM_MODEL=gemini-2.0-flash
python src/server.py
"""

import json
import logging
import os
from typing import Any

log = logging.getLogger("cx-mcp.providers")

# ---------------------------------------------------------------------------
# Provider / model defaults
# ---------------------------------------------------------------------------

PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic").lower().strip()

_DEFAULTS: dict[str, str] = {
    "anthropic":        "claude-sonnet-4-20250514",
    "bedrock":          "anthropic.claude-sonnet-4-5-20251001-v1:0",
    "ollama":           "llama3.2",
    "openai":           "gpt-4o",
    "openai_corporate": "gpt-4o",
    "gemini":           "gemini-2.0-flash",
}

MODEL = os.environ.get("LLM_MODEL", "") or _DEFAULTS.get(PROVIDER, "")

VALID_PROVIDERS = set(_DEFAULTS.keys())


# ---------------------------------------------------------------------------
# Lazy-loaded clients (one per process)
# ---------------------------------------------------------------------------

_anthropic_client: Any = None
_bedrock_client: Any = None
_openai_client: Any = None
_openai_corporate_client: Any = None
_gemini_client: Any = None


def _get_anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic as _anthropic_sdk
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set. "
                "Export it before starting the server."
            )
        _anthropic_client = _anthropic_sdk.Anthropic(api_key=api_key)
        log.info("Anthropic client initialised (model: %s)", MODEL)
    return _anthropic_client


def _get_bedrock():
    global _bedrock_client
    if _bedrock_client is None:
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 is required for Bedrock. Install it:\n"
                "  pip install boto3"
            )
        region = os.environ.get("AWS_REGION", "us-east-1")
        _bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name=region,
        )
        log.info(
            "Bedrock client initialised (region: %s, model: %s)",
            region, MODEL,
        )
    return _bedrock_client


def _get_openai():
    global _openai_client
    if _openai_client is None:
        try:
            import openai as _openai_sdk
        except ImportError:
            raise ImportError(
                "openai is required for the OpenAI provider. Install it:\n"
                "  pip install openai"
            )
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. "
                "Export it before starting the server."
            )
        org = os.environ.get("OPENAI_ORG")
        _openai_client = _openai_sdk.OpenAI(
            api_key=api_key,
            organization=org or None,
        )
        log.info("OpenAI client initialised (model: %s)", MODEL)
    return _openai_client


def _get_openai_corporate():
    global _openai_corporate_client
    if _openai_corporate_client is None:
        try:
            import openai as _openai_sdk
        except ImportError:
            raise ImportError(
                "openai is required for the openai_corporate provider. Install it:\n"
                "  pip install openai"
            )
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. "
                "Export it before starting the server."
            )
        base_url = os.environ.get("OPENAI_BASE_URL")
        if not base_url:
            raise EnvironmentError(
                "OPENAI_BASE_URL is not set. "
                "Set it to your corporate gateway URL."
            )
        org = os.environ.get("OPENAI_ORG")
        _openai_corporate_client = _openai_sdk.OpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=org or None,
        )
        log.info(
            "OpenAI corporate client initialised (base_url: %s, model: %s)",
            base_url, MODEL,
        )
    return _openai_corporate_client


def _get_gemini():
    global _gemini_client
    if _gemini_client is None:
        try:
            from google import genai as _genai
        except ImportError:
            raise ImportError(
                "google-genai is required for the Gemini provider. Install it:\n"
                "  pip install google-genai"
            )
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set. "
                "Get a key from Google AI Studio and export it before starting the server."
            )
        _gemini_client = _genai.Client(api_key=api_key)
        log.info("Gemini client initialised (model: %s)", MODEL)
    return _gemini_client


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

def _complete_anthropic(
    system: str,
    user: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> tuple[str, dict]:
    """Call the Anthropic API directly."""
    client = _get_anthropic()
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = message.content[0].text
    usage = {
        "input_tokens":  message.usage.input_tokens,
        "output_tokens": message.usage.output_tokens,
        "stop_reason":   message.stop_reason,
    }
    return text, usage


def _complete_bedrock(
    system: str,
    user: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> tuple[str, dict]:
    """
    Call an Anthropic model via Amazon Bedrock using the Converse API.

    The Bedrock Converse API supports the same system / messages structure
    as the Anthropic SDK, so no prompt reformatting is needed.
    """
    client = _get_bedrock()

    body = {
        "system": [{"text": system}],
        "messages": [{"role": "user", "content": [{"text": user}]}],
        "inferenceConfig": {
            "maxTokens":   max_tokens,
            "temperature": temperature,
        },
    }

    response = client.converse(modelId=model, **body)

    output   = response["output"]["message"]["content"][0]["text"]
    tok_in   = response["usage"]["inputTokens"]
    tok_out  = response["usage"]["outputTokens"]
    stop     = response["stopReason"]

    usage = {
        "input_tokens":  tok_in,
        "output_tokens": tok_out,
        "stop_reason":   stop,
    }
    return output, usage


def _complete_ollama(
    system: str,
    user: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> tuple[str, dict]:
    """
    Call a locally hosted model via Ollama's OpenAI-compatible /chat endpoint.

    Ollama must be running:  ollama serve
    Model must be pulled:    ollama pull <model>
    """
    try:
        import httpx
    except ImportError:
        raise ImportError(
            "httpx is required for Ollama. Install it:\n"
            "  pip install httpx"
        )

    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    url = f"{base_url.rstrip('/')}/api/chat"

    payload = {
        "model": model,
        "messages": [
            {"role": "system",  "content": system},
            {"role": "user",    "content": user},
        ],
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
        "stream": False,
    }

    with httpx.Client(timeout=180) as client:
        resp = client.post(url, json=payload)

    if resp.status_code != 200:
        raise RuntimeError(
            f"Ollama returned HTTP {resp.status_code}: {resp.text[:300]}"
        )

    data = resp.json()
    text = data["message"]["content"]
    usage = {
        "input_tokens":  data.get("prompt_eval_count", 0),
        "output_tokens": data.get("eval_count", 0),
        "stop_reason":   data.get("done_reason", "stop"),
    }
    return text, usage


def _complete_openai(
    system: str,
    user: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> tuple[str, dict]:
    """Call the OpenAI API directly (api.openai.com)."""
    client = _get_openai()

    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )

    text  = response.choices[0].message.content or ""
    usage = {
        "input_tokens":  response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "stop_reason":   response.choices[0].finish_reason,
    }
    return text, usage


def _complete_openai_corporate(
    system: str,
    user: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> tuple[str, dict]:
    """
    Call an OpenAI-compatible API via a corporate/custom gateway.

    Requires OPENAI_BASE_URL pointing to your gateway endpoint.
    Works with Azure OpenAI and any OpenAI-compatible /chat/completions gateway.
    """
    client = _get_openai_corporate()

    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )

    text  = response.choices[0].message.content or ""
    usage = {
        "input_tokens":  response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "stop_reason":   response.choices[0].finish_reason,
    }
    return text, usage


def _complete_gemini(
    system: str,
    user: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> tuple[str, dict]:
    """Call the Google Gemini API."""
    from google.genai import types as _genai_types

    client = _get_gemini()

    response = client.models.generate_content(
        model=model,
        contents=user,
        config=_genai_types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
        ),
    )

    text = response.text or ""
    candidate = response.candidates[0] if response.candidates else None
    usage_meta = response.usage_metadata
    usage = {
        "input_tokens":  usage_meta.prompt_token_count if usage_meta else 0,
        "output_tokens": usage_meta.candidates_token_count if usage_meta else 0,
        "stop_reason":   candidate.finish_reason.name if candidate else "stop",
    }
    return text, usage


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

_DISPATCH = {
    "anthropic":        _complete_anthropic,
    "bedrock":          _complete_bedrock,
    "ollama":           _complete_ollama,
    "openai":           _complete_openai,
    "openai_corporate": _complete_openai_corporate,
    "gemini":           _complete_gemini,
}


def complete(
    system: str,
    user: str,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1500,
) -> tuple[str, dict]:
    """
    Send a system + user prompt to the configured LLM provider.

    Args:
        system:      System prompt string.
        user:        User message string.
        model:       Model identifier. If None, uses the LLM_MODEL env var
                     (or the provider default).
        temperature: Sampling temperature 0.0–1.0.
        max_tokens:  Maximum tokens to generate.

    Returns:
        (text, usage) where text is the raw model output string and usage is a
        dict with input_tokens, output_tokens, stop_reason.

    Raises:
        ValueError:      Unknown provider.
        EnvironmentError: Missing required credentials.
        RuntimeError:    API call failed.
    """
    if PROVIDER not in VALID_PROVIDERS:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{PROVIDER}'. "
            f"Valid options: {sorted(VALID_PROVIDERS)}"
        )

    effective_model = model or MODEL
    fn = _DISPATCH[PROVIDER]

    log.debug(
        "LLM call: provider=%s model=%s temperature=%s max_tokens=%s",
        PROVIDER, effective_model, temperature, max_tokens,
    )

    return fn(
        system=system,
        user=user,
        model=effective_model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def provider_info() -> dict:
    """Return a dict describing the active provider and model for logging/debug."""
    return {
        "provider": PROVIDER,
        "model":    MODEL,
        "config": {
            "anthropic": {"api_key_set": bool(os.environ.get("ANTHROPIC_API_KEY"))},
            "bedrock":   {"region": os.environ.get("AWS_REGION", "us-east-1")},
            "ollama":    {"base_url": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")},
            "openai":    {"api_key_set": bool(os.environ.get("OPENAI_API_KEY"))},
            "openai_corporate": {
                "base_url":    os.environ.get("OPENAI_BASE_URL", ""),
                "api_key_set": bool(os.environ.get("OPENAI_API_KEY")),
            },
            "gemini": {"api_key_set": bool(os.environ.get("GEMINI_API_KEY"))},
        }.get(PROVIDER, {}),
    }
