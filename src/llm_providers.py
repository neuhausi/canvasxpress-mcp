#!/usr/bin/env python3
"""
llm_providers.py
================
Unified LLM provider abstraction for the CanvasXpress MCP server.

Supports four providers, selected via the LLM_PROVIDER environment variable:

  anthropic  — Direct Anthropic API (default)
  bedrock    — Anthropic models via Amazon Bedrock
  ollama     — Locally hosted models via Ollama
  openai     — OpenAI-compatible API (including corporate gateways)

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
  OPENAI_API_KEY      — required (use your gateway key / token)
  OPENAI_BASE_URL     — default: https://api.openai.com/v1
                        Override with your corporate gateway URL
  LLM_MODEL           — default: gpt-4o
  OPENAI_ORG          — optional organisation ID

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

# OpenAI via corporate gateway
export LLM_PROVIDER=openai
export OPENAI_API_KEY="your-gateway-token"
export OPENAI_BASE_URL="https://api.your-company.com/openai/v1"
export LLM_MODEL=gpt-4o
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
    "anthropic": "claude-sonnet-4-20250514",
    "bedrock":   "anthropic.claude-sonnet-4-5-20251001-v1:0",
    "ollama":    "llama3.2",
    "openai":    "gpt-4o",
}

MODEL = os.environ.get("LLM_MODEL", "") or _DEFAULTS.get(PROVIDER, "")

VALID_PROVIDERS = set(_DEFAULTS.keys())


# ---------------------------------------------------------------------------
# Lazy-loaded clients (one per process)
# ---------------------------------------------------------------------------

_anthropic_client: Any = None
_bedrock_client: Any = None
_openai_client: Any = None


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
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        org = os.environ.get("OPENAI_ORG")
        _openai_client = _openai_sdk.OpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=org or None,
        )
        log.info(
            "OpenAI client initialised (base_url: %s, model: %s)",
            base_url, MODEL,
        )
    return _openai_client


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
    """
    Call an OpenAI-compatible API.

    Works with:
      - OpenAI directly (api.openai.com)
      - Azure OpenAI (set OPENAI_BASE_URL to your Azure endpoint)
      - Corporate gateways that expose an OpenAI-compatible /chat/completions endpoint
    """
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


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

_DISPATCH = {
    "anthropic": _complete_anthropic,
    "bedrock":   _complete_bedrock,
    "ollama":    _complete_ollama,
    "openai":    _complete_openai,
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
            "openai":    {
                "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                "api_key_set": bool(os.environ.get("OPENAI_API_KEY")),
            },
        }.get(PROVIDER, {}),
    }
