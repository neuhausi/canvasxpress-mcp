"""Public-deployment controls (P1): bearer-token auth + cost kill-switch/ceiling.

In-process and offline. The fail-closed startup check runs in a subprocess because
it is a module-import-time guard.
"""

import asyncio
import os
import subprocess
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

import server  # noqa: E402


# --- ASGI driver for the auth middleware --------------------------------------

def _drive(mw, method="POST", path="/mcp", headers=None):
    """Run one request through the middleware; return (status, inner_reached)."""
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or [])],
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    sent = []

    async def send(msg):
        sent.append(msg)

    reached = {"inner": False}

    async def inner(s, r, sd):
        reached["inner"] = True
        await sd({"type": "http.response.start", "status": 200, "headers": []})
        await sd({"type": "http.response.body", "body": b"ok"})

    mw._app = inner
    asyncio.run(mw(scope, receive, send))
    status = next((m["status"] for m in sent if m.get("type") == "http.response.start"), None)
    return status, reached["inner"]


@pytest.fixture
def auth_on(monkeypatch):
    monkeypatch.setattr(server, "AUTH_REQUIRED", True)
    monkeypatch.setattr(server, "AUTH_TOKENS", {"good-token"})
    monkeypatch.setattr(server, "AUTH_EXEMPT_PATHS", ("/health",))
    monkeypatch.setattr(server, "RATE_LIMIT_PER_MIN", 0)
    return server._BearerAuthMiddleware(app=None)


# --- auth --------------------------------------------------------------------

def test_valid_token_passes(auth_on):
    status, reached = _drive(auth_on, headers=[("Authorization", "Bearer good-token")])
    assert status == 200 and reached


def test_missing_token_rejected(auth_on):
    status, reached = _drive(auth_on, headers=[])
    assert status == 401 and not reached


def test_bad_token_rejected(auth_on):
    status, reached = _drive(auth_on, headers=[("Authorization", "Bearer nope")])
    assert status == 401 and not reached


def test_health_is_exempt(auth_on):
    status, reached = _drive(auth_on, method="GET", path="/health")
    assert status == 200 and reached


def test_options_preflight_passes(auth_on):
    status, reached = _drive(auth_on, method="OPTIONS")
    assert status == 200 and reached


def test_rate_limit(monkeypatch):
    monkeypatch.setattr(server, "AUTH_REQUIRED", True)
    monkeypatch.setattr(server, "AUTH_TOKENS", {"good-token"})
    monkeypatch.setattr(server, "AUTH_EXEMPT_PATHS", ("/health",))
    monkeypatch.setattr(server, "RATE_LIMIT_PER_MIN", 2)
    mw = server._BearerAuthMiddleware(app=None)
    hdr = [("Authorization", "Bearer good-token")]
    assert _drive(mw, headers=hdr)[0] == 200
    assert _drive(mw, headers=hdr)[0] == 200
    assert _drive(mw, headers=hdr)[0] == 429  # 3rd in the same minute


def test_auth_off_is_passthrough(monkeypatch):
    monkeypatch.setattr(server, "AUTH_REQUIRED", False)
    mw = server._BearerAuthMiddleware(app=None)
    status, reached = _drive(mw, headers=[])  # no token, still allowed
    assert status == 200 and reached


# --- cost controls -----------------------------------------------------------

def test_cost_ceiling_pauses(monkeypatch):
    monkeypatch.setattr(server, "GENERATION_DISABLED", False)
    monkeypatch.setattr(server, "DAILY_COST_LIMIT_USD", 1.0)
    monkeypatch.setattr(server, "_cost_state", {"day": None, "spent_usd": 0.0})
    assert server.generation_block_reason() is None
    server.record_generation_cost(0.5)
    assert server.generation_block_reason() is None
    server.record_generation_cost(0.6)  # now 1.1 >= 1.0
    assert "ceiling" in server.generation_block_reason()


def test_manual_kill_switch(monkeypatch):
    monkeypatch.setattr(server, "GENERATION_DISABLED", True)
    assert "disabled" in server.generation_block_reason()
    with pytest.raises(server.GenerationPaused):
        server._guarded_llm_complete("system", "user")


def test_no_cost_controls_by_default(monkeypatch):
    monkeypatch.setattr(server, "GENERATION_DISABLED", False)
    monkeypatch.setattr(server, "DAILY_COST_LIMIT_USD", 0.0)
    monkeypatch.setattr(server, "_cost_state", {"day": None, "spent_usd": 0.0})
    server.record_generation_cost(9999.0)
    assert server.generation_block_reason() is None


# --- fail-closed startup (import-time guard, so: subprocess) ------------------

def test_public_profile_without_token_fails_closed():
    env = dict(os.environ, CX_PUBLIC="1", CX_AUTH_TOKENS="", CX_AUTH_TOKENS_FILE="")
    r = subprocess.run(
        [sys.executable, "-c", "import sys; sys.path.insert(0, 'src'); import server"],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )
    assert r.returncode != 0
    assert "fail-closed" in (r.stdout + r.stderr).lower()
