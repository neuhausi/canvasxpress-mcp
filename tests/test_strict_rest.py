"""P2b: admin key never printed, admin endpoints disabled without ADMIN_KEY,
CX_STRICT_REST hardening, and blocking calls kept off the event loop. Offline."""

import asyncio
import json
import os
import subprocess
import sys
import time

import pytest
from starlette.requests import Request

ROOT = os.path.join(os.path.dirname(__file__), "..")
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

import server  # noqa: E402


def _request(headers=None, query=""):
    raw = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    return Request({"type": "http", "method": "GET", "headers": raw,
                    "query_string": query.encode(), "path": "/"})


def test_admin_disabled_without_configured_key(monkeypatch):
    monkeypatch.setattr(server, "ADMIN_KEY_CONFIGURED", False)
    deny = server._require_admin_key(_request({"X-Admin-Key": server.ADMIN_KEY}))
    assert deny.status_code == 403 and b"disabled" in deny.body


def test_admin_accepts_only_the_configured_key(monkeypatch):
    monkeypatch.setattr(server, "ADMIN_KEY_CONFIGURED", True)
    monkeypatch.setattr(server, "ADMIN_KEY", "s3cret")
    assert server._require_admin_key(_request({"X-Admin-Key": "s3cret"})) is None
    assert server._require_admin_key(_request({"X-Admin-Key": "nope"})).status_code == 403
    assert server._require_admin_key(_request()).status_code == 403


class _Log:
    def export(self, tool=None, rated_only=False, limit=500):
        return [{"id": 1}]


def test_export_open_by_default_and_gated_when_strict(monkeypatch):
    monkeypatch.setattr(server, "_call_log", _Log())
    monkeypatch.setattr(server, "STRICT_REST", False)
    open_resp = asyncio.run(server.rest_feedback_export(_request()))
    assert json.loads(open_resp.body)["count"] == 1
    monkeypatch.setattr(server, "STRICT_REST", True)
    monkeypatch.setattr(server, "ADMIN_KEY_CONFIGURED", True)
    monkeypatch.setattr(server, "ADMIN_KEY", "s3cret")
    assert asyncio.run(server.rest_feedback_export(_request())).status_code == 403
    ok = asyncio.run(server.rest_feedback_export(_request({"X-Admin-Key": "s3cret"})))
    assert json.loads(ok.body)["count"] == 1


def test_jsonp_acao_header_dropped_when_strict(monkeypatch):
    cx = {"callback": "cb"}
    monkeypatch.setattr(server, "STRICT_REST", False)
    assert server._cx_response({"valid": True}, cx).headers.get("access-control-allow-origin") == "*"
    monkeypatch.setattr(server, "STRICT_REST", True)
    assert "access-control-allow-origin" not in server._cx_response({"valid": True}, cx).headers


def test_strict_mode_has_no_cors_middleware_and_default_keeps_it():
    probe = ("import sys; sys.path.insert(0, %r); import server as s; "
             "print([m.cls.__name__ for m in s._cors_middleware])" % SRC)
    def names(strict):
        env = {k: v for k, v in os.environ.items() if k != "CX_STRICT_REST"}
        if strict:
            env["CX_STRICT_REST"] = "1"
        out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                             env=env, check=True)
        return out.stdout.strip().splitlines()[-1]
    assert "CORSMiddleware" in names(False)
    assert "CORSMiddleware" not in names(True)


def test_blocking_generate_does_not_stall_the_event_loop(monkeypatch):
    def slow_generate(**kwargs):
        time.sleep(0.4)
        return {"config": {}, "valid": True}

    async def fake_kwargs(request, require_description=False):
        return {"description": "d", "_cx": {}}, 200, ""

    monkeypatch.setattr(server, "generate_canvasxpress_config", slow_generate)
    monkeypatch.setattr(server, "_kwargs_from_request", fake_kwargs)

    async def run():
        ticks = 0
        task = asyncio.create_task(server.rest_generate(_request()))
        while not task.done():
            await asyncio.sleep(0.02)
            ticks += 1
        return ticks, await task

    ticks, resp = asyncio.run(run())
    assert resp.status_code == 200
    assert ticks >= 5  # the loop kept running while the call blocked its worker thread
