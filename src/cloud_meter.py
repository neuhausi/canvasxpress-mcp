"""CanvasXpress Cloud AI metering for the MCP server.

Bridges the MCP server to the CanvasXpress Cloud entitlement store (owned by the
dashboards ``cxd_server``): when a signed-in Cloud account runs a model-backed
tool, one prepaid AI credit is reserved. This is **off by default** — a
standalone or self-hosted MCP server is never metered, consistent with the
open-core model (the engine and the MCP server are free; only the hosted Cloud
meters AI usage).

Enable on the hosted Cloud by setting:

* ``CX_AI_METER=on``
* ``CX_CLOUD_BILLING_URL`` — base URL of the cxd_server (e.g. ``http://127.0.0.1:8200``)
* ``CXD_METER_KEY`` — the shared secret that matches cxd_server's ``CXD_METER_KEY``

The account id sent to the meter is the caller's ``api_key_id`` (the same sha256
prefix the call log uses); the Cloud grants AI credits against that id.
"""

from __future__ import annotations

import json
import os
import urllib.request

#: Tool paths that invoke a model and therefore cost one AI credit. Retrieval /
#: static tools (``/params``, ``/minimal-params``) are intentionally excluded.
METERED_PATHS = frozenset({
    "/generate", "/modify", "/action", "/km", "/select", "/axes",
    "/explain", "/explain-r", "/explain-ggplot", "/map",
})


def metering_enabled() -> bool:
    """True only when the hosted Cloud has explicitly turned metering on and the
    billing endpoint + shared secret are configured."""
    return (os.getenv("CX_AI_METER", "off").lower() in ("on", "1", "true", "yes")
            and bool(os.getenv("CX_CLOUD_BILLING_URL"))
            and bool(os.getenv("CXD_METER_KEY")))


def _consume(account: str, n: int) -> dict:
    """Call cxd_server's ``/billing/consume``. ``n == 0`` is a pure balance read;
    ``n >= 1`` reserves that many credits. Returns the endpoint's
    ``{"ok", "ai_credits"}`` or, on any transport error, a closed-fail result so
    a billing outage never silently gives away paid generations."""
    base = os.getenv("CX_CLOUD_BILLING_URL", "").rstrip("/")
    req = urllib.request.Request(
        base + "/billing/consume",
        data=json.dumps({"account": account, "n": n}).encode(),
        headers={"Content-Type": "application/json",
                 "X-Cloud-Meter-Key": os.getenv("CXD_METER_KEY", "")},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.load(resp)
    except Exception:
        return {"ok": False, "ai_credits": 0, "error": "meter_unavailable"}


def reserve_credit(account: str, n: int = 1) -> dict:
    """Reserve ``n`` AI credits for ``account`` before a generation runs."""
    return _consume(account, n)


def check_balance(account: str) -> dict:
    """Read the account's AI credit balance without spending (``n == 0``)."""
    return _consume(account, 0)
