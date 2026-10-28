"""Publish a composed dashboard spec to a running canvasxpress-dashboards server.

The compose/suggest tools return a spec you can render locally; this module
takes that final step — saving the spec into a dashboards *install* (the
self-hosted app or canvasxpress.org/dashboards) so it opens in the Builder and,
optionally, gets a public share link.

Configuration (all via environment / ``.env``):

    CXD_DASHBOARDS_URL      Base URL of the dashboards install, e.g.
                            "https://canvasxpress.org/dashboards" or
                            "http://127.0.0.1:8200". Required to publish.
    CXD_DASHBOARDS_API_KEY  Service token sent as the ``X-Cxd-Api-Key`` header.
                            The dashboards server authenticates it as its
                            configured ``CXD_API_USER`` (see that server's
                            ``CXD_API_KEY``/``CXD_API_USER``). Required to publish.
    CXD_DASHBOARDS_AUTH     Optional ``Authorization`` header value for an install
                            behind an authenticating proxy (e.g. "Bearer <token>").
    CXD_DASHBOARDS_TIMEOUT  Per-request timeout in seconds (default 30).

Pure functions apart from the two HTTP calls; no server state. Validation reuses
the vendored ``cx_dashboard_validate.validate_spec`` so a bad spec never reaches
the server.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

from cx_dashboard_validate import validate_spec


class PublishError(RuntimeError):
    """Publishing could not complete (no config, bad spec, or an HTTP error)."""


def base_url() -> Optional[str]:
    """The configured dashboards install base URL (``CXD_DASHBOARDS_URL``), or None."""
    url = (os.getenv("CXD_DASHBOARDS_URL") or "").strip()
    return url.rstrip("/") or None


def _api_key() -> Optional[str]:
    return (os.getenv("CXD_DASHBOARDS_API_KEY") or "").strip() or None


def _timeout() -> float:
    try:
        return float(os.getenv("CXD_DASHBOARDS_TIMEOUT", "30"))
    except (TypeError, ValueError):
        return 30.0


def enabled() -> bool:
    """Whether publishing is configured (both a base URL and a service token)."""
    return bool(base_url() and _api_key())


def _headers() -> dict:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    key = _api_key()
    if key:
        headers["X-Cxd-Api-Key"] = key
    auth = (os.getenv("CXD_DASHBOARDS_AUTH") or "").strip()
    if auth:
        headers["Authorization"] = auth
    return headers


def _post(path: str, payload: Any) -> dict:
    """POST ``payload`` as JSON to ``<base><path>``; return the parsed JSON body.

    Raises :class:`PublishError` with a readable message on any HTTP/transport
    error (the server's ``detail`` string is surfaced when present).
    """
    base = base_url()
    if not base:
        raise PublishError("CXD_DASHBOARDS_URL is not set — nowhere to publish to.")
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(base + path, data=body, headers=_headers(),
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=_timeout()) as resp:
            raw = resp.read().decode("utf-8") or "{}"
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = (json.loads(exc.read().decode("utf-8")) or {}).get("detail", "")
        except Exception:  # noqa: BLE001 - best-effort error text
            detail = ""
        hint = ""
        if exc.code in (401, 403):
            hint = (" (check CXD_DASHBOARDS_API_KEY here matches CXD_API_KEY on the "
                    "server, and that CXD_API_USER may create/share dashboards)")
        raise PublishError("POST %s failed: HTTP %s%s%s"
                           % (path, exc.code, (" — " + detail) if detail else "", hint))
    except urllib.error.URLError as exc:
        raise PublishError("POST %s failed: cannot reach %s (%s)"
                           % (path, base, getattr(exc, "reason", exc)))


def publish_dashboard(spec: dict, visibility: str = "private") -> dict:
    """Save a dashboard spec to the configured install and return its links.

    Args:
        spec: A composed dashboard spec (as returned by
            ``compose_canvasxpress_dashboard`` / ``suggest_canvasxpress_dashboard``).
            Must be an object with a non-empty ``id``.
        visibility: "private" (default — saved to the API user's account, opens in
            the Builder, no public link), "auth" (any signed-in viewer), or
            "public" (anyone with the link). Non-private visibilities mint a share
            link and need the API user to have the ``share.public`` permission.

    Returns:
        On success: ``{"published": true, "id", "builder_url", "open_url",
        "share_url"?, "embed"?, "valid": true, "errors": []}``. ``builder_url``
        opens the dashboard straight in the Builder (the server's ``?open=<id>``
        deep link). On failure: ``{"published": false, "errors": [...]}``.
    """
    if not isinstance(spec, dict) or not str(spec.get("id") or "").strip():
        return {"published": False, "valid": False,
                "errors": ["spec must be a dashboard object with a non-empty 'id' "
                           "(compose or suggest a dashboard first)"]}

    visibility = (visibility or "private").strip().lower()
    if visibility not in ("private", "auth", "public"):
        return {"published": False, "valid": False,
                "errors": ["visibility must be 'private', 'auth' or 'public'"]}

    # Never ship a spec the server would reject — validate locally first.
    check = validate_spec(spec)
    if not check["valid"]:
        return {"published": False, "valid": False, "errors": check["errors"]}

    base = base_url()
    if not base:
        return {"published": False, "valid": True,
                "errors": ["CXD_DASHBOARDS_URL is not set. Point it at your "
                           "canvasxpress-dashboards install (e.g. "
                           "https://canvasxpress.org/dashboards) and set "
                           "CXD_DASHBOARDS_API_KEY to its service token."]}
    if not _api_key():
        return {"published": False, "valid": True,
                "errors": ["CXD_DASHBOARDS_API_KEY is not set. It must match the "
                           "server's CXD_API_KEY so the publish is authenticated."]}

    dash_id = spec["id"]
    try:
        saved = _post("/api/dashboards", spec)
        out = {
            "published": True,
            "valid": True,
            "errors": [],
            "id": dash_id,
            "open_url": "%s/?open=%s" % (base, dash_id),
            "builder_url": "%s/?open=%s" % (base, dash_id),
        }
        summary = saved.get("dashboard") if isinstance(saved, dict) else None
        if isinstance(summary, dict) and summary.get("version"):
            out["version"] = summary["version"]

        if visibility != "private":
            shared = _post("/api/dashboards/%s/share" % urllib.parse.quote(dash_id),
                           {"visibility": visibility})
            s = shared.get("dashboard") if isinstance(shared, dict) else None
            if isinstance(s, dict):
                if s.get("share_url"):
                    out["share_url"] = s["share_url"]
                if s.get("embed"):
                    out["embed"] = s["embed"]
            out["visibility"] = visibility
        else:
            out["visibility"] = "private"
        return out
    except PublishError as exc:
        return {"published": False, "valid": True, "errors": [str(exc)]}
