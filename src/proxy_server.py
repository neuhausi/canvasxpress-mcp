#!/usr/bin/env python3
"""
proxy_server.py
===============
A thin HTTP proxy that sits in front of the MCP server and exposes a simple
REST-style API with named URL query parameters.

The MCP server speaks JSON-RPC over HTTP — not REST. This proxy translates
clean URL calls like:

    GET /api/generate?description=heatmap&headers=Gene,S1,S2&color_scheme=RdBu

into the correct JSON-RPC initialize → notifications/initialized → tools/call
sequence, and returns the result as plain JSON.

Usage:
    python proxy_server.py               # starts on port 8200
    PROXY_PORT=9200 python proxy_server.py

Environment variables:
    MCP_URL      URL of the MCP server  (default: http://localhost:8100/mcp)
    PROXY_HOST   Bind host              (default: 0.0.0.0)
    PROXY_PORT   Port to listen on      (default: 8200)

Endpoints
---------
GET  /api/generate
    description    (str, required)  Plain English chart description
    headers        (str)            Comma-separated column names
    column_types   (str)            JSON object  {"Col":"numeric",...}
    data           (str)            JSON array   [["Col1","Col2"],[1,2],...]
    temperature    (float)          0.0–1.0, default 0.0

GET  /api/modify
    config         (str, required)  JSON object  {"graphType":"Bar",...}
    instruction    (str, required)  Plain English modification instruction
    headers        (str)            Comma-separated column names
    column_types   (str)            JSON object
    data           (str)            JSON array
    temperature    (float)

GET  /api/km
    description    (str)            Plain English KM description
    headers        (str)            Comma-separated column names
    data           (str)            JSON array
    config         (str)            Existing KM config JSON object
    add_annotations (str)           "true"|"false", default "true"
    temperature    (float)

GET  /api/query_params
    graph_type     (str)            CanvasXpress graph type
    param_name     (str)            Parameter name to look up
    refresh        (str)            "true" to force re-fetch from GitHub

GET  /api/health
    Returns {"status":"ok","mcp_url":"..."} — useful for connectivity checks

GET  /api/url_builder
    Returns the web UI for building URLs (serves web_client.html)
"""

import http.server
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

MCP_URL    = os.environ.get("MCP_URL",    "http://localhost:8100/mcp")
PROXY_HOST = os.environ.get("PROXY_HOST", "0.0.0.0")
PROXY_PORT = int(os.environ.get("PROXY_PORT", "8200"))

_REQ_ID = 0


def _next_id() -> int:
    global _REQ_ID
    _REQ_ID += 1
    return _REQ_ID


# ---------------------------------------------------------------------------
# MCP JSON-RPC helpers
# ---------------------------------------------------------------------------

def _post_json(url: str, payload: dict, session_id: str | None = None) -> tuple[dict, str | None]:
    body    = json.dumps(payload).encode()
    headers = {
        "Content-Type": "application/json",
        "Accept":       "application/json, text/event-stream",
    }
    if session_id:
        headers["Mcp-Session-Id"] = session_id

    req  = urllib.request.Request(url, data=body, headers=headers, method="POST")
    resp = urllib.request.urlopen(req, timeout=120)

    new_sid = resp.headers.get("Mcp-Session-Id") or resp.headers.get("mcp-session-id")
    ct      = resp.headers.get("Content-Type", "")
    raw     = resp.read().decode("utf-8").strip()

    if not raw:
        return {}, new_sid

    if "event-stream" in ct or raw.startswith("data:") or "\ndata:" in raw:
        return _parse_sse(raw), new_sid

    return json.loads(raw), new_sid


def _get_stream(url: str, session_id: str, timeout: int = 120) -> dict:
    """Open a GET SSE stream and read until we get a result message."""
    req  = urllib.request.Request(
        url,
        headers={"Accept": "text/event-stream", "Mcp-Session-Id": session_id},
        method="GET",
    )
    resp = urllib.request.urlopen(req, timeout=timeout)
    cur  = ""
    for raw_line in resp:
        line = raw_line.decode("utf-8").rstrip("\r\n")
        if line.startswith("data:"):
            cur += line[5:].strip()
        elif line == "" and cur:
            chunk = cur.strip()
            cur   = ""
            if chunk.startswith("{"):
                obj = json.loads(chunk)
                if "result" in obj or "error" in obj:
                    return obj
    return {}


def _parse_sse(body: str) -> dict:
    chunks: list[str] = []
    cur = ""
    for line in body.splitlines():
        line = line.rstrip("\r")
        if line.startswith("data:"):
            cur += line[5:].strip()
        elif line == "" and cur:
            chunks.append(cur)
            cur = ""
    if cur:
        chunks.append(cur)
    for chunk in reversed(chunks):
        chunk = chunk.strip()
        if chunk.startswith("{"):
            return json.loads(chunk)
    return {}


def _call_tool(tool_name: str, arguments: dict) -> dict:
    """
    Full MCP session: initialize → notifications/initialized → tools/call.
    Handles both inline SSE and 202 + GET-stream patterns.
    """
    # Step 1: initialize
    init_payload = {
        "jsonrpc": "2.0", "id": _next_id(), "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities":    {},
            "clientInfo":      {"name": "proxy-server", "version": "1.0"},
        },
    }
    init_resp, sid = _post_json(MCP_URL, init_payload)
    if not sid:
        sid = init_resp.get("_session_id")
    if "error" in init_resp:
        raise RuntimeError(f"MCP initialize failed: {init_resp['error']}")

    # Step 2: notifications/initialized (always 202)
    notif = {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
    try:
        _post_json(MCP_URL, notif, session_id=sid)
    except Exception:
        pass  # 202 with empty body raises on some urllib versions

    # Step 3: tools/call
    call_payload = {
        "jsonrpc": "2.0", "id": _next_id(), "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
    }

    try:
        result, _ = _post_json(MCP_URL, call_payload, session_id=sid)
    except urllib.error.HTTPError as e:
        if e.code == 202 and sid:
            result = _get_stream(MCP_URL, sid)
        else:
            raise

    # Handle 202 with empty body (result came back as empty dict)
    if not result and sid:
        result = _get_stream(MCP_URL, sid)

    if "error" in result:
        raise RuntimeError(f"MCP tool call failed: {result['error']}")

    content = result.get("result", {}).get("content", [])
    if not content:
        return {}
    return json.loads(content[0]["text"])


# ---------------------------------------------------------------------------
# Parameter parsing helpers
# ---------------------------------------------------------------------------

def _parse_qs(query_string: str) -> dict[str, str]:
    params = urllib.parse.parse_qs(query_string, keep_blank_values=False)
    return {k: v[0] for k, v in params.items()}


def _parse_headers(raw: str) -> list[str]:
    """'Gene,Sample1,Treatment' → ['Gene', 'Sample1', 'Treatment']"""
    return [h.strip() for h in raw.split(",") if h.strip()]


def _parse_json_param(raw: str, name: str) -> object:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in '{name}': {e}")


def _float_param(raw: str, name: str, default: float = 0.0) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"'{name}' must be a number, got {repr(raw)}")


# ---------------------------------------------------------------------------
# Endpoint handlers
# ---------------------------------------------------------------------------

def handle_generate(params: dict) -> dict:
    if "description" not in params:
        raise ValueError("'description' is required")

    args: dict = {"description": params["description"]}

    if "data" in params:
        args["data"] = _parse_json_param(params["data"], "data")
    elif "headers" in params:
        args["headers"] = _parse_headers(params["headers"])

    if "column_types" in params:
        args["column_types"] = _parse_json_param(params["column_types"], "column_types")

    if "temperature" in params:
        args["temperature"] = _float_param(params["temperature"], "temperature")

    return _call_tool("generate_canvasxpress_config", args)


def handle_modify(params: dict) -> dict:
    for req in ("config", "instruction"):
        if req not in params:
            raise ValueError(f"'{req}' is required")

    args: dict = {
        "config":      _parse_json_param(params["config"], "config"),
        "instruction": params["instruction"],
    }

    if "data" in params:
        args["data"] = _parse_json_param(params["data"], "data")
    elif "headers" in params:
        args["headers"] = _parse_headers(params["headers"])

    if "column_types" in params:
        args["column_types"] = _parse_json_param(params["column_types"], "column_types")

    if "temperature" in params:
        args["temperature"] = _float_param(params["temperature"], "temperature")

    return _call_tool("modify_canvasxpress_config", args)


def handle_km(params: dict) -> dict:
    if not any(k in params for k in ("description", "headers", "data", "config")):
        raise ValueError("At least one of: description, headers, data, config is required")

    args: dict = {}

    if "description" in params:
        args["description"] = params["description"]

    if "data" in params:
        args["data"] = _parse_json_param(params["data"], "data")
    elif "headers" in params:
        args["headers"] = _parse_headers(params["headers"])

    if "config" in params:
        args["config"] = _parse_json_param(params["config"], "config")

    args["add_annotations"] = params.get("add_annotations", "true").lower() != "false"

    if "temperature" in params:
        args["temperature"] = _float_param(params["temperature"], "temperature")

    return _call_tool("generate_km_config", args)


def handle_query_params(params: dict) -> dict:
    args: dict = {}
    if "graph_type"  in params: args["graph_type"]  = params["graph_type"]
    if "param_name"  in params: args["param_name"]  = params["param_name"]
    if "refresh"     in params: args["refresh"]      = params["refresh"].lower() == "true"
    return _call_tool("query_canvasxpress_params", args)


def handle_health(_params: dict) -> dict:
    return {"status": "ok", "mcp_url": MCP_URL, "proxy_port": PROXY_PORT}


# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------

_ROUTE_MAP = {
    "/api/generate":     handle_generate,
    "/api/modify":       handle_modify,
    "/api/km":           handle_km,
    "/api/query_params": handle_query_params,
    "/api/health":       handle_health,
}

_WEB_CLIENT = Path(__file__).parent.parent / "web_client.html"


class ProxyHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[proxy] {self.address_string()} {fmt % args}")

    def _send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type",  "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, body: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type",  "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path   = parsed.path.rstrip("/")
        params = _parse_qs(parsed.query)

        # Serve the web client HTML
        if path in ("/", "/api/url_builder", "/web_client"):
            if _WEB_CLIENT.exists():
                self._send_html(_WEB_CLIENT.read_bytes())
            else:
                self._send_json({"error": "web_client.html not found"}, 404)
            return

        handler = _ROUTE_MAP.get(path)
        if not handler:
            self._send_json({"error": f"Unknown endpoint '{path}'"}, 404)
            return

        try:
            result = handler(params)
            self._send_json(result)
        except ValueError as e:
            self._send_json({"error": str(e), "type": "invalid_parameters"}, 400)
        except RuntimeError as e:
            self._send_json({"error": str(e), "type": "mcp_error"}, 502)
        except Exception as e:
            self._send_json({"error": str(e), "type": "internal_error"}, 500)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    server = http.server.ThreadingHTTPServer((PROXY_HOST, PROXY_PORT), ProxyHandler)
    print(f"CanvasXpress MCP Proxy")
    print(f"  MCP server  : {MCP_URL}")
    print(f"  Proxy URL   : http://localhost:{PROXY_PORT}")
    print(f"  Web client  : http://localhost:{PROXY_PORT}/")
    print()
    print(f"  Endpoints:")
    print(f"    GET http://localhost:{PROXY_PORT}/api/generate?description=...")
    print(f"    GET http://localhost:{PROXY_PORT}/api/modify?config={{...}}&instruction=...")
    print(f"    GET http://localhost:{PROXY_PORT}/api/km?description=...")
    print(f"    GET http://localhost:{PROXY_PORT}/api/query_params?graph_type=Heatmap")
    print(f"    GET http://localhost:{PROXY_PORT}/api/health")
    print()
    server.serve_forever()
