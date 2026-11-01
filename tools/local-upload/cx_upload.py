#!/usr/bin/env python3
"""canvasxpress-upload: a local MCP server (stdio) with one tool, upload_csv(path).

It reads a CSV/TSV file from this machine and POSTs it to the CanvasXpress MCP server, which holds it for
30 minutes and returns a link. The model then calls suggest_canvasxpress_dashboard(url=<link>) on the remote
server: the file's rows never pass through the model, and no sandbox network policy is involved because
this process runs outside the sandbox, like any local MCP server.

Standard library only (no install step): JSON-RPC 2.0 over stdin/stdout, newline-delimited.
Config: CX_UPLOAD_URL (default https://canvasxpress.org/mcp/upload).
"""

import json
import os
import sys
import urllib.parse
import urllib.request

UPLOAD_URL = os.environ.get("CX_UPLOAD_URL", "https://canvasxpress.org/mcp/upload")
MAX_BYTES = 50 * 1024 * 1024

TOOL = {
    "name": "upload_csv",
    "description": (
        "Send a CSV/TSV file from this computer to the CanvasXpress server and get a link to it (valid 30 "
        "minutes). Use it FIRST whenever a dashboard or chart is wanted from a local file: pass the file's "
        "path, then call suggest_canvasxpress_dashboard with the returned url. Never read the file and type its "
        "rows into a tool call instead."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Path of the CSV/TSV file on this computer (~ is expanded)."}},
        "required": ["path"],
    },
}


def upload(path: str) -> dict:
    full = os.path.expanduser(os.path.expandvars(path.strip().strip("'\"")))
    if not os.path.isfile(full):
        return {"error": "no such file: %s" % full}
    size = os.path.getsize(full)
    if size > MAX_BYTES:
        return {"error": "file larger than 50 MB (%d bytes)" % size}
    with open(full, "rb") as handle:
        raw = handle.read()
    url = UPLOAD_URL + ("&" if "?" in UPLOAD_URL else "?") + "name=" + urllib.parse.quote(os.path.basename(full))
    request = urllib.request.Request(url, data=raw, method="POST",
                                     headers={"Content-Type": "text/plain; charset=utf-8", "User-Agent": "canvasxpress-upload/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            return {"error": json.loads(exc.read().decode("utf-8")).get("error") or str(exc)}
        except Exception:  # noqa: BLE001
            return {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"error": "upload failed: %s" % exc}
    body["next"] = "call suggest_canvasxpress_dashboard with url=%s" % body.get("url", "")
    return body


def handle(message: dict):
    method = message.get("method")
    ident = message.get("id")
    if method == "initialize":
        return {"protocolVersion": message.get("params", {}).get("protocolVersion") or "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "canvasxpress-upload", "version": "1.0.0"}}
    if method == "tools/list":
        return {"tools": [TOOL]}
    if method == "tools/call":
        params = message.get("params", {})
        if params.get("name") != "upload_csv":
            return {"content": [{"type": "text", "text": json.dumps({"error": "unknown tool"})}], "isError": True}
        out = upload(str(params.get("arguments", {}).get("path", "")))
        return {"content": [{"type": "text", "text": json.dumps(out)}], "isError": "error" in out}
    if method == "ping":
        return {}
    if ident is None:                                   # a notification: nothing to answer
        return None
    return {"_error": {"code": -32601, "message": "method not found: %s" % method}}


def main():
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line.decode("utf-8"))
        except ValueError:
            continue
        result = handle(message)
        if result is None or message.get("id") is None:
            continue
        reply = {"jsonrpc": "2.0", "id": message["id"]}
        if isinstance(result, dict) and "_error" in result:
            reply["error"] = result["_error"]
        else:
            reply["result"] = result
        stdout.write((json.dumps(reply) + "\n").encode("utf-8"))
        stdout.flush()


if __name__ == "__main__":
    main()
