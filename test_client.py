#!/usr/bin/env python3
"""
CanvasXpress MCP Test Client — Python
======================================
Usage:
    python test_client.py
    python test_client.py "Violin plot by cell type" "Gene,CellType,Expression"
    python test_client.py "Heatmap" '{"Gene":"string","Sample1":"numeric","Treatment":"factor"}'
    python test_client.py "Heatmap" "Gene,Sample1,Treatment" '{"Gene":"string","Sample1":"numeric","Treatment":"factor"}'

Requirements:
    pip install httpx
"""

import json
import sys
import os
import httpx

MCP_URL = os.environ.get("MCP_URL", "http://localhost:8100/mcp")


def post_mcp(client: httpx.Client, session_id: str | None, payload: dict) -> dict:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if session_id:
        headers["Mcp-Session-Id"] = session_id

    response = client.post(MCP_URL, json=payload, headers=headers)
    new_session_id = response.headers.get("mcp-session-id")
    body = response.text.strip()

    if "text/event-stream" in response.headers.get("content-type", "") or "data:" in body:
        json_str = _extract_sse_json(body)
        if not json_str:
            raise ValueError(f"Could not extract JSON from SSE response:\n{body}")
        result = json.loads(json_str)
    elif body.startswith("{"):
        result = json.loads(body)
    else:
        raise ValueError(f"Unexpected response ({response.status_code}):\n{body}")

    if new_session_id:
        result["_session_id"] = new_session_id
    return result


def _extract_sse_json(body: str) -> str | None:
    chunks, current = [], ""
    for line in body.splitlines():
        line = line.rstrip("\r")
        if line.startswith("data:"):
            current += line[5:].strip()
        elif line == "" and current:
            chunks.append(current)
            current = ""
    if current:
        chunks.append(current)
    for chunk in reversed(chunks):
        chunk = chunk.strip()
        if chunk.startswith("{"):
            return chunk
    return None


def generate_config(
    description: str,
    headers: list[str] | None = None,
    data: list[list] | None = None,
    column_types: dict[str, str] | None = None,
) -> dict:
    with httpx.Client(timeout=120) as client:
        init = post_mcp(client, None, {
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "python-client", "version": "1.0.0"},
            },
        })
        if "error" in init:
            raise RuntimeError(f"Initialize failed: {init['error']}")

        session_id = init.get("_session_id")
        print(f"Connected  : {MCP_URL}")

        post_mcp(client, session_id, {
            "jsonrpc": "2.0", "method": "notifications/initialized", "params": {},
        })

        args: dict = {"description": description}
        if data is not None:
            args["data"] = data
        elif headers is not None:
            args["headers"] = headers
        if column_types:
            args["column_types"] = column_types

        result = post_mcp(client, session_id, {
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "generate_canvasxpress_config", "arguments": args},
        })
        if "error" in result:
            raise RuntimeError(f"Tool call failed: {result['error']}")

        return json.loads(result["result"]["content"][0]["text"])


def main():
    description   = sys.argv[1] if len(sys.argv) > 1 else "Clustered heatmap with RdBu colors"
    headers       = None
    data          = None
    column_types  = None

    # Parse args: flexible — accept headers, data array, or column_types in any order
    for arg in sys.argv[2:]:
        if arg.startswith("{"):
            column_types = json.loads(arg)
        elif arg.startswith("["):
            data = json.loads(arg)
        else:
            headers = arg.split(",")

    # Default to sample data if nothing provided
    if headers is None and data is None:
        data = [
            ["Gene",  "Sample1", "Sample2", "Treatment"],
            ["BRCA1", 1.2,       3.4,       "Control"  ],
            ["TP53",  2.1,       0.9,       "Treated"  ],
            ["EGFR",  0.8,       2.3,       "Control"  ],
        ]
        column_types = {"Gene": "string", "Sample1": "numeric", "Sample2": "numeric", "Treatment": "factor"}

    print(f"Description : {description}")
    if data is not None:
        print(f"Data        : {len(data)-1} rows × {len(data[0])} columns")
        print(f"Columns     : {', '.join(str(h) for h in data[0])}")
    elif headers:
        print(f"Headers     : {', '.join(headers)}")
    if column_types:
        print(f"Types       : {', '.join(f'{k}={v}' for k,v in column_types.items())}")
    print()

    try:
        response = generate_config(description, headers=headers, data=data, column_types=column_types)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if response.get("headers_used"):
        print(f"Headers used : {', '.join(response['headers_used'])}")
    if response.get("types_used"):
        print(f"Types used   : {', '.join(f'{k}={v}' for k,v in response['types_used'].items())}")
    print()

    print("── Config ──────────────────────────────────")
    print(json.dumps(response["config"], indent=2))

    print("\n── Validation ──────────────────────────────")
    if response["valid"]:
        print("✅ All column references are valid")
    else:
        print("⚠️  Column reference warnings:")
        for w in response["warnings"]:
            print(f"   • {w}")
        if response["invalid_refs"]:
            print(f"\n   Invalid refs: {json.dumps(response['invalid_refs'], indent=2)}")


if __name__ == "__main__":
    main()
