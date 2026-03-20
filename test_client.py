#!/usr/bin/env python3
"""
CanvasXpress MCP Test Client — Python
======================================
Usage:
    # Run built-in examples (generate + modify)
    python test_client.py --examples

    # Generate a config from a description
    python test_client.py "Violin plot by cell type"
    python test_client.py "Heatmap" "Gene,Sample1,Treatment"
    python test_client.py "Scatter plot" "Gene,Expr,Treatment" '{"Gene":"string","Expr":"numeric","Treatment":"factor"}'
    python test_client.py "Heatmap" '[["Gene","S1","S2"],["BRCA1",1.2,3.4]]' '{"Gene":"string","S1":"numeric","S2":"numeric"}'

    # Modify an existing config
    python test_client.py --modify '{"graphType":"Bar","xAxis":["Gene"]}' "add a title My Chart and switch to dark theme"
    python test_client.py --modify '{"graphType":"Heatmap","xAxis":["Gene"]}' "change colorScheme to Spectral"

Requirements:
    pip install httpx
"""

import json
import sys
import os
import httpx

MCP_URL = os.environ.get("MCP_URL", "http://localhost:8100/mcp")

GENERATE_EXAMPLES = [
    {
        "label": "Clustered heatmap",
        "description": "Clustered heatmap with RdBu colors and dendrograms on both axes",
        "data": [
            ["Gene",  "Control1", "Control2", "Drug1", "Drug2"],
            ["BRCA1", 2.1,        0.9,        3.8,     3.2   ],
            ["TP53",  1.2,        1.4,        0.3,     0.5   ],
            ["EGFR",  0.8,        0.6,        2.9,     3.1   ],
            ["MYC",   3.2,        2.8,        0.4,     0.6   ],
        ],
        "column_types": {
            "Gene": "string", "Control1": "numeric", "Control2": "numeric",
            "Drug1": "numeric", "Drug2": "numeric",
        },
    },
    {
        "label": "Volcano plot",
        "description": "Volcano plot with log2 fold change on x-axis and -log10 p-value on y-axis",
        "data": [
            ["Gene",  "log2FC", "negLog10P"],
            ["GeneA",  2.3,      4.1       ],
            ["GeneB", -1.8,      3.7       ],
            ["GeneC",  0.2,      0.4       ],
            ["GeneD",  3.1,      6.2       ],
        ],
        "column_types": {"Gene": "string", "log2FC": "numeric", "negLog10P": "numeric"},
    },
    {
        "label": "Violin plot",
        "description": "Violin plot of gene expression grouped by cell type with Tableau colors",
        "headers": ["CellID", "Expression", "CellType"],
        "column_types": {"CellID": "string", "Expression": "numeric", "CellType": "factor"},
    },
    {
        "label": "PCA scatter plot",
        "description": "PCA scatter plot with PC1 vs PC2 colored by Treatment with regression ellipses",
        "headers": ["Sample", "PC1", "PC2", "Treatment"],
        "column_types": {"Sample": "string", "PC1": "numeric", "PC2": "numeric", "Treatment": "factor"},
    },
    {
        "label": "Kaplan-Meier survival curve",
        "description": "Kaplan-Meier survival curve for two treatment groups",
        "headers": ["Patient", "Time", "Event", "Treatment"],
        "column_types": {"Patient": "string", "Time": "numeric", "Event": "numeric", "Treatment": "factor"},
    },
    {
        "label": "Stacked percent bar",
        "description": "Stacked percent bar chart of market share by year and company",
        "data": [
            ["Company", "Y2021", "Y2022", "Y2023"],
            ["Alpha",    35,      28,      31     ],
            ["Beta",     28,      33,      29     ],
            ["Gamma",    37,      39,      40     ],
        ],
        "column_types": {"Company": "string", "Y2021": "numeric", "Y2022": "numeric", "Y2023": "numeric"},
    },
    {
        "label": "Ridgeline density",
        "description": "Ridgeline density plot of expression values by cell population",
        "headers": ["Cell", "Value", "Population"],
        "column_types": {"Cell": "string", "Value": "numeric", "Population": "factor"},
    },
    {
        "label": "Sankey flow diagram",
        "description": "Sankey diagram showing patient flow from diagnosis through treatment to outcome",
        "headers": ["Diagnosis", "Treatment", "Outcome"],
        "column_types": {"Diagnosis": "factor", "Treatment": "factor", "Outcome": "factor"},
    },
]

MODIFY_EXAMPLES = [
    {
        "label": "Add title and switch theme",
        "start_config": {
            "graphType": "Heatmap",
            "xAxis": ["Gene"],
            "samplesClustered": True,
            "variablesClustered": True,
            "colorScheme": "RdBu",
        },
        "instruction": "add a title Expression Heatmap and switch to dark theme",
    },
    {
        "label": "Change color scheme and add title",
        "start_config": {
            "graphType": "Bar",
            "xAxis": ["Region"],
            "graphOrientation": "horizontal",
        },
        "instruction": "change the color scheme to Tableau and add a title Regional Sales",
    },
    {
        "label": "Remove legend and set axis titles",
        "start_config": {
            "graphType": "Scatter2D",
            "xAxis": ["PC1"],
            "yAxis": ["PC2"],
            "colorBy": "Treatment",
            "showLegend": True,
        },
        "instruction": "remove the legend and set xAxisTitle to PC1 (32%) and yAxisTitle to PC2 (18%)",
    },
    {
        "label": "Add grouping and jitter",
        "start_config": {
            "graphType": "Boxplot",
            "xAxis": ["Expression"],
        },
        "instruction": "add groupingFactors for the CellType column and enable jitter on the data points",
    },
]

SEP  = "─" * 50
SEP2 = "═" * 50


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


def _post_mcp(client: httpx.Client, session_id: str | None, payload: dict) -> dict:
    hdrs = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    if session_id:
        hdrs["Mcp-Session-Id"] = session_id
    response = client.post(MCP_URL, json=payload, headers=hdrs)
    new_sid = response.headers.get("mcp-session-id")
    body = response.text.strip()
    if "text/event-stream" in response.headers.get("content-type", "") or "data:" in body:
        json_str = _extract_sse_json(body)
        if not json_str:
            raise ValueError(f"Could not extract JSON from SSE:\n{body[:300]}")
        result = json.loads(json_str)
    elif body.startswith("{"):
        result = json.loads(body)
    elif response.status_code == 202 and not body:
        return {"_session_id": new_sid} if new_sid else {}
    else:
        raise ValueError(f"Unexpected response ({response.status_code}):\n{body[:300]}")
    if new_sid:
        result["_session_id"] = new_sid
    return result


def _connect(client: httpx.Client) -> str | None:
    init = _post_mcp(client, None, {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "python-client", "version": "1.0.0"},
        },
    })
    if "error" in init:
        raise RuntimeError(f"Initialize failed: {init['error']}")
    sid = init.get("_session_id")
    _post_mcp(client, sid, {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
    return sid


def _call_tool(client: httpx.Client, sid: str | None, tool: str, arguments: dict) -> dict:
    result = _post_mcp(client, sid, {
        "jsonrpc": "2.0", "id": 2, "method": "tools/call",
        "params": {"name": tool, "arguments": arguments},
    })
    if "error" in result:
        raise RuntimeError(f"Tool call failed: {result['error']}")
    return json.loads(result["result"]["content"][0]["text"])


def _make_client():
    return httpx.Client(timeout=httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=5.0))


def generate_config(description, headers=None, data=None, column_types=None):
    with _make_client() as client:
        sid = _connect(client)
        print(f"Connected  : {MCP_URL}")
        args = {"description": description}
        if data is not None:   args["data"] = data
        elif headers:          args["headers"] = headers
        if column_types:       args["column_types"] = column_types
        return _call_tool(client, sid, "generate_canvasxpress_config", args)


def modify_config(config, instruction, headers=None, data=None, column_types=None):
    with _make_client() as client:
        sid = _connect(client)
        print(f"Connected  : {MCP_URL}")
        args = {"config": config, "instruction": instruction}
        if data is not None:   args["data"] = data
        elif headers:          args["headers"] = headers
        if column_types:       args["column_types"] = column_types
        return _call_tool(client, sid, "modify_canvasxpress_config", args)


def _print_generate_result(response):
    if response.get("headers_used"):
        print(f"Headers used : {', '.join(response['headers_used'])}")
    if response.get("types_used"):
        print(f"Types used   : {', '.join(f'{k}={v}' for k, v in response['types_used'].items())}")
    print()
    print(f"── Config {SEP}")
    print(json.dumps(response["config"], indent=2))
    print(f"\n── Validation {SEP}")
    if response["valid"]:
        print("✅ All column references are valid")
    else:
        print("⚠️  Column reference warnings:")
        for w in response["warnings"]:
            print(f"   • {w}")
        if response.get("invalid_refs"):
            print(f"\n   Invalid refs: {json.dumps(response['invalid_refs'], indent=2)}")


def _print_modify_result(original, response, instruction):
    changes = response.get("changes", {})
    print(f"── Changes {SEP}")
    print(f"   Instruction : {instruction}")
    print(f"   Added       : {changes.get('added')   or 'none'}")
    print(f"   Removed     : {changes.get('removed') or 'none'}")
    print(f"   Changed     : {changes.get('changed') or 'none'}")
    print(f"\n── Modified config {SEP}")
    print(json.dumps(response["config"], indent=2))
    print(f"\n── Validation {SEP}")
    if response["valid"]:
        print("✅ All column references are valid")
    else:
        print("⚠️  Column reference warnings:")
        for w in response["warnings"]:
            print(f"   • {w}")


def run_examples():
    print(f"\n{SEP2}")
    print("  CanvasXpress MCP — Built-in Examples")
    print(f"  Server : {MCP_URL}")
    print(SEP2)

    print(f"\n{SEP}\n  GENERATE EXAMPLES\n{SEP}")
    for i, ex in enumerate(GENERATE_EXAMPLES, 1):
        print(f"\n[{i}/{len(GENERATE_EXAMPLES)}] {ex['label']}")
        print(f"  Description : {ex['description']}")
        if "data" in ex:
            rows = ex["data"]
            print(f"  Data        : {len(rows)-1} rows × {len(rows[0])} columns  "
                  f"({', '.join(str(h) for h in rows[0])})")
        elif "headers" in ex:
            print(f"  Headers     : {', '.join(ex['headers'])}")
        print()
        try:
            response = generate_config(
                description=ex["description"],
                headers=ex.get("headers"),
                data=ex.get("data"),
                column_types=ex.get("column_types"),
            )
            _print_generate_result(response)
        except Exception as e:
            print(f"  ❌ Error: {e}")
        if i < len(GENERATE_EXAMPLES):
            print(f"\n{SEP}")

    print(f"\n\n{SEP}\n  MODIFY EXAMPLES\n{SEP}")
    for i, ex in enumerate(MODIFY_EXAMPLES, 1):
        print(f"\n[{i}/{len(MODIFY_EXAMPLES)}] {ex['label']}")
        print(f"  Instruction  : {ex['instruction']}")
        print(f"  Start config : {json.dumps(ex['start_config'])}")
        print()
        try:
            response = modify_config(
                config=ex["start_config"],
                instruction=ex["instruction"],
            )
            _print_modify_result(ex["start_config"], response, ex["instruction"])
        except Exception as e:
            print(f"  ❌ Error: {e}")
        if i < len(MODIFY_EXAMPLES):
            print(f"\n{SEP}")

    print(f"\n{SEP2}\n")


def main():
    args = sys.argv[1:]

    if args and args[0] == "--examples":
        run_examples()
        return

    if args and args[0] == "--modify":
        if len(args) < 3:
            print("Usage: python test_client.py --modify '<config_json>' '<instruction>'", file=sys.stderr)
            sys.exit(1)
        original = json.loads(args[1])
        instruction = args[2]
        headers = data = column_types = None
        for arg in args[3:]:
            if arg.startswith("{"):   column_types = json.loads(arg)
            elif arg.startswith("["): data = json.loads(arg)
            else:                     headers = arg.split(",")
        print(f"Tool        : modify_canvasxpress_config")
        print(f"Instruction : {instruction}")
        print(f"Config keys : {list(original.keys())}")
        print()
        try:
            response = modify_config(original, instruction, headers=headers, data=data, column_types=column_types)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        _print_modify_result(original, response, instruction)
        return

    # Default: generate
    description  = args[0] if args else "Clustered heatmap with RdBu colors"
    headers = data = column_types = None
    for arg in args[1:]:
        if arg.startswith("{"):   column_types = json.loads(arg)
        elif arg.startswith("["): data = json.loads(arg)
        else:                     headers = arg.split(",")

    if headers is None and data is None:
        data = [
            ["Gene",  "Sample1", "Sample2", "Treatment"],
            ["BRCA1", 1.2,       3.4,       "Control"  ],
            ["TP53",  2.1,       0.9,       "Treated"  ],
            ["EGFR",  0.8,       2.3,       "Control"  ],
        ]
        column_types = {"Gene": "string", "Sample1": "numeric", "Sample2": "numeric", "Treatment": "factor"}

    print(f"Tool        : generate_canvasxpress_config")
    print(f"Description : {description}")
    if data is not None:
        print(f"Data        : {len(data)-1} rows × {len(data[0])} columns")
        print(f"Columns     : {', '.join(str(h) for h in data[0])}")
    elif headers:
        print(f"Headers     : {', '.join(headers)}")
    if column_types:
        print(f"Types       : {', '.join(f'{k}={v}' for k, v in column_types.items())}")
    print()

    try:
        response = generate_config(description, headers=headers, data=data, column_types=column_types)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    _print_generate_result(response)


if __name__ == "__main__":
    main()
