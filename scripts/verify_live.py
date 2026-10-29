#!/usr/bin/env python3
"""Check a running canvasxpress-mcp endpoint the way a client sees it, and screenshot what the
dashboard view draws from the endpoint's own tool result.

    .venv/bin/python scripts/verify_live.py [https://canvasxpress.org/mcp] [out.png]

Needs fastmcp; the screenshot also needs playwright (`pip install playwright && playwright install chromium`).
The "host" is a stand-in page that answers the view's ui/initialize and delivers the tool result.
This shows what the SERVER sends and what the VIEW draws; it cannot show what a given chat client
(claude.ai etc.) chooses to display.
"""
import asyncio
import json
import sys

URL = sys.argv[1] if len(sys.argv) > 1 else "https://canvasxpress.org/mcp"
OUT = sys.argv[2] if len(sys.argv) > 2 else "verify_live.png"
HOST = """<body style="margin:0"><iframe id=f style="width:700px;height:1000px;border:0"></iframe><script>
addEventListener('message',function(e){var m=e.data;if(!m||!m.method)return;
 if(m.method==='ui/initialize'){e.source.postMessage({jsonrpc:'2.0',id:m.id,result:{}},'*');
  setTimeout(function(){e.source.postMessage({method:'ui/notifications/tool-result',
   params:{content:[{type:'text',text:window.PAYLOAD}]}},'*')},500);}});</script></body>"""


async def fetch():
    from fastmcp import Client
    async with Client(URL) as c:
        tools = {t.name: t for t in await c.list_tools()}
        resources = [str(r.uri) for r in await c.list_resources()]
        view = (await c.read_resource("ui://canvasxpress/dashboard"))[0].text
        result = await c.call_tool("suggest_canvasxpress_dashboard", {"dataset": "penguins"})
        return tools, resources, view, result.content[0].text


def main():
    tools, resources, view, payload = asyncio.run(fetch())
    result = json.loads(payload)
    desc = tools["suggest_canvasxpress_dashboard"].description
    checks = [
        ("resources include ui://canvasxpress/dashboard", "ui://canvasxpress/dashboard" in resources),
        ("suggest tool points at the dashboard view",
         (tools["suggest_canvasxpress_dashboard"].meta or {}).get("ui", {}).get("resourceUri")
         == "ui://canvasxpress/dashboard"),
        ("view has Save + Open in Builder", "cx-save" in view and "cx-builder" in view),
        ("result has no render_inline recipe", "render_inline" not in result),
        ("result has builder_url", bool(result.get("builder_url"))),
        ("filters panel has no scheme box (saveSchemes false)",
         result["spec"]["panels"]["filters"].get("saveSchemes") is False),
        ("theme is auto", result["spec"].get("theme") == "auto"),
        ("description says not to redraw in a widget", "do not redraw" in desc),
    ]
    for label, ok in checks:
        print("PASS" if ok else "FAIL", label)
    print("panel titles:", [p["title"] for p in result["panels"]])
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("(playwright not installed: no screenshot)")
        return
    try:
        browser = sync_playwright().start().chromium.launch()
    except Exception as exc:  # browsers not installed
        print('(no screenshot: %s)' % str(exc).splitlines()[0])
        return
    with browser:
        page = browser.new_page(viewport={"width": 700, "height": 1000})
        page.set_content(HOST)
        page.evaluate("t=>{window.PAYLOAD=t}", payload)
        page.evaluate("v=>{document.getElementById('f').srcdoc=v}", view)
        page.wait_for_timeout(8000)
        page.screenshot(path=OUT)
    print("screenshot:", OUT)


main()
