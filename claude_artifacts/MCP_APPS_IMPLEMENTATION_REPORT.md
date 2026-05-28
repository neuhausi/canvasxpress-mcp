# MCP Apps Implementation Report

## 1. Parent Commit

`90d2fa3` — "Changes for better handling /sellect. Added llm_first"

Branch: `feat/mcp-apps-support` (off `main`)

---

## 2. Spec References Used

| Resource | URL |
|----------|-----|
| MCP Apps Spec (2026-01-26) | https://raw.githubusercontent.com/modelcontextprotocol/ext-apps/main/specification/2026-01-26/apps.mdx |
| MCP Apps Quickstart | https://apps.extensions.modelcontextprotocol.io/api/documents/quickstart.html |
| Basic Vanilla JS server example | https://raw.githubusercontent.com/modelcontextprotocol/ext-apps/main/examples/basic-server-vanillajs/server.ts |
| Vanilla JS HTML example | https://raw.githubusercontent.com/modelcontextprotocol/ext-apps/main/examples/basic-server-vanillajs/mcp-app.html |
| Vanilla JS app client | https://raw.githubusercontent.com/modelcontextprotocol/ext-apps/main/examples/basic-server-vanillajs/src/mcp-app.ts |

---

## 3. Exact Protocol Values

| Value | Field |
|-------|-------|
| `text/html;profile=mcp-app` | MIME type (spec §UI Resource Format) |
| `ui` | `_meta` key for tool–resource linkage (spec §Resource Discovery) |
| `resourceUri` | sub-key under `_meta.ui` pointing to the UI resource |
| `ui://canvasxpress/chart` | full resource URI for this server |
| `ui/initialize` | lifecycle request sent by View to Host on load |
| `ui/notifications/tool-result` | postMessage notification from Host to View with `CallToolResult` |
| `ui/notifications/tool-input` | postMessage notification sent before tool-result |
| `*` | postMessage target origin (spec says to use `*`; host enforces sandbox isolation) |

### postMessage envelope shapes

**View → Host (ui/initialize request):**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "ui/initialize",
  "params": {
    "appCapabilities": { "availableDisplayModes": ["inline"] },
    "clientInfo": { "name": "canvasxpress-chart-view", "version": "1.0.0" },
    "protocolVersion": "2026-01-26"
  }
}
```

**Host → View (ui/notifications/tool-result notification):**
```json
{
  "jsonrpc": "2.0",
  "method": "ui/notifications/tool-result",
  "params": {
    "content": [{ "type": "text", "text": "{...canvasxpress config json...}" }],
    "isError": false
  }
}
```

---

## 4. FastMCP API Path Taken

**FastMCP version:** 3.2.4

**Resource registration:** Decorator form `@mcp.resource(uri, mime_type=..., name=...)`. The `resource()` decorator on `FastMCP` accepts `uri` as the first positional argument plus keyword `mime_type`. This worked directly without needing `add_resource()`.

**Tool `_meta` annotation:** FastMCP's `@mcp.tool(...)` decorator accepts a `meta` keyword argument (type `dict[str, Any] | None`). Internally, `Tool.get_meta()` merges the caller-supplied `meta` dict with FastMCP's own bookkeeping (`fastmcp.tags`, `fastmcp.version`) and the result is exposed as the MCP protocol `_meta` field via `Tool.to_mcp_tool()` (which uses Pydantic alias `_meta` for the `meta` model field).

The correct way to annotate a tool is:
```python
@mcp.tool(
    description="...",
    meta={"ui": {"resourceUri": "ui://canvasxpress/chart"}},
)
```

This produces `_meta = {"ui": {"resourceUri": "ui://canvasxpress/chart"}, "fastmcp": {"tags": []}}` on the wire.

**Deprecated alternative (not used):** The spec mentions a deprecated `_meta["ui/resourceUri"]` flat format that will be removed before GA. We use the current `_meta.ui.resourceUri` nested form exclusively.

---

## 5. CanvasXpress CDN URL

```
https://www.canvasxpress.org/dist/canvasXpress.min.js
```

**Note:** This is an **unversioned** URL. As of implementation date (2026-05-28), version 2026.4.5.1935 was current but no versioned CDN path was publicly available. If a versioned URL becomes available (e.g. `/dist/canvasXpress.2026.4.5.1935.min.js`), it should be substituted for reproducibility.

The CSP `resourceDomains` should include `https://www.canvasxpress.org` for hosts that enforce MCP Apps CSP policies.

---

## 6. Tests Output

```
.........                                                                [100%]
=============================== warnings summary ===============================
.venv/lib/python3.12/.../torch/cuda/__init__.py:180
  UserWarning: CUDA initialization: The NVIDIA driver on your system is too old...
    return torch._C._cuda_getDeviceCount() > 0

tests/test_mcp_apps.py:31
  DeprecationWarning: There is no current event loop
    _tools_list = asyncio.get_event_loop().run_until_complete(_mcp.list_tools())

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
9 passed, 2 warnings in 7.66s
```

All 9 tests pass. The `DeprecationWarning` about event loop is a Python 3.12 deprecation in the test harness (using `asyncio.get_event_loop()` at module import time); the tests are functionally correct and can be fixed in future by switching to `asyncio.run()`.

### Test coverage

| Test | What it verifies |
|------|-----------------|
| `test_resource_is_registered` | `ui://canvasxpress/chart` appears in `mcp.list_resources()` |
| `test_resource_mime_type` | MIME type is `text/html;profile=mcp-app` |
| `test_resource_body_contains_cx_chart_div` | HTML contains `id="cx-chart"` |
| `test_resource_body_contains_canvasxpress_constructor` | HTML contains `new CanvasXpress` |
| `test_chart_tool_has_ui_meta[create_map_config]` | `_meta.ui.resourceUri` present |
| `test_chart_tool_has_ui_meta[generate_canvasxpress_config]` | `_meta.ui.resourceUri` present |
| `test_chart_tool_has_ui_meta[generate_km_config]` | `_meta.ui.resourceUri` present |
| `test_chart_tool_has_ui_meta[modify_canvasxpress_config]` | `_meta.ui.resourceUri` present |
| `test_non_chart_tool_has_no_ui_meta` | `list_chart_types` does NOT have `_meta.ui` |

---

## 7. Smoke Test Confirmation

```
GET http://localhost:8100/generate?description=Bar chart&headers=Gene,Expr&column_types={"Gene":"string","Expr":"numeric"}

Headers used : Gene, Expr
Types used   : Gene=string, Expr=numeric

── Config ──────────────────────────────────────────────────
{
  "graphType": "Bar",
  "xAxis": ["Expr"],
  "smpTitle": "Gene"
}

── Validation ──────────────────────────────────────────────────
✅ All column references are valid
SMOKE_OK=1
```

REST endpoint `/generate` returns the same result as before the changes. All existing integrations are unaffected.

---

## 8. Spec Ambiguities Encountered

1. **`postMessage` target origin:** The spec says to use `'*'` as target origin in the plain-`postMessage` examples (`window.parent.postMessage({...}, '*')`). Security is delegated to the host's sandbox isolation (double-iframe architecture per spec §Sandbox proxy). This is implemented as specified, with a code comment explaining the rationale.

2. **`_meta` key naming:** The spec at §Resource Discovery shows both `_meta.ui.resourceUri` (current) and the deprecated `_meta["ui/resourceUri"]` (flat). We implement only the current nested form. FastMCP's `meta` kwarg directly populates `_meta` on the wire.

3. **Resource in `resources/list`:** The spec says (§Resource Discovery): "Servers MAY omit UI-only resources from `resources/list`". We chose to include it because FastMCP's `@mcp.resource()` automatically adds resources to `resources/list`, and omitting it would require lower-level manipulation. Including it is also correct per spec.

4. **CanvasXpress CDN versioning:** No versioned CDN URL was available in public docs at implementation time. The unversioned URL is used with a comment.

5. **`_meta.fastmcp` key in tool output:** FastMCP 3.2.4 always merges a `fastmcp` key into `_meta` (containing `tags` and optionally `version`). This is a FastMCP internal and should be transparent to MCP Apps hosts which only look for `_meta.ui.resourceUri`.

---

## 9. Deferred Work

- **Real-host verification:** The implementation has not been tested inside a live MCP Apps-supporting host (Claude Desktop, ChatGPT, Goose). Testing requires a host with MCP Apps extension support enabled.
- **Playwright/E2E tests:** End-to-end tests verifying the iframe lifecycle (`ui/initialize` → `ui/notifications/tool-result` → `new CanvasXpress()`) require a browser automation framework.
- **Versioned CDN URL:** Pin CanvasXpress to a specific version once a versioned URL is available.
- **CSP declaration on resource:** The `cx_chart_view.html` loads CanvasXpress from `https://www.canvasxpress.org`. The resource registration could add `_meta.ui.csp.resourceDomains = ["https://www.canvasxpress.org"]` to inform hosts. Currently omitted to keep the first-pass implementation minimal.
- **Event loop deprecation warning in tests:** The tests use `asyncio.get_event_loop().run_until_complete()` at module level, which triggers a Python 3.12 `DeprecationWarning`. This can be fixed by using a `pytest-asyncio` fixture or `asyncio.run()` pattern.

---

## 10. Commits

```
ca8b027 docs(mcp-apps): document MCP Apps inline-chart rendering
81069d2 test(mcp-apps): cover ui:// resource registration and tool _meta annotations
8153762 feat(mcp-apps): annotate 4 chart-producing tools with ui:// _meta
f8008f9 feat(mcp-apps): register ui://canvasxpress/chart resource
fedf6c0 feat(mcp-apps): add cx_chart_view.html for MCP Apps iframe rendering
```

---

## 11. Files Changed

```
 README.md                 |  46 ++++-
 src/server.py             |  45 ++++-
 src/ui/cx_chart_view.html | 206 +++++++++++++++++++++
 tests/__init__.py         |   0
 tests/test_mcp_apps.py    | 180 ++++++++++++++++++
 5 source files, 473 total insertions (+4 deletions)
```
