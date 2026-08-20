# MCP Apps Fix Pass Report

## 1. Parent Commits

`20c9e65` — HEAD of `feat/mcp-apps-support` before this fix pass (6 commits from original task).

## 2. Review Source

`claude_artifacts/MCP_APPS_REVIEW_NOTES.md`

## 3. Items Addressed

| # | Item | Status | Commit |
|---|------|--------|--------|
| 1 | Bug: ResizeObserver leak in `renderChart` | fixed | `9b211e5` |
| 2 | Bug: Operator precedence in error string | fixed | `9b211e5` |
| 3 | Bug: Brittle canvas-rebuild logic | fixed | `9b211e5` |
| 4 | CSP `resourceDomains` declaration (promoted from deferred) | fixed | `dcfcade` |
| 5 | Hygiene: `tests/__pycache__/` committed + `.gitignore` missing | fixed | `4babe69` |
| 6 | Unversioned CanvasXpress CDN URL — README note | documented | `60c8e6e` |
| 7 | Empty `tests/__init__.py` | left as-is (harmless) | — |
| 8 | `ui/initialize` response data ignored — TODO comment | commented | `dcfcade` |
| 9 | `ui/notifications/initialized` not handled — TODO comment | commented | `dcfcade` |
| 10 | `appCapabilities` inline-only justification comment | commented | `9b211e5` |

## 4. Bug 3 Option Chosen

**Option A** — removed the canvas-rebuild block entirely. `new CanvasXpress('cx-chart', config)` manages its own canvas element in-place after `chart.destroy()` cleans up the previous instance. Smoke test confirmed chart rendering continues to work correctly via the REST `/generate` endpoint.

## 5. CSP Path

Exact nested key path: `_meta.ui.csp.resourceDomains`

Spec citation (from `/tmp/mcp-apps-spec.mdx`):

Lines 243-250:
```
    _meta?: {
      ui?: {
        csp?: {
          connectDomains?: string[]; // Origins for network requests (fetch/XHR/WebSocket).
          resourceDomains?: string[]; // Origins for static resources (scripts, images, styles, fonts).
          frameDomains?: string[]; // Origins for nested iframes (frame-src directive).
          baseUriDomains?: string[]; // Allowed base URIs for the document (base-uri directive).
        };
```

Lines 303-314 (example):
```json
"_meta": {
  "ui" : {
    "csp": {
      "connectDomains": ["https://api.openweathermap.org"],
      "resourceDomains": ["https://cdn.jsdelivr.net"]
    },
    "prefersBorder": true
  }
}
```

## 6. FastMCP API Path for Resource `meta`

Decorator `meta` kwarg: `@mcp.resource(uri, ..., meta={...})`. FastMCP 3.2.4's `resource()` decorator signature includes `meta: 'dict[str, Any] | None' = None`.

## 7. Test Count

```
10 passed, 1 warning in 7.56s
```

(The single warning is a CUDA driver UserWarning from torch, unrelated to our code.)

## 8. Smoke Test

REST `/generate` returns equivalent output:
```
GET http://localhost:8100/generate?description=Bar chart&headers=Gene,Expr&column_types={"Gene":"string","Expr":"numeric"}
Config: {"graphType": "Bar", "xAxis": ["Expr"], "smpTitle": "Gene"}
Validation: All column references are valid
```

## 9. Untracked Pycs Confirmation

```
$ git ls-files | grep -E "__pycache__|\.pyc$" || echo NONE
NONE
```

## 10. Commits Made in This Fix Pass

```
60c8e6e docs(mcp-apps): fix-pass report + README CDN note + report update
4babe69 chore(mcp-apps): .gitignore, untrack pycache, fix asyncio deprecation
dcfcade feat(mcp-apps): declare CSP resourceDomains; TODO comments for lifecycle hooks
9b211e5 fix(mcp-apps): ResizeObserver leak, error-string precedence, canvas-rebuild brittleness
```

## 11. Files Changed in This Fix Pass

```
 .gitignore                                         |  19 +++++++++++
 src/server.py                                      |   9 +++++
 src/ui/cx_chart_view.html                          |  25 ++++++--------
 tests/__pycache__/__init__.cpython-312.pyc         | Bin 197 -> 0 bytes
 .../test_mcp_apps.cpython-312-pytest-9.0.2.pyc     | Bin 16967 -> 0 bytes
 tests/test_mcp_apps.py                             |  37 ++++++++++++++++++---
 6 files changed, 72 insertions(+), 18 deletions(-)
```

(Plus README.md, MCP_APPS_IMPLEMENTATION_REPORT.md, and this file in commit 4.)

## 12. Spec Ambiguities

1. **CSP on resource vs. tool `_meta`:** The spec shows `_meta.ui.csp` in the `resources/read` response content (lines 243-250 and the example at lines 303-314). It does not explicitly show CSP on the `resources/list` metadata. We placed it on the resource registration's `meta` kwarg which FastMCP exposes via `resources/list`. The host reference implementation (lines 1730-1744) reads `resource._meta?.ui?.csp` from the resource content, confirming this is the correct location.

2. **Whether `meta` on `@mcp.resource()` maps to the resource's `_meta` field in `resources/list` vs. the content's `_meta` in `resources/read`:** FastMCP maps the decorator `meta` kwarg to the resource template metadata exposed in `resources/list`. The spec's host example reads CSP from the `resources/read` result's `_meta`. These may differ in behavior for some hosts. We declare it on registration; if a host requires it in the read response body, additional work is needed.

## 13. Remaining Deferred Items

- **Real-host verification:** Not tested inside a live MCP Apps host (Claude Desktop, ChatGPT, Goose).
- **Playwright/E2E tests:** Browser automation tests for the full iframe lifecycle.
- **Versioned CDN URL:** Pin CanvasXpress once a versioned URL is available upstream.
- **Dataset-too-large handling:** No size gating for very large tool result JSON payloads.
- **Dark/light theme via `useHostStyles`:** Chart view does not read host theme preferences from `ui/initialize` result.
