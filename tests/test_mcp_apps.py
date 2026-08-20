"""
Tests for MCP Apps support in the CanvasXpress MCP server.

All tests are offline, in-process, and complete in < 10 seconds total.

Per MCP Apps spec (2026-01-26):
  - UI resource URI scheme: ui://
  - UI resource MIME type: text/html;profile=mcp-app
  - Tool _meta key: _meta.ui.resourceUri
"""

import asyncio
import sys
import os

import pytest

# ---------------------------------------------------------------------------
# Module-level fixture: import server once, cache mcp + lists
# ---------------------------------------------------------------------------

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# We import server at module level once to avoid repeated expensive init
import server as _server_module

_mcp = _server_module.mcp


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# Pre-fetch tools and resources synchronously (called once at module import)
_tools_list = _run(_mcp.list_tools())
_resources_list = _run(_mcp.list_resources())

# Constants from server
_RESOURCE_URI = _server_module._CX_APP_RESOURCE_URI  # "ui://canvasxpress/chart"
_MCP_APPS_MIME = "text/html;profile=mcp-app"

# The 4 tools that must carry ui._meta
_CHART_TOOLS = {
    "generate_canvasxpress_config",
    "modify_canvasxpress_config",
    "generate_km_config",
    "create_map_config",
}

# A tool that must NOT carry the ui._meta (out-of-scope sanity check)
_NON_CHART_TOOL = "list_chart_types"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_tool(name: str):
    for t in _tools_list:
        if t.name == name:
            return t
    return None


def _get_resource(uri: str):
    for r in _resources_list:
        if str(r.uri) == uri:
            return r
    return None


def _tool_mcp_meta(tool) -> dict:
    """Return the _meta dict as serialized by to_mcp_tool()."""
    mcp_tool = tool.to_mcp_tool()
    return mcp_tool.model_dump(by_alias=True, exclude_none=True).get("_meta", {})


# ---------------------------------------------------------------------------
# Test 1: Resource ui://canvasxpress/chart is registered
# ---------------------------------------------------------------------------

def test_resource_is_registered():
    """ui://canvasxpress/chart resource must appear in resources list."""
    r = _get_resource(_RESOURCE_URI)
    assert r is not None, (
        f"Resource '{_RESOURCE_URI}' not found in registered resources. "
        f"Found: {[str(x.uri) for x in _resources_list]}"
    )


# ---------------------------------------------------------------------------
# Test 2: Resource MIME type equals spec value
# ---------------------------------------------------------------------------

def test_resource_mime_type():
    """Resource mimeType must be 'text/html;profile=mcp-app' per spec §UI Resource Format."""
    r = _get_resource(_RESOURCE_URI)
    assert r is not None, f"Resource '{_RESOURCE_URI}' not registered"
    assert r.mime_type == _MCP_APPS_MIME, (
        f"Expected mimeType '{_MCP_APPS_MIME}', got '{r.mime_type}'"
    )


# ---------------------------------------------------------------------------
# Test 3: Resource body contains required HTML elements
# ---------------------------------------------------------------------------

def test_resource_body_contains_cx_chart_div():
    """Resource HTML must contain <canvas id=\"cx-chart\"> (the CanvasXpress target)."""
    result = _run(
        _mcp.read_resource(_RESOURCE_URI)
    )
    body = result.contents[0].content
    assert 'id="cx-chart"' in body or "id='cx-chart'" in body, (
        "Resource HTML must contain element with id='cx-chart'"
    )


def test_resource_body_contains_canvasxpress_constructor():
    """Resource HTML must reference 'new CanvasXpress' to instantiate the chart."""
    result = _run(
        _mcp.read_resource(_RESOURCE_URI)
    )
    body = result.contents[0].content
    assert "new CanvasXpress" in body, (
        "Resource HTML must contain 'new CanvasXpress' constructor call"
    )


# ---------------------------------------------------------------------------
# Test 4: Each of the 4 chart tools has _meta.ui.resourceUri
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tool_name", sorted(_CHART_TOOLS))
def test_chart_tool_has_ui_meta(tool_name: str):
    """
    Each chart-producing tool must have _meta.ui.resourceUri pointing to
    the ui://canvasxpress/chart resource.
    Per spec §Resource Discovery (2026-01-26): tools reference UI resources
    via _meta.ui.resourceUri.
    """
    tool = _get_tool(tool_name)
    assert tool is not None, f"Tool '{tool_name}' not found in tools list"

    # Check FastMCP-level meta
    assert tool.meta is not None, f"Tool '{tool_name}' has no meta set"
    assert "ui" in tool.meta, f"Tool '{tool_name}' meta has no 'ui' key"
    assert tool.meta["ui"].get("resourceUri") == _RESOURCE_URI, (
        f"Tool '{tool_name}' meta.ui.resourceUri = {tool.meta['ui'].get('resourceUri')!r}, "
        f"expected {_RESOURCE_URI!r}"
    )

    # Also verify it propagates to the MCP wire format
    meta = _tool_mcp_meta(tool)
    assert "ui" in meta, f"MCP-level _meta for '{tool_name}' has no 'ui' key"
    assert meta["ui"].get("resourceUri") == _RESOURCE_URI, (
        f"MCP-level _meta.ui.resourceUri for '{tool_name}' = "
        f"{meta['ui'].get('resourceUri')!r}, expected {_RESOURCE_URI!r}"
    )


# ---------------------------------------------------------------------------
# Test 5: An out-of-scope tool does NOT have ui._meta
# ---------------------------------------------------------------------------

def test_non_chart_tool_has_no_ui_meta():
    """
    list_chart_types should NOT have _meta.ui.resourceUri — it is out of scope
    for MCP Apps rendering.
    """
    tool = _get_tool(_NON_CHART_TOOL)
    assert tool is not None, f"Tool '{_NON_CHART_TOOL}' not found (needed as negative fixture)"

    # FastMCP-level meta should not have a 'ui' key
    if tool.meta:
        assert "ui" not in tool.meta, (
            f"Tool '{_NON_CHART_TOOL}' unexpectedly has 'ui' key in meta"
        )

    # MCP wire format _meta should not have a 'ui' key
    meta = _tool_mcp_meta(tool)
    assert "ui" not in meta, (
        f"Tool '{_NON_CHART_TOOL}' unexpectedly has 'ui' key in MCP-level _meta"
    )


# ---------------------------------------------------------------------------
# Test 6: Resource _meta declares CSP resourceDomains
# ---------------------------------------------------------------------------

def test_resource_has_csp_resource_domains():
    """Resource _meta.ui.csp.resourceDomains must include the CanvasXpress CDN origin
    on the resources/list registration (Resource.meta)."""
    r = _get_resource(_RESOURCE_URI)
    assert r is not None, f"Resource '{_RESOURCE_URI}' not registered"
    assert r.meta is not None, "Resource has no meta"
    ui = r.meta.get("ui")
    assert ui is not None, "Resource meta has no 'ui' key"
    csp = ui.get("csp")
    assert csp is not None, "Resource meta.ui has no 'csp' key"
    domains = csp.get("resourceDomains")
    assert isinstance(domains, list), "resourceDomains must be a list"
    assert "https://www.canvasxpress.org" in domains, (
        f"Expected 'https://www.canvasxpress.org' in resourceDomains, got {domains}"
    )


def test_resource_read_response_has_csp_resource_domains():
    """Per MCP Apps spec lines 243-250 / 303-314, _meta.ui.csp belongs on the
    resources/read response content. Verify it is present on the content item
    returned by FastMCP's read pipeline (not just on the resources/list entry)."""
    resource_obj = _get_resource(_RESOURCE_URI)
    assert resource_obj is not None
    # _read_resource_mcp returns the wire-format mcp.types.ReadResourceResult,
    # which is what hosts receive over the protocol.
    result = _run(_mcp._read_resource_mcp(_RESOURCE_URI))
    assert result.contents, "resources/read returned no contents"
    first = result.contents[0]
    meta = getattr(first, "meta", None)
    assert meta is not None, (
        "resources/read content[0] has no _meta — hosts that read CSP from the "
        "read response (per spec lines 243-250) will not see resourceDomains"
    )
    domains = meta.get("ui", {}).get("csp", {}).get("resourceDomains")
    assert isinstance(domains, list) and "https://www.canvasxpress.org" in domains, (
        f"Expected _meta.ui.csp.resourceDomains to include CDN origin in read response, "
        f"got {meta!r}"
    )


# ---------------------------------------------------------------------------
# Chart tools must echo `data` in their return dicts so the MCP App iframe
# can render the chart from the tool result alone (per CanvasXpress object-form
# constructor: new CanvasXpress({renderTo, config, data}).
# ---------------------------------------------------------------------------

# Use create_map_config — no LLM, fully deterministic.
_create_map_config = _server_module.create_map_config


def test_chart_tool_result_includes_data_when_provided():
    """When `data` is passed in, the tool result must echo it back so the
    iframe can render via `new CanvasXpress({renderTo, config, data})`."""
    data = [
        ["Country", "GDP"],
        ["USA", 21000],
        ["CAN",  1800],
        ["MEX",  1200],
    ]
    result = _create_map_config(map_id="World", data=data, title="GDP")
    assert "data" in result, (
        "Chart tool result missing 'data' key — MCP App iframe cannot render "
        "without it. See src/ui/cx_chart_view.html handleToolResult()."
    )
    assert result["data"] == data, (
        f"Expected echoed data to equal input, got {result['data']!r}"
    )
    assert result["config"].get("graphType"), "Result missing config.graphType"


def test_chart_tool_result_data_is_none_when_omitted():
    """When `data` is NOT passed, the tool result must explicitly carry
    `data: None` so the iframe can surface a friendly 'data required' message
    instead of silently rendering an empty chart."""
    result = _create_map_config(map_id="World", title="Empty map")
    assert "data" in result, "Result must always include the 'data' key"
    assert result["data"] is None, (
        f"Expected data to be None when omitted, got {result['data']!r}"
    )
