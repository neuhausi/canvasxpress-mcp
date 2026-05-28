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

# Pre-fetch tools and resources synchronously (called once at module import)
_tools_list = asyncio.get_event_loop().run_until_complete(_mcp.list_tools())
_resources_list = asyncio.get_event_loop().run_until_complete(_mcp.list_resources())

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
    result = asyncio.get_event_loop().run_until_complete(
        _mcp.read_resource(_RESOURCE_URI)
    )
    body = result.contents[0].content
    assert 'id="cx-chart"' in body or "id='cx-chart'" in body, (
        "Resource HTML must contain element with id='cx-chart'"
    )


def test_resource_body_contains_canvasxpress_constructor():
    """Resource HTML must reference 'new CanvasXpress' to instantiate the chart."""
    result = asyncio.get_event_loop().run_until_complete(
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
