"""Dashboard-spec composition for the canvasxpress-mcp server.

Deterministic (no LLM call): assembles a canvasxpress-dashboards spec from a
list of panels — each a CanvasXpress ``{data, config}`` pair, typically produced
by ``generate_canvasxpress_config`` — plus optional text. The result renders
with the ``canvasxpress-dashboards`` npm package (``renderDashboard``), whose
engine-level ``broadcast`` mechanism links selections across every panel
automatically.

Pure functions; no server state. Validation is the vendored
``cx_dashboard_validate.validate_spec``.
"""

from __future__ import annotations

import os
import json
import re
from typing import Any, Optional

import cx_dashboard_suggest
import cx_datasets
from cx_dashboard_validate import validate_spec, DASHBOARD_SCHEMA_VERSION

_GRID_COLS = 12          # default spec grid
_PANEL_H = 3             # rows per chart panel (rowHeight 130 => ~390px tall)
_TEXT_H = 1              # rows for the intro text panel


def _slug(text: str, fallback: str) -> str:
    """A spec-safe id from free text (lowercase, dashes, no punctuation)."""
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return slug or fallback


def compose_dashboard(
    title: str,
    panels: list,
    description: Optional[str] = None,
    columns: int = 2,
    theme: Optional[str] = None,
) -> dict:
    """Assemble a dashboard spec from chart panels.

    Args:
        title: dashboard title (also derives the spec id).
        panels: list of ``{data, config, title?, width?}`` — ``data``/``config``
            are a CanvasXpress data object and config; ``width`` is an optional
            panel width in grid columns (1..12, default = 12 // columns).
        description: optional intro text rendered as a full-width text panel.
        columns: chart panels per row (1..4; default 2).
        theme: optional spec theme ("light", "dark", or "auto").

    Returns:
        {"spec": dict, "valid": bool, "errors": [...], "usage": str}
    """
    problems: list = []
    if not isinstance(title, str) or not title.strip():
        problems.append("title must be a non-empty string")
    if not isinstance(panels, list) or not panels:
        problems.append("panels must be a non-empty list of {data, config} objects")
    if problems:
        return {"spec": None, "valid": False, "errors": problems, "usage": ""}

    columns = columns if isinstance(columns, int) and 1 <= columns <= 4 else 2
    default_w = _GRID_COLS // columns
    dash_id = _slug(title, "dashboard")

    spec: dict = {
        "id": dash_id,
        "title": title.strip(),
        "schemaVersion": DASHBOARD_SCHEMA_VERSION,
        "version": 1,
        "broadcastGroup": dash_id,
        "layout": {"cols": _GRID_COLS, "rowHeight": 130, "gap": 12, "items": []},
        "data": {},
        "panels": {},
    }
    if theme in ("light", "dark", "auto"):
        spec["theme"] = theme

    y = 0
    if description:
        spec["panels"]["intro"] = {"type": "text", "html": _intro_html(title, description)}
        spec["layout"]["items"].append(
            {"panel": "intro", "x": 0, "y": 0, "w": _GRID_COLS, "h": _TEXT_H})
        y = _TEXT_H

    x = 0
    seen_data: dict = {}   # serialized data -> dataRef (dedupe shared datasets)
    for i, panel in enumerate(panels):
        if not isinstance(panel, dict) or not isinstance(panel.get("config"), dict) \
                or panel.get("data") is None:
            problems.append("panels[%d] must be an object with data and config" % i)
            continue

        panel_title = panel.get("title") or panel["config"].get("title") \
            or panel["config"].get("graphType") or ("Panel %d" % (i + 1))
        panel_id = _slug(str(panel_title), "panel-%d" % (i + 1))
        if panel_id in spec["panels"]:
            panel_id = "%s-%d" % (panel_id, i + 1)

        # Identical data objects share one source, so a selection in one panel
        # broadcasts to the others over the same rows.
        panel_data = panel["data"]
        if isinstance(panel_data, list):
            # A headers+rows table (as generate_canvasxpress_config takes it) becomes a data object.
            try:
                panel_data = cx_datasets.table_to_data(panel_data)
            except ValueError as exc:
                problems.append("panels[%d].data: %s" % (i, exc))
                continue
        data_key = json.dumps(panel_data, sort_keys=True, default=str)
        data_ref = seen_data.get(data_key)
        if data_ref is None:
            data_ref = panel_id if panel_id not in spec["data"] else "data-%d" % (len(seen_data) + 1)
            seen_data[data_key] = data_ref
            spec["data"][data_ref] = {"kind": "inline", "value": panel_data}

        config = dict(panel["config"])
        config.pop("title", None)  # the panel header shows the title
        # generate_canvasxpress_config sets the engine's transposeData for a row-per-sample
        # scatter, which is right standalone; the dashboard renderer flips such a panel itself,
        # so keeping the flag flips it twice and the chart draws blank or as noise.
        config = cx_dashboard_suggest.panel_config_for(config, panel_data)
        spec["panels"][panel_id] = {
            "title": str(panel_title),
            "dataRef": data_ref,
            "config": config,
        }

        w = panel.get("width")
        w = w if isinstance(w, int) and 1 <= w <= _GRID_COLS else default_w
        if x + w > _GRID_COLS:
            x = 0
            y += _PANEL_H
        spec["layout"]["items"].append(
            {"panel": panel_id, "x": x, "y": y, "w": w, "h": _PANEL_H})
        x += w
        if x >= _GRID_COLS:
            x = 0
            y += _PANEL_H

    if problems:
        return {"spec": None, "valid": False, "errors": problems, "usage": ""}

    result = validate_spec(spec)
    out = {"spec": spec, "valid": result["valid"], "errors": result["errors"],
           "usage": _usage_snippet(dash_id),
           "render_inline": inline_render_guide(),
           "builder_url": builder_base_url()}
    if result.get("warnings"):
        out["warnings"] = result["warnings"]
    return out


def _intro_html(title: str, description: str) -> str:
    """The intro text panel's HTML (title line + muted description)."""
    esc = (lambda s: s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    return ('<div style="font-weight: 700; font-size: 25px">%s</div>'
            '<div style="font-size: 16px; color: var(--cxd-muted)">%s</div>'
            % (esc(title), esc(description)))


# Pinned builds for the inline-widget fallback (bump with each engine / dashboards release).
INLINE_ENGINE_VERSION = "72.9.1"
INLINE_DASHBOARDS_VERSION = "0.15.0"

_WIDGET_TEMPLATE = (
    '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/canvasxpress@%(engine)s/src/canvasXpress.css">\n'
    '<script src="https://cdn.jsdelivr.net/npm/canvasxpress@%(engine)s/src/canvasXpress.js"></script>\n'
    '<script src="https://cdn.jsdelivr.net/npm/canvasxpress-dashboards@%(dash)s/dist/canvasxpress-dashboards.umd.js"></script>\n'
    '<div id="status" style="font:12px sans-serif;color:#b00"></div>\n'
    '<div id="dashboard" style="width:100%%;min-height:960px"></div>\n'
    '<script>\n'
    'var spec = /*SPEC_JSON*/;\n'
    'try { CanvasXpressDashboards.renderDashboard(spec, "dashboard"); }\n'
    'catch (e) { document.getElementById("status").textContent = "Render error: " + e.message; }\n'
    'window.addEventListener("error", function (ev) {\n'
    '  document.getElementById("status").textContent = "Load error: " + ev.message; });\n'
    '</script>'
) % {"engine": INLINE_ENGINE_VERSION, "dash": INLINE_DASHBOARDS_VERSION}


def builder_base_url() -> str:
    """Where "Open in Builder" goes: the configured install (CXD_DASHBOARDS_URL), else the
    public canvasxpress.org dashboards app."""
    return (os.getenv("CXD_DASHBOARDS_URL") or "").strip().rstrip("/") or "https://canvasxpress.org/dashboards"


def inline_render_guide() -> dict:
    """How an agent shows a returned spec as a live, interactive dashboard inside the chat."""
    return {
        "when": ("The user asked for a dashboard / MCP app / 'show it here'. Render it INLINE as a "
                 "live interactive dashboard. Do not return the spec JSON, do not build a server or "
                 "project, do not publish a separate artifact or file unless asked."),
        "preferred": ("If this client shows the MCP App view (ui://canvasxpress/dashboard) it renders "
                      "from this result automatically. Otherwise use an HTML widget tool "
                      "(e.g. visualize show_widget) as below."),
        "steps": [
            "Take `spec` from this result (never show it to the user).",
            "Fill `widget_template`: replace /*SPEC_JSON*/ with the spec as JSON. Data stays embedded "
            "in the spec (the widget cannot fetch other hosts); use compact data if large.",
            "Widget is ~680px wide: set spec.layout to cols 12, rowHeight 30, gap 10; filters panel "
            "w=12 h=6, then charts in two columns (w=6, h=12).",
            "Keep every chart panel on one dataRef so a selection broadcasts across all panels.",
            "Keep the status line: it shows load/render errors instead of failing silently.",
            "Reply with one or two sentences only.",
        ],
        "widget_template": _WIDGET_TEMPLATE,
    }


def _usage_snippet(dash_id: str) -> str:
    """A ready-to-paste HTML snippet that renders the spec."""
    return (
        "<!-- Save the spec as %s.spec.json next to this page -->\n"
        '<link href="https://www.canvasxpress.org/dist/canvasXpress.css" rel="stylesheet" />\n'
        '<script src="https://www.canvasxpress.org/dist/canvasXpress.min.js"></script>\n'
        '<script src="https://cdn.jsdelivr.net/npm/canvasxpress-dashboards/dist/canvasxpress-dashboards.umd.js"></script>\n'
        '<div id="dashboard"></div>\n'
        "<script>\n"
        "  fetch('%s.spec.json')\n"
        "    .then(function (r) { return r.json(); })\n"
        "    .then(function (spec) { return CanvasXpressDashboards.renderDashboard(spec, 'dashboard'); });\n"
        "</script>" % (dash_id, dash_id)
    )


# ---------------------------------------------------------------------------
# Documentation payload for the explain tool
# ---------------------------------------------------------------------------

_DOCS = {
    "overview": (
        "canvasxpress-dashboards composes, coordinates and renders dashboards — "
        "collections of linked CanvasXpress charts — from one declarative JSON spec. "
        "The CanvasXpress engine itself provides the cross-chart coordination "
        "(selections/filters propagate via its broadcast mechanism); the package adds "
        "the spec, a CSS-grid layout, a renderer, data binding, persistence and a "
        "no-code builder. npm: canvasxpress-dashboards (canvasxpress is a peer "
        "dependency). Docs & demos: https://www.canvasxpress.org/dashboards.html and "
        "https://github.com/neuhausi/canvasxpress-dashboards"
    ),
    "install": (
        "npm install canvasxpress-dashboards canvasxpress — then "
        "import { renderDashboard } from 'canvasxpress-dashboards' and call "
        "renderDashboard(spec, element). Or use the UMD <script> drop-in from "
        "cdn.jsdelivr.net/npm/canvasxpress-dashboards and call "
        "CanvasXpressDashboards.renderDashboard(spec, 'element-id'). For the full "
        "self-hosted app (sign-in, builder, saved dashboards, share links): "
        "docker compose up in the repo, or `cd server && pip install -e '.[web]'`."
    ),
    "spec": (
        "A spec is one JSON object: id, title, schemaVersion ('%s'), layout "
        "({cols, rowHeight, gap, items: [{panel, x, y, w, h}]}), data (a map of named "
        "sources — kind: inline | connector | dataset | join | function | live), and "
        "panels (a map of chart panels {title, dataRef, config} plus text, image, "
        "filters and control panels). Optional: params (declared parameters that "
        "$param tokens in queries reference), relationships (cross-source marking), "
        "filterSchemes, markingMode, theme, broadcastGroup. Sources support query "
        "pushdown (groupBy/measures/where/orderBy/limit — run in the database for "
        "connector sources) and calculatedFields (formulas and binning computed by "
        "the engine)." % DASHBOARD_SCHEMA_VERSION
    ),
    "linking": (
        "Panels that share a broadcastGroup (the spec sets one dashboard-wide by "
        "default) coordinate automatically: clicking a bar, slice or table row "
        "selects the matching samples in every other panel showing the same data. "
        "Cross-SOURCE linking uses relationships ([{left, right, on}]) or join "
        "sources. Per-panel actions (spec 1.4) add drill-through on mark click: "
        "setParam (set a dashboard parameter and re-query), setFilter (write a "
        "values predicate into the global filter model), focusPanel (scroll a "
        "target panel into view). Controls: filters panels, param controls "
        "(dropdown/radio/buttons/search/slider) and config controls that switch a "
        "target panel's config."
    ),
    "workflow": (
        "Shortcut for one dataset: call suggest_canvasxpress_dashboard with the data. If it "
        "is a patient cohort with survival columns or a sequencing run-QC table, it returns "
        "the whole validated, linked dashboard (filters plus the panels that shape calls "
        "for) in one step; otherwise use the workflow below. "
        "Recommended MCP workflow: 1) per chart, call "
        "recommend_canvasxpress_graphtype then generate_canvasxpress_config to get "
        "each panel's {data, config}; 2) call compose_canvasxpress_dashboard with "
        "those panels to assemble and validate the spec; 3) optionally edit the "
        "spec (add controls, params, relationships, pushdown) and re-check it with "
        "validate_dashboard_spec; 4) render with the usage snippet returned by the "
        "compose tool, or import the spec into the builder of a self-hosted "
        "canvasxpress-dashboards server."
    ),
}


def explain(topic: Optional[str] = None) -> dict:
    """Documentation for the explain_canvasxpress_dashboards tool."""
    if topic and topic in _DOCS:
        return {"topic": topic, "explanation": _DOCS[topic],
                "topics": sorted(_DOCS)}
    return {"topics": sorted(_DOCS), "sections": _DOCS}
