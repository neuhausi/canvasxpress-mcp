"""Whole-dashboard suggestion for the canvasxpress-mcp server.

Given ONE parsed dataset (a CanvasXpress data object), recognise a known analysis shape —
a patient cohort with survival columns, or a per-sample NGS run-QC table — and suggest a
whole *linked* dashboard: a Filters panel plus the panels that shape calls for, all bound to
the one dataset so the engine's broadcast mechanism links selections across them.

This is the deterministic Python twin of ``src/suggestDashboard.js`` in
canvasxpress-dashboards (which the dashboards wizard uses). The JS is the reference: its
output for a set of datasets is recorded in ``tests/fixtures/suggest-dashboard-parity.json``
and ``tests/test_dashboard_suggest.py`` requires this module to reproduce every case
exactly. Change the two together (regenerate the fixture from the JS side).

Like the wizard it only ever *suggests*: ``suggest_dashboard`` returns ``None`` when no
template fits (use ``select_canvasxpress_chart`` + ``compose_canvasxpress_dashboard``
instead) and never alters the data. Pure functions; no server state. The spec is validated
by the vendored ``cx_dashboard_validate.validate_spec``.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any, Optional

from cx_dashboard_validate import validate_spec, DASHBOARD_SCHEMA_VERSION

# ---------------------------------------------------------------------------
# Detection vocabulary (mirrors wizard.js / suggestDashboard.js)
# ---------------------------------------------------------------------------

_SURVIVAL_TIME_RE = re.compile(
    r"^(time|t|surv(ival)?|os|pfs|dfs|efs|tte|duration|follow[\s._-]?up|fu|days?|months?|weeks?|years?)$"
    r"|^(os|pfs|dfs|efs|rfs|dss|ttp)[\s._-]?(time|months?|days?|weeks?|years?|duration)$"
    r"|^(overall|progression[\s._-]?free|disease[\s._-]?free|event[\s._-]?free|relapse[\s._-]?free)"
    r"[\s._-]?survival[\s._-]?(time|months?|days?|weeks?|years?)$",
    re.I)
_SURVIVAL_EVENT_RE = re.compile(r"(event|status|dead|death|censor|vital|outcome|relapse|progress|recur)", re.I)
_EVENT_LEVEL_RE = re.compile(
    r"^(dead|alive|death|censor(ed)?|event|no[\s._-]?event|yes|no|deceased|living|1|0|relapse|progression|cr|pd)$",
    re.I)
_DATE_RE = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}([ T]\d{1,2}:\d{2}(:\d{2})?)?$|^\d{1,2}/\d{1,2}/\d{2,4}$")
_MISSING_RE = re.compile(r"^(na|n/a|nan|null|-)$", re.I)
# What JavaScript's Number() reads as a finite decimal number (hex/octal/binary literals are not
# accepted: they do not occur in survival or QC tables).
_NUMBER_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")

_QC_METRICS = {
    "reads": re.compile(r"(^|[\s._-])(total[\s._-]?)?reads?($|[\s._-])|read[\s._-]?(pairs?|count)|^n[\s._-]?reads$|fragments?", re.I),
    "mapping": re.compile(r"(mapped|mapping|aligned|alignment)", re.I),
    "quality": re.compile(r"(q30|q20|phred|base[\s._-]?qual|mean[\s._-]?qual)", re.I),
    "duplication": re.compile(r"(dup|duplicat)", re.I),
    "gc": re.compile(r"(^|[\s._-])gc($|[\s._-])|gc[\s._-]?(content|pct|percent)", re.I),
    "insert": re.compile(r"insert[\s._-]?size", re.I),
    "coverage": re.compile(r"(coverage|depth|cov[\s._-]?x)", re.I),
    "adapter": re.compile(r"adapter", re.I),
    "contamination": re.compile(r"(contamin|freemix)", re.I),
}
_BATCH_RE = re.compile(r"(lane|flow[\s._-]?cell|batch|run|plate|library|prep|instrument|machine|seq)", re.I)
_COHORT_GROUP_RE = re.compile(r"(arm|group|treat|cohort|stage|condition|sex|subtype|genotype)", re.I)

# Graph types the engine draws as Scatter2D: their axes name SAMPLES (see wizard.js).
_SCATTER_FAMILY = {
    "Bin", "Binplot", "Bump", "CDF", "Contour", "Density", "Distribution", "Hex", "Hexplot",
    "Histogram", "KaplanMeier", "QQ", "Quantile", "Ridgeline", "Scatter2D", "ScatterBubble2D",
    "Scatter3D", "Spaghetti", "SPLOM", "Streamgraph", "TimeSeries", "Volcano",
}
# The dashboard renderer transposes row-per-sample data for these itself (shouldTranspose).
_ROW_POINT_GRAPHS = {"Scatter2D", "Scatter3D", "ScatterBubble2D", "KaplanMeier", "Pie"}

_MIN_PANELS = 3
_MAX_FILTER_FIELDS = 6
_GRID_H = 12
_GRID_COLS = 12
_CARD_H = 4

# The case drill: a click on a panel whose marks are single rows sets this param to the clicked
# row; the case card is a table over a second source of the same data, narrowed to that row.
_CASE_PARAM = "case"
_CASE_SOURCE = "case"
# A one-row table: no banner / toolbar / status bar / paging, shrunk to its rows, and no heat-colouring:
# the engine re-derives dataTableColored from the UI style (it is on for editorial and most others, and
# the config key is ignored), and the colours of a single record's values mean nothing — so their
# strength is set to 0, which IS honoured.
_CASE_CARD_CONFIG = {
    "graphType": "Bar", "view": "table", "title": False, "dataTableBannerShow": False,
    "dataTableToolbarShow": False, "dataTableStatusShow": False, "dataTablePaginationShow": False,
    "dataTableFitHeight": True, "dataTableColorTransparency": 0,
}


# ---------------------------------------------------------------------------
# Table + column profile (wizard.js tableFromData / profileColumns)
# ---------------------------------------------------------------------------

def _js_str(value: Any) -> str:
    """``String(value)`` as JavaScript prints it for the JSON scalars that occur in tables."""
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _is_numeric(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return value == value and value not in (float("inf"), float("-inf"))
    if not isinstance(value, str):
        return False
    s = value.strip()
    return s != "" and bool(_NUMBER_RE.match(s))


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    s = _js_str(value).strip()
    return s == "" or bool(_MISSING_RE.match(s))


def _vars_are_columns(data: dict) -> bool:
    y = data.get("y") or {}
    return len(y.get("vars") or []) <= len(y.get("smps") or [])


def _table_from_data(data: dict) -> list:
    """Header + rows: row-id column, measured columns, then the annotation columns."""
    y = data.get("y") or {}
    variables = y.get("vars") or []
    samples = y.get("smps") or []
    matrix = y.get("data") or []
    by_var = _vars_are_columns(data)
    columns = variables if by_var else samples
    row_ids = samples if by_var else variables
    annotations = (data.get("x") if by_var else data.get("z")) or {}
    keys = list(annotations.keys())
    table = [["Id"] + [str(c) for c in columns] + keys]
    for r in range(len(row_ids)):
        row = [row_ids[r]]
        for c in range(len(columns)):
            if by_var:
                col = matrix[c] if c < len(matrix) and matrix[c] is not None else []
                row.append(col[r] if r < len(col) else None)
            else:
                line = matrix[r] if r < len(matrix) and matrix[r] is not None else []
                row.append(line[c] if c < len(line) else None)
        for key in keys:
            values = annotations.get(key) or []
            row.append(values[r] if r < len(values) else None)
        table.append(row)
    return table


def _profile_columns(table: list) -> list:
    headers = table[0] if table else []
    rows = table[1:]
    out = []
    for col, name in enumerate(headers):
        numeric = dates = present = 0
        seen = set()
        for row in rows:
            value = row[col] if col < len(row) else None
            if _is_missing(value):
                continue
            present += 1
            seen.add(_js_str(value))
            if _is_numeric(value):
                numeric += 1
            elif isinstance(value, str) and _DATE_RE.match(value.strip()):
                dates += 1
        if present and numeric / present > 0.5:
            kind = "numeric"
        elif present and dates / present > 0.5:
            kind = "date"
        else:
            kind = "text"
        out.append({"name": str(name), "type": kind, "unique": len(seen)})
    return out


def _describe_columns(data: dict) -> list:
    return _profile_columns(_table_from_data(data))[1:]


# ---------------------------------------------------------------------------
# Survival-shape detection (wizard.js detectDataShapes, KaplanMeier part)
# ---------------------------------------------------------------------------

def _find_header(headers: list, pattern: "re.Pattern") -> int:
    for i, header in enumerate(headers):
        if pattern.search(str(header).strip()):
            return i
    return -1


def _looks_like_event(cells: list) -> bool:
    if not cells:
        return False
    if all(_js_str(v).strip() in ("0", "1", "true", "false") for v in cells):
        return True
    levels = {_js_str(v).strip().lower() for v in cells}
    if len(levels) > 3:
        return False
    return all(_EVENT_LEVEL_RE.match(level) for level in levels)


def _detect_survival(data: dict) -> Optional[dict]:
    """The matched ``{time, event}`` columns of a survival table, or None."""
    table = _table_from_data(data)
    headers = list(table[0][1:]) if table else []
    rows = [row[1:] for row in table[1:]]
    if not headers or not rows:
        return None
    profile = _profile_columns(table)[1:]
    time_col = _find_header(headers, _SURVIVAL_TIME_RE)
    event_col = _find_header(headers, _SURVIVAL_EVENT_RE)
    if time_col == -1 or profile[time_col]["type"] != "numeric" or event_col == -1 or event_col == time_col:
        return None
    # The engine's Kaplan-Meier reads the event from a NUMERIC column (a text event draws a wrong curve).
    if profile[event_col]["type"] != "numeric":
        return None
    cells = [row[event_col] for row in rows if event_col < len(row) and not _is_missing(row[event_col])]
    if not _looks_like_event(cells):
        return None
    return {"time": str(headers[time_col]), "event": str(headers[event_col])}


# ---------------------------------------------------------------------------
# Config builders (wizard.js orientConfig / buildChartConfig / buildScientificConfig)
# ---------------------------------------------------------------------------

def _orient_config(config: dict, data: dict) -> dict:
    out = copy.deepcopy(config)
    if out.get("graphType") in _SCATTER_FAMILY and out.get("transposeData") is None and _vars_are_columns(data):
        out["transposeData"] = True
    return out


def _chart_fields_for(graph_type: str) -> dict:
    if graph_type == "Scatter2D":
        return {"y": True, "groupKey": None}
    if graph_type == "Ridgeline":
        return {"y": False, "groupKey": "ridgeBy"}
    if graph_type in ("Histogram", "Density"):
        return {"y": False, "groupKey": None}
    return {"y": False, "groupKey": "groupingFactors"}


def _build_chart_config(graph_type: str, fields: dict, extra: Optional[dict], data: dict) -> dict:
    spec = _chart_fields_for(graph_type)
    config: dict = {"graphType": graph_type}
    if fields.get("x"):
        config["xAxis"] = [fields["x"]]
    if spec["y"] and fields.get("y"):
        config["yAxis"] = [fields["y"]]
    if spec["groupKey"] and fields.get("group"):
        config[spec["groupKey"]] = fields["group"] if spec["groupKey"] == "ridgeBy" else [fields["group"]]
    if fields.get("color"):
        config["colorBy"] = fields["color"]
    config.update(extra or {})
    return _orient_config(config, data)


def _build_km_config(fields: dict, data: dict) -> dict:
    config: dict = {"graphType": "KaplanMeier", "showKMConfidenceIntervals": True, "kmPvalue": True,
                    "kmRiskTable": True, "kmCensoredRug": True}
    if fields.get("time"):
        config["xAxis"] = [fields["time"]]
    if fields.get("event"):
        config["yAxis"] = [fields["event"]]
    if fields.get("group"):
        config["colorBy"] = fields["group"]
    return _orient_config(config, data)


def _should_transpose(config: dict, data: dict) -> bool:
    """renderDashboard.js ``shouldTranspose`` for a panel with no explicit ``transpose``."""
    if config.get("graphType") not in _ROW_POINT_GRAPHS or not data.get("y"):
        return False
    names = [n for n in list(config.get("xAxis") or []) + list(config.get("yAxis") or []) if isinstance(n, str)]
    y = data["y"]
    variables = y.get("vars") or []
    samples = y.get("smps") or []
    if not names:
        return config.get("graphType") == "Pie" and len(variables) == 1 and len(samples) > 1
    return all(n in variables for n in names) and not any(n in samples for n in names)


def _panel_config_for(config: dict, data: dict) -> dict:
    """The config as a dashboard panel must carry it (see wizard.js ``panelConfigFor``): the
    engine's ``transposeData`` is dropped exactly when the dashboard renderer will transpose."""
    out = copy.deepcopy(config)
    if out.get("transposeData") and _should_transpose(out, data):
        del out["transposeData"]
    return out


def panel_config_for(config: dict, data: Any) -> dict:
    """``_panel_config_for`` for callers outside the templates (compose): a copy of the config
    without the engine's ``transposeData`` when the dashboard renderer will transpose the data."""
    if not isinstance(data, dict) or not isinstance(data.get("y"), dict):
        return dict(config)
    return _panel_config_for(config, data)


# ---------------------------------------------------------------------------
# Templates (suggestDashboard.js)
# ---------------------------------------------------------------------------

def _pick_group(columns: list, preferred: "re.Pattern", exclude: list) -> Optional[str]:
    candidates = [c for c in columns if c["type"] == "text" and 2 <= c["unique"] <= 6 and c["name"] not in exclude]
    for c in candidates:
        if preferred.search(c["name"]):
            return c["name"]
    return candidates[0]["name"] if candidates else None


def _filter_fields(columns: list, skip: list) -> list:
    categories = [c for c in columns if c["type"] == "text" and 2 <= c["unique"] <= 12 and c["name"] not in skip]
    measures = [c for c in columns if c["type"] == "numeric" and c["name"] not in skip]
    return [c["name"] for c in (categories + measures)[:_MAX_FILTER_FIELDS]]


def _without_title(config: dict) -> dict:
    config["title"] = False
    return config


def _with_dashboard_configs(panels: list, data: dict) -> list:
    for panel in panels:
        if "config" not in panel:
            continue
        for_dashboard = _panel_config_for(panel["config"], data)
        if for_dashboard != panel["config"]:
            panel["dashboardConfig"] = for_dashboard
    return panels


def _with_case_drill(suggestion: dict, data: dict, card_title: str, row_panels: list) -> dict:
    """Add the case drill: the click action on each panel whose marks are single rows (a scatter
    point, a sample bar — NOT a box or a curve, whose click names a group), the ``drill``
    description, and the case card panel under everything. Added after the template's own size
    check, so the card never counts toward the minimum."""
    for panel in suggestion["panels"]:
        if panel["id"] in row_panels:
            panel["actions"] = [{"on": "click", "setParam": {"name": _CASE_PARAM}}]
    suggestion["panels"].append({
        "id": "case", "card": True, "title": card_title, "config": copy.deepcopy(_CASE_CARD_CONFIG),
        "x": 0, "y": _GRID_H * 2, "w": 12, "h": _CARD_H})
    suggestion["drill"] = {"param": _CASE_PARAM, "source": _CASE_SOURCE,
                           "initial": str(data["y"]["smps"][0])}
    return suggestion


def _cohort_template(data: dict, columns: list, survival: dict) -> Optional[dict]:
    # The engine's Kaplan-Meier reads the event from a NUMERIC column. A text event (Dead / Alive)
    # is an annotation: the curve then draws silently wrong (flat steps, p = 1), so offer nothing.
    event_column = next((c for c in columns if c["name"] == survival["event"]), None)
    if event_column is None or event_column["type"] != "numeric":
        return None
    used = [survival["time"], survival["event"]]
    group = _pick_group(columns, _COHORT_GROUP_RE, used)
    measures = [c for c in columns if c["type"] == "numeric" and c["name"] not in used]
    measure = measures[0]["name"] if measures else None
    panels: list = [{"id": "filters", "type": "filters", "title": "Filters",
                     "fields": _filter_fields(columns, used), "x": 0, "y": 0, "w": 3, "h": _GRID_H * 2}]
    panels.append({
        "id": "km", "title": "Survival by " + group if group else "Survival",
        "config": _without_title(_build_km_config(
            {"time": survival["time"], "event": survival["event"], "group": group}, data)),
        "x": 3, "y": 0, "w": 9, "h": _GRID_H})
    if measure:
        panels.append({
            "id": "compare", "title": measure + (" by " + group if group else ""),
            "config": _without_title(_build_chart_config(
                "Boxplot", {"x": measure, "group": group, "color": group},
                {"graphOrientation": "vertical", "showBoxplotOriginalData": True}, data)),
            "x": 3, "y": _GRID_H, "w": 4, "h": _GRID_H})
        panels.append({
            "id": "vs-time", "title": measure + " vs " + survival["time"],
            "config": _without_title(_build_chart_config(
                "Scatter2D", {"x": measure, "y": survival["time"], "color": group},
                {"dataPointSize": "auto"}, data)),
            "x": 7, "y": _GRID_H, "w": 5, "h": _GRID_H})
    if len(panels) < _MIN_PANELS:
        return None
    return _with_case_drill({
        "template": "cohort", "label": "Cohort overview",
        "panels": _with_dashboard_configs(panels, data),
        "reason": 'Survival table (time "' + survival["time"] + '", event "' + survival["event"] + '")'
                  + (' with grouping "' + group + '"' if group else "")
                  + ": filters, a Kaplan–Meier curve and the measures that move with it; "
                  "click a point to load that patient's record."},
        data, "Patient record — click a point to load", ["vs-time"])


def _qc_metric_columns(columns: list) -> dict:
    found: dict = {}
    for c in columns:
        if c["type"] != "numeric":
            continue
        for family, pattern in _QC_METRICS.items():
            if family not in found and pattern.search(c["name"]):
                found[family] = c["name"]
                break
    return found


def _qc_template(data: dict, columns: list, metrics: dict) -> Optional[dict]:
    names = list(metrics.values())
    batch = _pick_group(columns, _BATCH_RE, [])
    depth = metrics.get("reads") or metrics.get("coverage") or (names[0] if names else None)
    scatter_x = metrics.get("mapping") or (names[0] if names else None)
    others_x = [n for n in names if n != scatter_x]
    scatter_y = metrics.get("duplication") or (others_x[0] if others_x else None)
    others_depth = [n for n in names if n != depth]
    quality = metrics.get("quality") or (others_depth[0] if others_depth else None)
    panels: list = [{"id": "filters", "type": "filters", "title": "Filters",
                     "fields": _filter_fields(columns, []), "x": 0, "y": 0, "w": 3, "h": _GRID_H * 2}]
    panels.append({
        "id": "depth", "title": depth + " per sample",
        "config": _without_title(_build_chart_config(
            "Bar", {"x": depth, "color": batch},
            {"graphOrientation": "vertical", "smpLabelRotate": 45}, data)),
        "x": 3, "y": 0, "w": 9, "h": _GRID_H})
    if scatter_x and scatter_y:
        panels.append({
            "id": "quality-scatter", "title": scatter_x + " vs " + scatter_y,
            "config": _without_title(_build_chart_config(
                "Scatter2D", {"x": scatter_x, "y": scatter_y, "color": batch},
                {"dataPointSize": "auto"}, data)),
            "x": 3, "y": _GRID_H, "w": 5, "h": _GRID_H})
    if quality:
        # Always a Boxplot, grouped by batch when there is one: a Histogram needs the engine's
        # transposeData, which transposes the dataset IN PLACE — and every panel of the dashboard
        # shares that one data object, so the sibling panels would be drawn flipped.
        by_batch = _build_chart_config(
            "Boxplot", {"x": quality, "group": batch, "color": batch},
            {"graphOrientation": "vertical", "showBoxplotOriginalData": True}, data)
        panels.append({
            "id": "quality-spread", "title": quality + (" by " + batch if batch else ""),
            "config": _without_title(by_batch), "x": 8, "y": _GRID_H, "w": 4, "h": _GRID_H})
    if len(panels) < _MIN_PANELS:
        return None
    return _with_case_drill({
        "template": "ngs-qc", "label": "Sequencing run QC",
        "panels": _with_dashboard_configs(panels, data),
        "reason": "Per-sample QC table (" + ", ".join(metrics.keys()) + "): filters, depth per sample, "
                  "mapping against duplication and the quality spread"
                  + (' by "' + batch + '"' if batch else "")
                  + "; click a bar or point to load that sample's record."},
        data, "Sample record — click a bar or point to load", ["depth", "quality-scatter"])


# ---------------------------------------------------------------------------
# Generic "explore" template (the fallback for any numeric + factor table)
# ---------------------------------------------------------------------------

_MIN_FACTOR_LEVELS = 2
_MAX_FACTOR_LEVELS = 12
_COUNT_FIELD = "n"


def _explore_columns(data: dict) -> Optional[dict]:
    """The numeric measures (``y`` variables) and factor columns (``x`` annotations) of a
    standard-orientation table, with each factor's non-missing levels. None when it lacks
    two measures or a usable factor."""
    y = data.get("y") or {}
    annotations = data.get("x") or {}
    measures = []
    for r, name in enumerate(y.get("vars") or []):
        row = (y.get("data") or [[]] * (r + 1))[r] or []
        values = {v for v in row if isinstance(v, (int, float)) and not isinstance(v, bool)}
        if len(values) >= 2:
            measures.append(str(name))
    factors = []
    for name, values in annotations.items():
        # A missing cell is its own "NA" level, as the filter and the count bar show it.
        levels = {"NA" if _is_missing(v) else _js_str(v) for v in (values or [])}
        if _MIN_FACTOR_LEVELS <= len(levels) <= _MAX_FACTOR_LEVELS:
            factors.append({"name": str(name), "levels": len(levels)})
    if len(measures) < 2 or not factors:
        return None
    return {"measures": measures, "factors": factors}


def _separation(data: dict, measure: str, factor: str) -> float:
    """Between-group share of the variance (eta squared) of a measure across a factor's
    levels: scale-free, so grams and millimetres compare."""
    y = data["y"]
    row = y["data"][y["vars"].index(measure)]
    labels = data["x"][factor]
    groups: dict = {}
    for value, label in zip(row, labels):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            # A missing label is its own "NA" group: no row is dropped.
            groups.setdefault("NA" if _is_missing(label) else _js_str(label), []).append(float(value))
    everything = [v for g in groups.values() for v in g]
    if len(groups) < 2 or len(everything) < 2:
        return 0.0
    mean = sum(everything) / len(everything)
    total = sum((v - mean) ** 2 for v in everything)
    if total == 0:
        return 0.0
    between = sum(len(g) * (sum(g) / len(g) - mean) ** 2 for g in groups.values())
    return between / total


_TIME_NAME_RE = re.compile(r"(^|[\s._-])(year|yr|date|time|datetime|timestamp|month|day|week|quarter)($|[\s._-])", re.I)


def _is_time_like(name: str, values: list) -> bool:
    """A year / date column: a poor thing to colour or group by, though fine as a filter."""
    present = [_js_str(v).strip() for v in values if not _is_missing(v)]
    return bool(_TIME_NAME_RE.search(name)) or (
        bool(present) and all(_DATE_RE.match(v) for v in present))


def _association(first: list, second: list) -> float:
    """Cramer's V between two factors (0 = independent, 1 = one determines the other); a missing
    cell is its own "NA" level."""
    a = ["NA" if _is_missing(v) else _js_str(v) for v in first]
    b = ["NA" if _is_missing(v) else _js_str(v) for v in second]
    rows, cols = sorted(set(a)), sorted(set(b))
    n = len(a)
    if n == 0 or len(rows) < 2 or len(cols) < 2:
        return 0.0
    counts: dict = {}
    for x, y in zip(a, b):
        counts[(x, y)] = counts.get((x, y), 0) + 1
    row_totals = {r: a.count(r) for r in rows}
    col_totals = {c: b.count(c) for c in cols}
    chi = sum((counts.get((r, c), 0) - row_totals[r] * col_totals[c] / n) ** 2
              / (row_totals[r] * col_totals[c] / n) for r in rows for c in cols)
    return (chi / (n * (min(len(rows), len(cols)) - 1))) ** 0.5


def _pick_main_and_second(data: dict, factors: list, measures: list) -> "tuple[Optional[str], Optional[str]]":
    """Main factor: fewest levels (NA counts as a level), ties to the one that separates the
    numeric columns most. Second: the next fewest levels, ties to the one least tied to the main
    factor (an independent factor adds a second view; one nested in the main factor repeats it).
    Year / date columns are never chosen."""
    candidates = [f for f in factors if not _is_time_like(f["name"], data["x"][f["name"]])]
    if not candidates:
        return None, None

    def spread(f: dict) -> float:
        return sum(_separation(data, m, f["name"]) for m in measures) / len(measures)

    fewest = min(f["levels"] for f in candidates)
    main = max((f for f in candidates if f["levels"] == fewest), key=spread)
    rest = [f for f in candidates if f is not main]
    if not rest:
        return main["name"], None
    fewest = min(f["levels"] for f in rest)
    second = min((f for f in rest if f["levels"] == fewest),
                 key=lambda f: _association(data["x"][main["name"]], data["x"][f["name"]]))
    return main["name"], second["name"]


def _explore_template(data: dict) -> Optional[dict]:
    found = _explore_columns(data)
    if found is None:
        return None
    main, second = _pick_main_and_second(data, found["factors"], found["measures"])
    if main is None:
        return None
    ranked = sorted(found["measures"], key=lambda m: -_separation(data, m, main))
    scatter_x, scatter_y = ranked[0], ranked[1]
    rest = [m for m in ranked if m not in (scatter_x, scatter_y)]
    spread = rest[0] if rest else ranked[0]
    distribution = rest[1] if len(rest) > 1 else (rest[0] if rest else ranked[1])
    fields = [f["name"] for f in found["factors"]]

    panels: list = [{"id": "filters", "type": "filters", "title": "Filters", "fields": fields,
                     "x": 0, "y": 0, "w": 3, "h": _GRID_H * 2}]
    panels.append({
        "id": "scatter", "title": scatter_x + " vs " + scatter_y,
        "config": _without_title(_build_chart_config(
            "Scatter2D", {"x": scatter_x, "y": scatter_y, "color": main},
            {"dataPointSize": "auto"}, data)),
        "x": 3, "y": 0, "w": 5, "h": _GRID_H})
    panels.append({
        "id": "compare", "title": spread + " by " + main + (" and " + second if second else ""),
        "config": _without_title(_build_chart_config(
            "Boxplot", {"x": spread, "group": main, "color": second or main},
            {"graphOrientation": "vertical", "showBoxplotOriginalData": True}, data)),
        "x": 8, "y": 0, "w": 4, "h": _GRID_H})
    panels.append({
        "id": "distribution", "title": distribution + " distribution",
        "config": _without_title(_build_chart_config(
            "Histogram", {"x": distribution, "group": main, "color": main},
            {"groupingFactors": [main]}, data)),
        # A Histogram draws one series per column of its data and takes rows as the values, so the
        # panel keeps only its measure and flips a COPY of the data (panel.transpose); the
        # engine's transposeData would flip the one data object every sibling panel shares.
        "measures": [distribution], "transpose": True,
        "x": 3, "y": _GRID_H, "w": 4, "h": _GRID_H})
    panels.append({
        "id": "counts", "title": "Count by " + (second or main),
        "config": _without_title({
            "graphType": "Stacked", "graphOrientation": "vertical", "xAxis": [_COUNT_FIELD],
            "groupingFactors": [second or main], "stackBy": main, "colorBy": main}),
        # The source's calculated field n = 1 sums to a row count per bar; the panel keeps only it.
        "measures": [_COUNT_FIELD],
        "x": 7, "y": _GRID_H, "w": 5, "h": _GRID_H})
    for panel in panels:
        if panel.get("transpose"):
            panel["config"].pop("transposeData", None)
    return {
        "template": "explore", "label": "Data explorer",
        "panels": _with_dashboard_configs(panels, data),
        "calculatedFields": [{"name": _COUNT_FIELD, "formula": "1"}],
        "reason": 'Table with %d numeric columns and %d factors: filters, "%s" against "%s", "%s" by "%s"%s, '
                  'the spread of "%s" and counts by "%s" -- all coloured by "%s".' % (
                      len(found["measures"]), len(found["factors"]), scatter_x, scatter_y, spread, main,
                      (' and "%s"' % second) if second else "", distribution, second or main, main)}


def _is_standard_orientation(data: Any) -> bool:
    """A table whose variables are the measures and whose rows are the samples."""
    y = data.get("y") if isinstance(data, dict) else None
    return isinstance(y, dict) and bool(y.get("vars")) and len(y["vars"]) <= len(y.get("smps") or [])


def suggest_dashboard(data: Any) -> Optional[dict]:
    """Suggest a whole linked dashboard for a dataset, if it matches a known analysis shape.

    Only the standard orientation is handled (variables are the measures, rows are the samples);
    a table stored the other way round gets no suggestion.

    Recognises a survival table (``cohort``: filters + Kaplan-Meier + a measure by group and
    against time) and a per-sample sequencing QC table (``ngs-qc``: three or more QC metric
    families among the numeric columns → filters + depth + mapping vs duplication + quality
    by batch). A template that would yield fewer than three panels is not offered.

    Args:
        data: a CanvasXpress data object ``{y: {vars, smps, data}, x?, z?}``.

    Returns:
        ``{template, label, reason, panels}`` — plain panel descriptors (a chart panel's
        ``config`` draws standalone; ``dashboardConfig`` is what the published dashboard
        carries where that differs) — or ``None`` when nothing fits.
    """
    if not isinstance(data, dict) or not isinstance(data.get("y"), dict) or not (data["y"].get("vars") or []):
        return None
    # Standard orientation only: the measures are the variables, the rows the samples. A table
    # stored the other way round (more variables than samples, annotations in `z`) needs its source
    # row axis declared and per-panel transposes, and draws wrong without them — so offer nothing.
    if len(data["y"]["vars"]) > len(data["y"].get("smps") or []):
        return None
    columns = _describe_columns(data)
    if not columns:
        return None
    survival = _detect_survival(data)
    cohort = _cohort_template(data, columns, survival) if survival else None
    if cohort:
        return cohort
    metrics = _qc_metric_columns(columns)
    if len(metrics) >= 3:
        return _qc_template(data, columns, metrics)
    return None


def dashboard_spec(suggestion: dict, dashboard_id: str, data: dict, title: Optional[str] = None,
                   theme: Optional[str] = None, thresholds: Optional[list] = None) -> dict:
    """The linked dashboard spec for a suggestion: every panel bound to one inline data
    source (the dataset), so a selection in one panel selects the same rows in the rest. With a
    ``drill``: a ``case`` param, a second source narrowed to it, the case card bound to that
    source, and ``setParam`` click actions on the row-mark panels. The spec opts in to proactive
    insights (``insights``: the data-derived checks, plus any ``thresholds`` the caller declares —
    never inferred)."""
    spec: dict = {
        "id": dashboard_id,
        "title": title or suggestion["label"],
        "schemaVersion": DASHBOARD_SCHEMA_VERSION,
        "version": 1,
        "broadcastGroup": dashboard_id,
        "layout": {"cols": _GRID_COLS, "rowHeight": 30, "gap": 12, "items": []},
        "data": {"data": {"kind": "inline", "value": data}},
        "insights": {"thresholds": copy.deepcopy(thresholds)} if thresholds else {},
        "panels": {},
    }
    if theme in ("light", "dark", "auto"):
        spec["theme"] = theme
    if suggestion.get("calculatedFields"):
        spec["data"]["data"]["calculatedFields"] = copy.deepcopy(suggestion["calculatedFields"])
    drill = suggestion.get("drill")
    if drill:
        # A second source of the same data, narrowed (browser-side pushdown on the row id) to
        # the param a click sets; the case card reads it. The main source stays whole.
        spec["params"] = {drill["param"]: {"value": drill["initial"]}}
        case_source = copy.deepcopy(spec["data"]["data"])
        case_source["pushdown"] = {"where": [{"column": "smps", "op": "=", "value": "$" + drill["param"]}]}
        spec["data"][drill["source"]] = case_source
    for panel in suggestion["panels"]:
        data_ref = drill["source"] if panel.get("card") and drill else "data"
        entry: dict = {"title": panel.get("title", ""), "dataRef": data_ref}
        if panel.get("type") == "filters":
            entry = {"type": "filters", "title": panel.get("title") or "Filters", "dataRef": "data",
                     "fields": list(panel.get("fields") or [])}
        else:
            entry["config"] = copy.deepcopy(panel.get("dashboardConfig") or panel["config"])
            if panel.get("transpose") is not None:
                entry["transpose"] = panel["transpose"]
            if panel.get("measures"):
                entry["measures"] = list(panel["measures"])
            if panel.get("actions"):
                entry["actions"] = copy.deepcopy(panel["actions"])
        spec["panels"][panel["id"]] = entry
        spec["layout"]["items"].append(
            {"panel": panel["id"], "x": panel["x"], "y": panel["y"], "w": panel["w"], "h": panel["h"]})
    return spec


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-") or "dashboard"


def _resolve_input(data: Any, dataset: Optional[str]) -> "tuple[Optional[dict], Optional[str]]":
    """The data object for a ``suggest`` call, or an error message. Exactly one of ``data``
    (a ``{y, x}`` object or a headers+rows array) and ``dataset`` (a bundled name) is required."""
    import cx_datasets  # local import: keeps the template module importable on its own

    if (data is None) == (dataset is None):
        return None, ("Pass exactly one of data (an object {y, x} or a headers+rows array) or "
                      "dataset (one of: %s)." % ", ".join(cx_datasets.available_datasets()))
    try:
        if dataset is not None:
            return cx_datasets.load_dataset(dataset), None
        if isinstance(data, list):
            return cx_datasets.table_to_data(data), None
    except KeyError as exc:
        return None, str(exc.args[0])
    except ValueError as exc:
        return None, str(exc)
    return data, None


def suggest(data: Any = None, title: Optional[str] = None, theme: Optional[str] = None,
            thresholds: Optional[list] = None, dataset: Optional[str] = None) -> dict:
    """The MCP-facing result: the suggestion, the validated spec, and a usage snippet — or a
    plain "nothing fits" answer pointing at the composing tools. ``dataset`` names a bundled
    sample (see ``cx_datasets``); ``data`` may be a data object or a headers+rows array."""
    data, error = _resolve_input(data, dataset)
    if error:
        return {"suggested": False, "error": error, "reason": error}
    # The wizard-parity templates first (cohort, ngs-qc); the generic explorer is only the fallback.
    suggestion = suggest_dashboard(data)
    if suggestion is None and _is_standard_orientation(data):
        suggestion = _explore_template(data)
    if suggestion is None:
        return {
            "suggested": False,
            "reason": ("No dashboard template fits: not a survival cohort table, a per-sample "
                       "sequencing QC table, or a table with at least two numeric columns and one "
                       "factor column. Choose charts with select_canvasxpress_chart and "
                       "assemble them with compose_canvasxpress_dashboard."),
        }
    from cx_dashboard import _usage_snippet  # local import: cx_dashboard owns the snippet text

    dash_id = _slug(title or suggestion["label"])
    spec = dashboard_spec(suggestion, dash_id, data, title=title, theme=theme, thresholds=thresholds)
    result = validate_spec(spec)
    out = {
        "suggested": True,
        "template": suggestion["template"],
        "label": suggestion["label"],
        "reason": suggestion["reason"],
        "panels": [{"id": p["id"], "title": p.get("title", ""),
                    "graphType": (p.get("config") or {}).get("graphType", p.get("type"))}
                   for p in suggestion["panels"]],
        "spec": spec,
        "valid": result["valid"],
        "errors": result["errors"],
        "usage": _usage_snippet(dash_id),
    }
    if result.get("warnings"):
        out["warnings"] = result["warnings"]
    return out
