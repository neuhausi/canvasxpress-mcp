"""Tests for the dashboards tools (compose / validate / explain).

All offline and in-process — no LLM, no network.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cx_dashboard  # noqa: E402
import cx_dashboard_validate  # noqa: E402

BAR = {
    "data": {"y": {"vars": ["Revenue"], "smps": ["North", "South"], "data": [[820, 640]]},
             "x": {"Region": ["North", "South"]}},
    "config": {"graphType": "Bar", "colorBy": "Region", "title": "Revenue by Region"},
}
PIE = {
    "data": BAR["data"],
    "config": {"graphType": "Pie", "colorBy": "Region"},
    "title": "Share by Region",
}
LINE = {
    "data": {"y": {"vars": ["North"], "smps": ["Q1", "Q2"], "data": [[180, 210]]}},
    "config": {"graphType": "Line"},
    "title": "Trend",
}


def test_compose_valid_spec():
    out = cx_dashboard.compose_dashboard("Sales Overview", [BAR, PIE, LINE],
                                         description="Three linked panels.")
    assert out["valid"], out["errors"]
    spec = out["spec"]
    assert spec["id"] == "sales-overview"
    assert spec["schemaVersion"] == cx_dashboard_validate.DASHBOARD_SCHEMA_VERSION
    # intro + 3 chart panels, every layout item references a real panel
    assert len(spec["panels"]) == 4
    assert all(item["panel"] in spec["panels"] for item in spec["layout"]["items"])
    # independent re-validation agrees
    assert cx_dashboard_validate.validate_spec(spec)["valid"]
    assert "renderDashboard" in out["usage"]


def test_compose_dedupes_shared_data():
    out = cx_dashboard.compose_dashboard("Shared", [BAR, PIE])
    spec = out["spec"]
    # BAR and PIE carry the same data object -> one inline source, two panels on it
    assert len(spec["data"]) == 1
    refs = {p["dataRef"] for p in spec["panels"].values()}
    assert len(refs) == 1


def test_compose_layout_wraps_rows():
    out = cx_dashboard.compose_dashboard("Wrap", [BAR, PIE, LINE], columns=2)
    items = out["spec"]["layout"]["items"]
    assert [i["x"] for i in items] == [0, 6, 0]
    assert items[2]["y"] > items[0]["y"]
    # titles stripped from configs (the panel header shows them)
    assert "title" not in list(out["spec"]["panels"].values())[0]["config"]


def test_compose_rejects_bad_input():
    assert not cx_dashboard.compose_dashboard("", [BAR])["valid"]
    assert not cx_dashboard.compose_dashboard("X", [])["valid"]
    assert not cx_dashboard.compose_dashboard("X", [{"config": {}}])["valid"]


def test_validate_catches_broken_references():
    out = cx_dashboard.compose_dashboard("Breakable", [BAR])
    spec = json.loads(json.dumps(out["spec"]))
    spec["layout"]["items"].append({"panel": "ghost", "x": 0, "y": 9, "w": 6, "h": 3})
    spec["panels"]["orphan"] = {"title": "Orphan", "dataRef": "nope", "config": {}}
    result = cx_dashboard_validate.validate_spec(spec)
    assert not result["valid"]
    joined = " ".join(result["errors"])
    assert "ghost" in joined and "nope" in joined


def test_validate_pushdown_and_params():
    out = cx_dashboard.compose_dashboard("Push", [BAR])
    spec = out["spec"]
    ref = next(iter(spec["data"]))
    spec["data"][ref]["pushdown"] = {
        "groupBy": ["Region"],
        "measures": [{"fn": "sum", "column": "Revenue"}],
        "where": [{"column": "Region", "op": "=", "value": "$region"}],
    }
    result = cx_dashboard_validate.validate_spec(spec)
    assert not result["valid"] and "undeclared param" in " ".join(result["errors"])
    spec["params"] = {"region": "North"}
    assert cx_dashboard_validate.validate_spec(spec)["valid"]


def test_explain_topics():
    everything = cx_dashboard.explain()
    assert set(everything["sections"]) == {"overview", "install", "spec", "linking", "workflow", "inline_widget"}
    one = cx_dashboard.explain("linking")
    assert one["topic"] == "linking" and "broadcast" in one["explanation"]


def test_tools_registered():
    import asyncio
    import server as server_module
    loop = asyncio.new_event_loop()
    try:
        tools = loop.run_until_complete(server_module.mcp.list_tools())
    finally:
        loop.close()
    names = {t.name for t in tools}
    assert {"compose_canvasxpress_dashboard", "validate_dashboard_spec",
            "explain_canvasxpress_dashboards", "suggest_canvasxpress_dashboard"} <= names


# ---------------------------------------------------------------------------
# compose: a generated scatter config must not be flipped twice
# ---------------------------------------------------------------------------

def _penguin_rows():
    import cx_datasets
    data = cx_datasets.load_dataset("penguins")
    y, x = data["y"], data["x"]
    rows = [list(y["vars"]) + list(x)]
    for j in range(len(y["smps"])):
        rows.append([y["data"][i][j] for i in range(len(y["vars"]))] + [x[k][j] for k in x])
    return data, rows


_SCATTER = {"graphType": "Scatter2D", "xAxis": ["bill_length_mm"], "yAxis": ["bill_depth_mm"],
            "colorBy": "species"}


def test_compose_drops_transpose_data_the_renderer_would_repeat():
    import cx_dashboard
    data, _ = _penguin_rows()
    out = cx_dashboard.compose_dashboard(
        "Penguins", [{"data": data, "config": dict(_SCATTER, transposeData=True)}])
    assert out["valid"], out["errors"]
    assert "transposeData" not in next(iter(out["spec"]["panels"].values()))["config"]


def test_compose_keeps_transpose_data_when_the_renderer_will_not_transpose():
    import cx_dashboard
    data, _ = _penguin_rows()
    cfg = {"graphType": "Histogram", "xAxis": ["body_mass_g"], "transposeData": True}
    out = cx_dashboard.compose_dashboard("H", [{"data": data, "config": cfg}])
    assert next(iter(out["spec"]["panels"].values()))["config"]["transposeData"] is True


def test_compose_turns_a_headers_plus_rows_panel_into_a_data_object():
    import cx_dashboard
    data, rows = _penguin_rows()
    out = cx_dashboard.compose_dashboard(
        "Penguins", [{"data": rows, "config": dict(_SCATTER, transposeData=True)},
                     {"data": data, "config": {"graphType": "Boxplot", "xAxis": ["body_mass_g"]}}])
    assert out["valid"], out["errors"]
    sources = list(out["spec"]["data"].values())
    assert len(sources) == 1                       # array and object of the same table share a source
    assert sources[0]["value"]["y"]["vars"][0] == "bill_length_mm"


def test_compose_reports_a_bad_headers_plus_rows_panel():
    import cx_dashboard
    out = cx_dashboard.compose_dashboard("X", [{"data": [["a", "b"]], "config": dict(_SCATTER)}])
    assert out["valid"] is False and "panels[0].data" in out["errors"][0]
