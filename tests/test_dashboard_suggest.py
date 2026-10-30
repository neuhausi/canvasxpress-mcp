"""Tests for suggest_canvasxpress_dashboard (whole linked dashboard from one dataset).

Offline and in-process. ``tests/fixtures/suggest-dashboard-parity.json`` is recorded from the
JavaScript implementation in canvasxpress-dashboards (src/suggestDashboard.js — the
reference); every case must come out identical here. Regenerate it on the JS side
(`node tests/fixtures/suggest-dashboard-parity.mjs`) and copy it over when the JS changes.
"""

import copy
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cx_dashboard_suggest as sg  # noqa: E402
import cx_dashboard_validate  # noqa: E402

with open(os.path.join(os.path.dirname(__file__), "fixtures", "suggest-dashboard-parity.json"),
          encoding="utf-8") as fh:
    FIXTURE = json.load(fh)["cases"]


@pytest.mark.parametrize("case", FIXTURE, ids=[c["name"] for c in FIXTURE])
def test_matches_the_javascript_reference(case):
    assert sg.suggest_dashboard(copy.deepcopy(case["data"])) == case["expected"]


def _case(name):
    return next(c for c in FIXTURE if c["name"] == name)


def test_fixture_covers_both_templates_and_the_misses():
    outcomes = {(c["expected"] or {}).get("template") for c in FIXTURE}
    assert outcomes == {"cohort", "ngs-qc", None}


def test_cohort_spec_is_valid_linked_and_inline():
    out = sg.suggest(_case("cohort: os_months/os_event")["data"], title="Trial")
    assert out["suggested"] and out["template"] == "cohort"
    assert out["valid"], out["errors"]
    spec = out["spec"]
    assert spec["title"] == "Trial" and spec["id"] == "trial"
    assert spec["broadcastGroup"] == "trial"
    assert spec["schemaVersion"] == cx_dashboard_validate.DASHBOARD_SCHEMA_VERSION
    assert spec["data"]["data"]["kind"] == "inline"
    # The case card reads a second source of the same data, narrowed to the clicked case.
    assert all(p["dataRef"] == ("case" if pid == "case" else "data") for pid, p in spec["panels"].items())
    assert [p["id"] for p in out["panels"]] == ["filters", "km", "compare", "vs-time", "case"]
    assert spec["panels"]["filters"]["type"] == "filters"
    assert spec["panels"]["km"]["config"]["graphType"] == "KaplanMeier"
    assert len(spec["layout"]["items"]) == 5
    assert cx_dashboard_validate.validate_spec(spec)["valid"]


def _wired(suggestion):
    return [p["id"] for p in suggestion["panels"] if p.get("actions")]


def test_the_drill_wires_a_click_only_where_a_click_names_exactly_one_row():
    cohort = sg.suggest_dashboard(copy.deepcopy(_case("cohort: os_months/os_event")["data"]))
    assert _wired(cohort) == ["vs-time"]
    assert [p for p in cohort["panels"] if p["id"] == "vs-time"][0]["actions"] == [
        {"on": "click", "setParam": {"name": "case"}}]
    qc = sg.suggest_dashboard(copy.deepcopy(_case("qc: lane + flowcell")["data"]))
    assert _wired(qc) == ["depth", "quality-scatter"]


def test_the_spec_carries_a_case_param_a_pushdown_source_and_the_click_actions():
    data = copy.deepcopy(_case("cohort: os_months/os_event")["data"])
    out = sg.suggest(data)
    assert out["valid"], out["errors"]
    spec = out["spec"]
    assert spec["params"] == {"case": {"value": "S1"}}
    assert spec["data"]["case"]["value"] == data
    assert spec["data"]["case"]["pushdown"] == {"where": [{"column": "smps", "op": "=", "value": "$case"}]}
    assert "pushdown" not in spec["data"]["data"]            # the charts keep the whole cohort
    assert spec["panels"]["case"]["dataRef"] == "case"
    assert spec["panels"]["case"]["config"]["view"] == "table"
    assert spec["panels"]["vs-time"]["actions"] == [{"on": "click", "setParam": {"name": "case"}}]
    assert "actions" not in spec["panels"]["km"] and "actions" not in spec["panels"]["case"]
    for panel in spec["panels"].values():                    # descriptor-only fields never leak
        assert "card" not in panel and "dashboardConfig" not in panel


def test_the_drill_starts_on_the_first_row_and_titles_match_the_template():
    data = copy.deepcopy(_case("qc: no batch column")["data"])
    data["y"]["smps"] = ["run7-" + s for s in data["y"]["smps"]]
    suggestion = sg.suggest_dashboard(data)
    assert suggestion["drill"] == {"param": "case", "source": "case", "initial": "run7-S1"}
    card = [p for p in suggestion["panels"] if p["id"] == "case"][0]
    assert card["title"].startswith("Sample record")
    assert "click a bar or point to load that sample" in suggestion["reason"]


def test_the_case_card_does_not_count_toward_the_three_panel_minimum():
    # time + event only: filters + Kaplan-Meier is two panels, and the card must not make it three.
    assert _case("cohort: only time + event (too few panels)")["expected"] is None
    assert sg.suggest_dashboard(copy.deepcopy(_case("cohort: only time + event (too few panels)")["data"])) is None


def test_scatter_type_panels_carry_no_engine_transpose_flag():
    # The dashboard renderer transposes row-per-sample data for these itself; a second,
    # engine-level transpose flips the data back (KaplanMeier fails, Scatter2D plots wrong axes).
    data = _case("cohort: os_months/os_event")["data"]
    suggestion = sg.suggest_dashboard(data)
    km = suggestion["panels"][1]
    assert km["config"]["transposeData"] is True          # standalone it is needed
    assert "transposeData" not in km["dashboardConfig"]   # published, it must not be there
    spec = sg.dashboard_spec(suggestion, "t", data)
    assert "transposeData" not in spec["panels"]["km"]["config"]
    assert "transposeData" not in spec["panels"]["vs-time"]["config"]


def test_qc_without_a_batch_column_still_gets_an_ungrouped_quality_boxplot():
    data = _case("qc: no batch column")["data"]
    spread = sg.dashboard_spec(sg.suggest_dashboard(data), "q", data)["panels"]["quality-spread"]["config"]
    assert spread["graphType"] == "Boxplot"
    assert spread["xAxis"] == ["q30"]
    assert "groupingFactors" not in spread


def test_no_published_panel_of_any_template_carries_the_engine_transpose_flag():
    # The engine transposes the dataset in place and every panel shares one data object, so a
    # panel with the flag flips its siblings (a Bar drawn with the metrics on x).
    checked = 0
    for case in FIXTURE:
        suggestion = sg.suggest_dashboard(copy.deepcopy(case["data"]))
        if not suggestion:
            continue
        spec = sg.dashboard_spec(suggestion, "x", case["data"])
        for panel_id, panel in spec["panels"].items():
            if "config" in panel:
                checked += 1
                assert "transposeData" not in panel["config"], (case["name"], panel_id)
    assert checked >= 40


def test_qc_spec_is_valid():
    out = sg.suggest(_case("qc: lane + flowcell")["data"])
    assert out["suggested"] and out["template"] == "ngs-qc" and out["valid"], out.get("errors")
    assert out["spec"]["title"] == "Sequencing run QC"
    assert out["usage"].startswith("<!-- Save the spec as sequencing-run-qc.spec.json")


def test_theme_is_passed_through_and_junk_ignored():
    data = _case("qc: lane + flowcell")["data"]
    assert sg.suggest(data, theme="dark")["spec"]["theme"] == "dark"
    assert sg.suggest(data, theme="neon")["spec"]["theme"] == "auto"


def test_no_match_says_so_and_says_to_retry_then_force():
    # Numeric columns only (no factor to explore by): none of the templates fits.
    data = {"y": {"vars": ["a", "b"], "smps": ["r1", "r2", "r3"], "data": [[1, 2, 3], [4, 5, 7]]}}
    out = sg.suggest(data)
    assert out["suggested"] is False
    assert "5th attempt" in out["reason"] and "force=true" in out["reason"]
    assert "spec" not in out


@pytest.mark.parametrize("junk", [None, {}, [], "x", 3, {"y": {}}, {"y": {"vars": []}}])
def test_junk_input_is_not_a_match(junk):
    assert sg.suggest_dashboard(junk) is None
    assert sg.suggest(junk)["suggested"] is False


def test_suggesting_never_mutates_the_data():
    data = copy.deepcopy(_case("cohort: os_months/os_event")["data"])
    before = json.dumps(data, sort_keys=True)
    sg.suggest(data)
    assert json.dumps(data, sort_keys=True) == before


def test_the_dashboard_opts_in_to_insights_and_passes_declared_thresholds_through():
    data = copy.deepcopy(_case("qc: lane + flowcell")["data"])
    plain = sg.suggest(data)
    assert plain["valid"] and plain["spec"]["insights"] == {}          # data-derived checks only
    declared = [{"field": "q30", "op": "lt", "value": 85, "label": "Q30 below 85%"}]
    out = sg.suggest(data, thresholds=declared)
    assert out["valid"], out["errors"]
    assert out["spec"]["insights"] == {"thresholds": declared}
    declared[0]["value"] = 1                                           # the spec holds its own copy
    assert out["spec"]["insights"]["thresholds"][0]["value"] == 85


def test_an_invalid_threshold_is_reported_by_validation_not_swallowed():
    out = sg.suggest(copy.deepcopy(_case("qc: lane + flowcell")["data"]),
                     thresholds=[{"field": "q30", "op": "near", "value": "x"}])
    assert out["valid"] is False
    assert "spec.insights.thresholds[0].op must be one of: lt, le, gt, ge" in out["errors"]
    assert "spec.insights.thresholds[0].value must be a number" in out["errors"]


# ---------------------------------------------------------------------------
# Bundled datasets, headers+rows input and the generic "explore" template
# ---------------------------------------------------------------------------

import cx_datasets  # noqa: E402


def _rows_from(data):
    """The headers+rows array equivalent of a {y, x} data object (as a caller would send it)."""
    y, x = data["y"], data.get("x") or {}
    headers = list(y["vars"]) + list(x)
    rows = [headers]
    for j in range(len(y["smps"])):
        rows.append([y["data"][i][j] for i in range(len(y["vars"]))] + [x[k][j] for k in x])
    return rows


def test_dataset_penguins_is_a_valid_five_panel_explorer():
    out = sg.suggest(dataset="penguins")
    assert out["suggested"] is True and out["template"] == "explore"
    assert out["valid"], out["errors"]
    assert len(out["panels"]) == 5 and len(out["spec"]["panels"]) == 5
    assert list(out["spec"]["data"]) == ["data"]
    assert {p["dataRef"] for p in out["spec"]["panels"].values()} == {"data"}
    assert [p["graphType"] for p in out["panels"]] == ["filters", "Scatter2D", "Boxplot", "Histogram", "Stacked"]
    assert out["spec"]["panels"]["filters"]["fields"] == ["species", "island", "sex", "year"]
    assert cx_dashboard_validate.validate_spec(out["spec"])["valid"]


def test_penguins_data_is_parsed_as_specified():
    data = cx_datasets.load_dataset("Palmer Penguins")  # case-insensitive alias
    assert len(data["y"]["smps"]) == 344
    assert data["y"]["vars"] == ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    assert None in data["y"]["data"][0]                      # numeric NA -> null
    assert "NA" in data["x"]["sex"]                          # factor NA -> "NA"
    assert "year" in data["x"] and set(data["x"]["year"]) == {"2007", "2008", "2009"}  # year is a factor


def test_explore_panels_follow_the_main_factor():
    spec = sg.suggest(dataset="penguins")["spec"]["panels"]
    assert spec["scatter"]["config"]["colorBy"] == "species"
    assert spec["compare"]["config"]["groupingFactors"] == ["species"]
    assert spec["compare"]["config"]["colorBy"] == "sex"             # the second factor
    assert spec["distribution"]["config"]["groupingFactors"] == ["species"]
    counts = spec["counts"]["config"]
    assert counts["groupingFactors"] == ["sex"] and counts["stackBy"] == "species"
    # The count bar reads the source's calculated n = 1; the histogram flips a copy of its data.
    assert spec["counts"]["measures"] == ["n"]
    assert sg.suggest(dataset="penguins")["spec"]["data"]["data"]["calculatedFields"] == [
        {"name": "n", "formula": "1"}]
    assert spec["distribution"]["transpose"] is True
    assert all("transposeData" not in p.get("config", {}) for p in spec.values())


def test_factor_choice_keeps_na_and_skips_time_columns():
    data = cx_datasets.load_dataset("penguins")
    assert "NA" in data["x"]["sex"]
    out = sg.suggest(data)
    # year has as few levels as the others but is never the main or second factor; it stays a filter.
    assert out["spec"]["panels"]["scatter"]["config"]["colorBy"] == "species"
    assert out["spec"]["panels"]["counts"]["config"]["groupingFactors"] == ["sex"]
    assert "year" in out["spec"]["panels"]["filters"]["fields"]
    # Only time-like factors: nothing to explore by.
    only_year = {"y": data["y"], "x": {"year": data["x"]["year"]}}
    assert sg.suggest(only_year)["suggested"] is False
    # No row is dropped: the NA rows are still in the data the spec carries.
    assert out["spec"]["data"]["data"]["value"]["x"]["sex"].count("NA") == 11


def test_array_input_matches_the_equivalent_object_input():
    data = cx_datasets.load_dataset("penguins")
    from_object = sg.suggest(data)
    from_array = sg.suggest(_rows_from(data))
    assert from_array == from_object


def test_other_bundled_datasets_get_an_explorer():
    for name in ("iris", "mtcars"):
        out = sg.suggest(dataset=name)
        assert out["suggested"] and out["template"] == "explore" and out["valid"], (name, out.get("errors"))
        assert len(out["panels"]) == 5


def test_cohort_and_qc_still_win_over_the_explorer():
    assert sg.suggest(_case("cohort: os_months/os_event")["data"])["template"] == "cohort"
    assert sg.suggest(_case("qc: lane + flowcell")["data"])["template"] == "ngs-qc"


def test_unknown_dataset_lists_the_available_ones():
    out = sg.suggest(dataset="titanic")
    assert out["suggested"] is False
    assert "titanic" in out["error"]
    for name in ("penguins", "iris", "mtcars"):
        assert name in out["error"]


def test_exactly_one_of_data_or_dataset():
    assert "exactly one" in sg.suggest()["error"]
    both = sg.suggest(cx_datasets.load_dataset("iris"), dataset="iris")
    assert both["suggested"] is False and "exactly one" in both["error"]


def test_bad_array_is_a_plain_error():
    assert sg.suggest([["a", "b"]])["suggested"] is False
    assert sg.suggest([["a", "a"], [1, 2]])["suggested"] is False


def test_numeric_means_more_than_ninety_percent_parse():
    rows = [["g", "m", "mostly"]] + [["a", i, "x" if i == 0 else i] for i in range(20)]
    data = cx_datasets.table_to_data(rows)
    assert data["y"]["vars"] == ["m", "mostly"]          # 19/20 = 95% numeric
    rows = [["half", "m"]] + [[1 if i % 2 else "x", i] for i in range(20)]
    assert cx_datasets.table_to_data(rows)["y"]["vars"] == ["m"]                     # 50% numeric: a factor

def test_render_every_penguins_panel():
    """Each panel's config draws something (not blank) in headless Chromium, prepared the way
    the dashboard renderer prepares it. Skipped when Playwright or the engine CDN is unavailable."""
    pytest.importorskip("playwright")
    from PIL import Image
    import io
    import cx_render

    data = cx_datasets.load_dataset("penguins")
    spec = sg.suggest(dataset="penguins")["spec"]
    source = copy.deepcopy(data)
    source["y"]["vars"].append("n")
    source["y"]["data"].append([1] * len(source["y"]["smps"]))      # calculatedFields: n = 1

    def prepare(panel):
        keep = panel.get("measures")
        dat = copy.deepcopy(source)
        if keep:
            rows = [dat["y"]["data"][dat["y"]["vars"].index(m)] for m in keep]
            dat["y"] = {"vars": list(keep), "smps": dat["y"]["smps"], "data": rows}
        if panel.get("transpose") or (panel["config"]["graphType"] == "Scatter2D"):
            y = dat["y"]
            flipped = {"y": {"vars": y["smps"], "smps": y["vars"],
                             "data": [[y["data"][i][j] for i in range(len(y["vars"]))]
                                      for j in range(len(y["smps"]))]}}
            if dat.get("x"):
                flipped["z"] = dat["x"]
            dat = flipped
        return dat

    for panel_id, panel in spec["panels"].items():
        if panel.get("type") == "filters":
            continue
        try:
            png = cx_render.render_png(prepare(panel), panel["config"], width=700, height=500)
        except RuntimeError as exc:
            pytest.skip(str(exc))
        colors = len(set(Image.open(io.BytesIO(png)).convert("RGB").getdata()))
        assert colors > 20, "panel %s rendered blank (%d colours)" % (panel_id, colors)


def test_result_does_not_invite_a_widget_redraw_but_explain_has_the_fallback():
    import cx_dashboard
    assert "render_inline" not in sg.suggest(dataset="penguins")
    topic = cx_dashboard.explain("inline_widget")["explanation"]
    assert "/*SPEC_JSON*/" in topic and "canvasxpress@72.9.1" in topic and "FALLBACK ONLY" in topic


def test_no_scheme_box_auto_theme_and_builder_link(monkeypatch):
    monkeypatch.delenv("CXD_DASHBOARDS_URL", raising=False)
    out = sg.suggest(dataset="penguins")
    assert out["spec"]["theme"] == "auto"
    assert sg.suggest(dataset="penguins", theme="dark")["spec"]["theme"] == "dark"
    filters = [p for p in out["spec"]["panels"].values() if p.get("type") == "filters"]
    assert filters and all(p["saveSchemes"] is False for p in filters)
    assert out["builder_url"] == "https://canvasxpress.org/dashboards"
    monkeypatch.setenv("CXD_DASHBOARDS_URL", "http://localhost:8200/")
    assert sg.suggest(dataset="iris")["builder_url"] == "http://localhost:8200"


# ---------------------------------------------------------------------------
# The retry-then-force fallback: errors say what is wrong, and force=True builds a filter panel
# plus bar charts from whatever the table has.
# ---------------------------------------------------------------------------
QC_TABLE = [["sample", "lane", "total_reads", "pct_mapped"],
            ["S1", "L1", 100, 95.0], ["S2", "L2", 200, 90.0], ["S3", "L1", 150, 92.0]]


def _discrete_table(columns):
    names = ["sample"] + columns
    return [names] + [["S%d" % i] + ["%s%d" % (c[0], i % (2 + k)) for k, c in enumerate(columns)]
                      for i in range(1, 25)]


def _bars(result):
    return [p for p in result["panels"] if p["graphType"] == "Bar"]


def test_force_builds_a_filter_panel_and_a_bar_of_the_first_numeric_column():
    result = sg.suggest(copy.deepcopy(QC_TABLE), force=True)
    assert result["suggested"] and result["template"] == "minimal" and result["valid"], result
    assert [p["id"] for p in result["panels"]] == ["filters", "bar"]
    bar = result["spec"]["panels"]["bar"]["config"]
    assert bar["graphType"] == "Bar" and bar["xAxis"] == ["total_reads"]
    assert result["spec"]["panels"]["filters"]["fields"]


def test_force_with_only_discrete_columns_adds_a_count_bar_per_column_up_to_four():
    result = sg.suggest(_discrete_table(["lane", "batch", "library", "site", "sex"]), force=True)
    assert result["suggested"] and result["valid"], result
    bars = _bars(result)
    assert len(bars) == 4
    for panel in bars:
        config = result["spec"]["panels"][panel["id"]]["config"]
        assert config["xAxis"] == ["n"] and config["summaryType"] == "sum" and len(config["groupingFactors"]) == 1
    assert "n" not in result["spec"]["panels"]["filters"]["fields"]


def test_force_with_one_discrete_column_gives_one_wide_count_bar():
    result = sg.suggest(_discrete_table(["lane"]), force=True)
    items = {i["panel"]: i for i in result["spec"]["layout"]["items"]}
    assert len(_bars(result)) == 1 and items["bar-1"]["w"] == 9


def test_force_skips_an_id_like_discrete_column_when_others_exist():
    table = _discrete_table(["lane", "batch"])
    for i, row in enumerate(table[1:]):
        row.append("unique%d" % i)
    table[0].append("barcode")
    titles = [p["title"] for p in _bars(sg.suggest(table, force=True))]
    assert "Count by barcode" not in titles and len(titles) == 2


def test_force_does_not_read_generated_row_ids_as_a_measure():
    # No id column: the rows are numbered 1..n, which must not become a measure called "Id".
    table = [["lane", "total_reads"], ["L1", 100], ["L2", 200], ["L1", 150]]
    result = sg.suggest(table, force=True)
    assert result["spec"]["panels"]["bar"]["config"]["xAxis"] == ["total_reads"]
    assert "Id" not in result["spec"]["panels"]["filters"]["fields"]


def test_force_needs_a_data_row():
    result = sg.suggest([["sample", "lane"]], force=True)
    assert result["suggested"] is False and result["error"]


def test_an_object_without_y_gets_an_error_that_names_the_accepted_shapes():
    result = sg.suggest({"headers": QC_TABLE[0], "rows": QC_TABLE[1:]})
    assert result["suggested"] is False
    assert "list of lists" in result["error"] and "headers" in result["error"]


def test_a_table_without_numbers_points_straight_at_force():
    result = sg.suggest([["a", "b"], ["x", "y"], ["z", "w"]])
    assert result["suggested"] is False
    assert "force=true" in result["error"] and "only discrete columns" in result["error"]


def test_tool_description_gives_the_retry_then_force_instruction_and_no_widget_fallback():
    import server
    tool = server.suggest_canvasxpress_dashboard
    text = getattr(tool, "description", None) or ""
    if not text:  # FastMCP versions differ in what the decorator returns
        import asyncio
        text = asyncio.run(server.mcp.get_tool("suggest_canvasxpress_dashboard")).description
    assert "up to 5 attempts" in text and "force=true" in text
    assert "do not redraw it in an HTML widget" in text
    assert "inline_widget" not in text
