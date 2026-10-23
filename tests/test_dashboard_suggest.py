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
    assert "theme" not in sg.suggest(data, theme="neon")["spec"]


def test_no_match_says_so_and_points_at_the_composer():
    out = sg.suggest(_case("none: sales table")["data"])
    assert out["suggested"] is False
    assert "compose_canvasxpress_dashboard" in out["reason"]
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
