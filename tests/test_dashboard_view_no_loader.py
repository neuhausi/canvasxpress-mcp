"""The dashboard view has no "load a file" path: a dashboard is drawn only from the data the tool returned.

A result that still carries ``sample`` (an older server) draws nothing and says why. Runs when Playwright is
installed and CX_TEST_ENGINE / CX_TEST_DASHBOARDS are set; otherwise it skips.
"""

import json
import os
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(__file__))

import cx_dashboard_suggest as sg  # noqa: E402
from test_dashboard_view_insights import browser_context, open_view, _table  # noqa: E402,F401

VIEW = os.path.join(ROOT, "src", "ui", "cx_dashboard_view.html")


def test_the_view_source_has_no_file_loader():
    text = open(VIEW, encoding="utf-8").read()
    for gone in ("Load full file", "cx-load", "cx-file", "loadFullFile", "buildFullData", "Showing all", "dragover"):
        assert gone not in text, gone


def test_a_dashboard_is_drawn_with_no_banner_and_no_button(browser_context):
    page, frame = open_view(browser_context, sg.suggest(_table()), "#cx-dashboard .cxd-filters")
    assert frame.locator("#cx-sample").count() == 0 and frame.locator("button", has_text="Load").count() == 0
    assert frame.locator("#cx-dashboard canvas").count() > 0
    page.close()


def test_a_result_that_is_a_sample_draws_nothing(browser_context):
    result = sg.suggest(_table())
    result["sample"] = {"rows": 120, "total_rows": 600}
    page, frame = open_view(browser_context, result, "#cx-error")
    assert "never drawn from a sample" in frame.locator("#cx-error").inner_text()
    assert frame.locator("#cx-dashboard canvas").count() == 0
    page.close()


def test_insights_ignore_columns_looked_up_from_another_file(browser_context):
    lanes = [["sample", "lane", "a", "b"]]
    rows = _table(rows=120, shift=0.0, seed=5)
    rows[0] = ["sample", "lane", "a", "b"]
    sites = [["lane", "beds"], ["L1", 100], ["L2", 200], ["L3", 900], ["L4", 300]]
    result = sg.suggest(tables=[{"name": "runs", "data": rows}, {"name": "lanes", "data": sites}])
    assert result["joined"]["lookup_columns"] == ["beds"]
    page, frame = open_view(browser_context, result, "#cx-dashboard .cxd-filters")
    text = frame.locator("#cx-insights").inner_text() if frame.locator("#cx-insights.on").count() else ""
    assert "Beds" not in text
    page.close()
