"""The dashboard view's insights strip and its date handling, in a browser.

Insights are computed in the view from the COMPLETE inline data (never from a sample): declared limits,
a group far from the overall average, and values far from the mean. The view's name humanizing is compared with the server's on the same inputs.

Runs when Playwright is installed and CX_TEST_ENGINE and CX_TEST_DASHBOARDS are set (served in place of
the CDN copies, no network); otherwise it skips.
"""

import json
import os
import random
import re
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

import cx_dashboard_suggest as sg  # noqa: E402

ENGINE = os.environ.get("CX_TEST_ENGINE", "")
DASHBOARDS = os.environ.get("CX_TEST_DASHBOARDS", "")
VIEW = os.path.join(ROOT, "src", "ui", "cx_dashboard_view.html")
CDN_PREFIX = "https://www.canvasxpress.org/"

HOST = """<body style="margin:0"><iframe id="f" style="width:900px;height:900px;border:0"></iframe><script>
addEventListener('message', function (e) {
  var m = e.data; if (!m || !m.method) return;
  if (m.method === 'ui/initialize') { e.source.postMessage({jsonrpc: '2.0', id: m.id, result: {}}, '*'); }
  else if (m.method === 'ui/notifications/initialized') {
    e.source.postMessage({jsonrpc: '2.0', method: 'ui/notifications/tool-result',
                          params: {content: [{type: 'text', text: window.PAYLOAD}]}}, '*');
  }
});
</script></body>"""


@pytest.fixture(scope="module")
def browser_context():
    playwright = pytest.importorskip("playwright.sync_api")
    if not (ENGINE and os.path.isfile(ENGINE) and DASHBOARDS and os.path.isfile(DASHBOARDS)):
        pytest.skip("set CX_TEST_ENGINE and CX_TEST_DASHBOARDS to run")
    with playwright.sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            context = browser.new_context(viewport={"width": 900, "height": 900})

            def serve(route):
                url = route.request.url
                if url.endswith("canvasXpress.min.js"):
                    route.fulfill(path=ENGINE, content_type="application/javascript")
                elif url.endswith("canvasxpress-dashboards.umd.js"):
                    route.fulfill(path=DASHBOARDS, content_type="application/javascript")
                elif url.endswith(".css"):
                    route.fulfill(body="", content_type="text/css")
                else:
                    route.abort()

            context.route(CDN_PREFIX + "**", serve)
            yield context
        finally:
            browser.close()


def open_view(context, result, wait_for):
    """Open the view with a tool result; returns (page, frame) once `wait_for` (a selector in the frame) exists."""
    page = context.new_page()
    page.set_content(HOST)
    page.evaluate("p => { window.PAYLOAD = p; }", json.dumps(result))
    page.evaluate("h => { document.getElementById('f').srcdoc = h; }", open(VIEW, encoding="utf-8").read())
    frame = page.frame_locator("#f")
    frame.locator(wait_for).first.wait_for(timeout=30000)
    page.wait_for_timeout(2500)
    return page, frame


def call(page, expression, *args):
    """Evaluate a function of the view's page (a top-level function of its script)."""
    inner = page.frames[1] if len(page.frames) > 1 else page.main_frame
    return inner.evaluate(expression, *args) if args else inner.evaluate(expression)


def _data(rows, seed=3):
    rng = random.Random(seed)
    lanes = ["L%d" % (1 + i % 4) for i in range(rows)]
    return lanes, rng


def _table(rows=120, shift=0.0, seed=3):
    """sample, lane, a, b: `a` is lower by `shift` SD in lane L3."""
    lanes, rng = _data(rows, seed)
    out = [["sample", "lane", "a", "b"]]
    for i, lane in enumerate(lanes):
        a = rng.gauss(50, 5) + (-shift * 5 if lane == "L3" else 0)
        out.append(["S%03d" % i, lane, round(a, 2), round(rng.gauss(10, 2), 2)])
    return out


def test_a_planted_bad_group_is_called_out_with_its_figures(browser_context):
    result = sg.suggest(_table(shift=2.5))
    page, frame = open_view(browser_context, result, "#cx-insights.on")
    text = frame.locator("#cx-insights").inner_text()
    assert "What stands out" in text and "computed from all 120 rows" in text
    assert "Lane L3: A averages" in text and "below the overall" in text and "of 120 rows" in text
    page.close()


def test_pure_noise_gets_no_group_call_outs(browser_context):
    data = sg.suggest(_table(shift=0.0, seed=9))["spec"]["data"]["data"]["value"]
    page, _frame = open_view(browser_context, sg.suggest(_table(shift=0.0, seed=9)), "#cx-dashboard .cxd-filters")
    found = call(page, "d => computeInsights(d, [], [])", data)
    assert [c for c in found if " averages " in c["text"]] == []
    page.close()


def test_declared_limits_are_counted_and_named(browser_context):
    table = _table(shift=0.0, seed=4)
    result = sg.suggest(table, thresholds=[{"field": "b", "op": "lt", "value": 8, "label": "B below 8"},
                                           {"field": "a", "op": "gt", "value": 9999, "label": "A above 9999"}])
    below = sum(1 for r in table[1:] if r[3] < 8)
    page, frame = open_view(browser_context, result, "#cx-insights.on")
    text = frame.locator("#cx-insights").inner_text()
    assert "B below 8: %d of 120 rows" % below in text
    assert "A above 9999: none of 120 rows" in text
    page.close()


def test_values_far_from_the_mean_are_counted(browser_context):
    table = _table(shift=0.0, seed=5)
    for row in table[1:4]:
        row[3] = 80.0                                                    # three values ~35 SD above the mean of 10
    page, _frame = open_view(browser_context, sg.suggest(table), "#cx-dashboard .cxd-filters")
    found = call(page, "d => computeInsights(d, [], [])", sg.suggest(table)["spec"]["data"]["data"]["value"])
    assert any("3 rows have B above" in c["text"] for c in found), found
    page.close()


@pytest.mark.parametrize("name", ["total_reads", "pct_mapped", "pct_q30", "humidity_pct", "OS_months", "gc_content",
                                  "gcContent", "contamination_freemix", "GENE12", "bmi", "age", "unit.price",
                                  "city_60_levels", "Rows", "_"])
def test_the_view_humanizes_names_like_the_server(browser_context, name):
    page, _frame = open_view(browser_context, sg.suggest(_table()), "#cx-dashboard .cxd-filters")
    assert call(page, "n => humanize(n)", name) == sg._humanize(name)
    page.close()
