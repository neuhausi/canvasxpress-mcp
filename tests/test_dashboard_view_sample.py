"""The dashboard view with a SAMPLE result: the banner, and loading the full file in the page.

The tool result of a sample carries ``sample`` (rows / total_rows). The view says so and lets the user
load the whole file: parsed in the page (never through the model), shaped like the sample (same numeric
and annotation columns, rows numbered 1..N) and swapped into the same spec.

Runs when Playwright is installed and both CX_TEST_ENGINE (a canvasXpress.min.js file) and
CX_TEST_DASHBOARDS (a canvasxpress-dashboards UMD file) are set; they are served in place of the CDN
copies, so the test needs no network. Otherwise it skips.
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

COLUMNS = ["sample", "lane", "batch", "total_reads", "pct_mapped", "pct_duplication", "pct_q30"]
FULL_ROWS = 600
SAMPLE_ROWS = 120


def _row(i):
    rng = random.Random(i)
    return ["S%04d" % i, "L%d" % (1 + i % 4), ["A", "B", "C"][i % 3], 30000000 + rng.randrange(40000000),
            round(rng.uniform(80, 99), 1), round(rng.uniform(5, 30), 1), round(rng.uniform(85, 98), 1)]


def _csv(rows, delim=","):
    return "\n".join(delim.join(str(c) for c in r) for r in rows) + "\n"


@pytest.fixture(scope="module")
def result():
    full = [_row(i) for i in range(1, FULL_ROWS + 1)]
    sample = [COLUMNS] + full[::5][:SAMPLE_ROWS]
    return full, sg.suggest(sample, total_rows=FULL_ROWS)


@pytest.fixture(scope="module")
def host(result):
    playwright = pytest.importorskip("playwright.sync_api")
    if not (ENGINE and os.path.isfile(ENGINE) and DASHBOARDS and os.path.isfile(DASHBOARDS)):
        pytest.skip("set CX_TEST_ENGINE and CX_TEST_DASHBOARDS to run")
    _, suggestion = result
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
            page = context.new_page()
            page.set_content(HOST)
            page.evaluate("p => { window.PAYLOAD = p; }", json.dumps(suggestion))
            page.evaluate("h => { document.getElementById('f').srcdoc = h; }", open(VIEW, encoding="utf-8").read())
            page.frame_locator("#f").locator("#cx-sample-msg").wait_for(timeout=30000)
            page.wait_for_timeout(2500)
            yield page, suggestion
        finally:
            browser.close()


def _filter_totals(frame, label):
    """The total number of rows a discrete filter accounts for: the sum of its values' counts.

    The Filters panel text reads: field name, '(Select All)', then value and 'n / n' lines; the
    field's block runs to the next field name."""
    lines = [ln.strip() for ln in frame.locator("#cx-dashboard .cxd-filters").first.inner_text().splitlines()
             if ln.strip()]
    starts = [k for k in range(len(lines) - 1) if lines[k + 1].startswith("(Select")]
    for n, k in enumerate(starts):
        if lines[k] != label:
            continue
        end = starts[n + 1] if n + 1 < len(starts) else len(lines)
        counts = [int(c) for c in re.findall(r"(\d+)\s*/\s*\d+", " ".join(lines[k:end]))]
        return sum(counts)
    raise AssertionError("no filter named %r in %r" % (label, lines[:20]))


def test_a_sample_is_never_drawn_the_view_waits_for_the_complete_file(host):
    page, suggestion = host
    frame = page.frame_locator("#f")
    message = frame.locator("#cx-sample-msg").inner_text()
    assert "never drawn from a sample" in message and "Load your complete file (%d rows)" % FULL_ROWS in message
    # nothing from the sample is on screen: no filters, no charts, no KPI cards
    assert frame.locator("#cx-dashboard .cxd-filters").count() == 0
    assert frame.locator("#cx-dashboard canvas").count() == 0
    assert frame.locator("#cx-load").is_visible()
    assert frame.locator("#cx-save").is_disabled() and frame.locator("#cx-builder").is_disabled()
    assert suggestion["sample"]["total_rows"] == FULL_ROWS


def test_loading_the_complete_file_builds_the_dashboard_from_all_rows(host, result, tmp_path):
    page, _ = host
    frame = page.frame_locator("#f")
    full, _s = result
    path = tmp_path / "full.csv"
    path.write_text(_csv([COLUMNS] + full))
    frame.locator("#cx-file").set_input_files(str(path))
    page.wait_for_function(
        "document.getElementById('f').contentDocument.getElementById('cx-sample-msg').textContent.indexOf('Showing all') === 0",
        timeout=30000)
    page.wait_for_timeout(2500)
    assert "Showing all %d rows from full.csv" % FULL_ROWS in frame.locator("#cx-sample-msg").inner_text()
    assert _filter_totals(frame, "batch") == FULL_ROWS and _filter_totals(frame, "lane") == FULL_ROWS
    # drawn from the complete data only; Save / Open in Builder work now (and carry the full table)
    assert frame.locator("#cx-save").is_enabled() and frame.locator("#cx-builder").is_enabled()


def test_a_file_without_the_dashboards_columns_is_refused_with_the_names(host, tmp_path):
    page, _ = host
    frame = page.frame_locator("#f")
    path = tmp_path / "other.csv"
    path.write_text(_csv([["sample", "lane", "colour"], ["S1", "L1", "red"], ["S2", "L2", "blue"]]))
    frame.locator("#cx-file").set_input_files(str(path))
    page.wait_for_timeout(1500)
    message = frame.locator("#cx-sample-msg").inner_text()
    assert "no column" in message and "total_reads" in message


def test_a_tab_separated_file_loads_too(host, result, tmp_path):
    page, _ = host
    frame = page.frame_locator("#f")
    full, _s = result
    path = tmp_path / "full.tsv"
    path.write_text(_csv([COLUMNS] + full[:300], delim="\t"))
    frame.locator("#cx-file").set_input_files(str(path))
    page.wait_for_function(
        "document.getElementById('f').contentDocument.getElementById('cx-sample-msg').textContent.indexOf('Showing all 300') === 0",
        timeout=30000)
    page.wait_for_timeout(2500)
    assert _filter_totals(frame, "batch") == 300


def test_the_view_script_is_shipped_with_the_loader():
    text = open(VIEW, encoding="utf-8").read()
    assert "Load full file" in text and "buildFullData" in text and "payload.sample" in text
