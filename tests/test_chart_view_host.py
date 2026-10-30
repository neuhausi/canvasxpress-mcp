"""The chart view against a strict MCP Apps host.

A strict host (Claude) answers ui/initialize and delivers nothing until the VIEW sends
ui/notifications/initialized. The stand-in host here does the same, so a view that never sends it
renders nothing. Runs when Playwright is installed and CX_TEST_ENGINE points at a canvasXpress.min.js
file (served in place of the CDN copy, so the test needs no network); otherwise it skips.
"""

import io
import json
import os
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

ENGINE = os.environ.get("CX_TEST_ENGINE", "")
VIEW = os.path.join(ROOT, "src", "ui", "cx_chart_view.html")
CDN = "https://www.canvasxpress.org/dist/canvasXpress.min.js"

# The host page embeds the view in an iframe and plays the host's side of the protocol.
HOST = """<body style="margin:0"><iframe id="f" style="width:700px;height:450px;border:0"></iframe><script>
window.log = [];
addEventListener('message', function (e) {
  var m = e.data; if (!m || !m.method) return;
  window.log.push(m.method);
  if (m.method === 'ui/initialize') { e.source.postMessage({jsonrpc: '2.0', id: m.id, result: {}}, '*'); }
  else if (m.method === 'ui/notifications/initialized') {   // strict: nothing is sent before this
    e.source.postMessage({jsonrpc: '2.0', method: 'ui/notifications/tool-result',
                          params: {content: [{type: 'text', text: window.PAYLOAD}]}}, '*');
  }
});
</script></body>"""

PAYLOAD = json.dumps({
    "config": {"graphType": "Scatter2D", "xAxis": ["x"], "yAxis": ["y"]},
    "data": [["x", "y"], [1, 2], [2, 3], [3, 5], [4, 4], [5, 8]],
})


@pytest.fixture(scope="module")
def page():
    playwright = pytest.importorskip("playwright.sync_api")
    if not ENGINE or not os.path.isfile(ENGINE):
        pytest.skip("set CX_TEST_ENGINE to a canvasXpress.min.js file to run")
    with playwright.sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            context = browser.new_context(viewport={"width": 700, "height": 450})
            context.route(CDN, lambda route: route.fulfill(
                path=ENGINE, content_type="application/javascript"))
            host = context.new_page()
            host.set_content(HOST)
            host.evaluate("p => { window.PAYLOAD = p; }", PAYLOAD)
            view_html = open(VIEW, encoding="utf-8").read()
            host.evaluate("h => { document.getElementById('f').srcdoc = h; }", view_html)
            yield host
        finally:
            browser.close()


def _colors(host) -> int:
    from PIL import Image
    png = host.locator("#f").screenshot()
    return len(set(Image.open(io.BytesIO(png)).convert("RGB").getdata()))


def test_view_sends_initialized_after_initialize(page):
    page.wait_for_function("window.log.indexOf('ui/notifications/initialized') !== -1", timeout=15000)
    log = page.evaluate("window.log")
    assert log.index("ui/initialize") < log.index("ui/notifications/initialized")


def test_chart_is_drawn_once_the_host_delivers_the_result(page):
    pytest.importorskip("PIL")
    page.wait_for_timeout(2500)  # the engine draws asynchronously after the result arrives
    assert _colors(page) > 40, "the chart view drew nothing (blank iframe)"
