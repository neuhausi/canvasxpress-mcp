"""A large dashboard result never goes through the host whole: the tool returns a short result with a
result_id, and the view fetches the full one in pieces through the host (tools/call). A host that
replaces the result with an error text still leaves a visible line, never a blank panel.

Runs when Playwright is installed and CX_TEST_ENGINE / CX_TEST_DASHBOARDS are set; otherwise it skips.
"""

import json
import os
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from test_dashboard_view_insights import browser_context, _table  # noqa: E402,F401

VIEW = os.path.join(ROOT, "src", "ui", "cx_dashboard_view.html")

# A host that delivers window.PAYLOAD as the tool result and answers tools/call from window.PARTS.
HOST = """<body style="margin:0"><iframe id="f" style="width:900px;height:900px;border:0"></iframe><script>
window.calls = [];
addEventListener('message', function (e) {
  var m = e.data; if (!m || !m.method) return;
  if (m.method === 'ui/initialize') { e.source.postMessage({jsonrpc: '2.0', id: m.id, result: {}}, '*'); }
  else if (m.method === 'ui/notifications/initialized') {
    e.source.postMessage({jsonrpc: '2.0', method: 'ui/notifications/tool-result',
                          params: {content: [{type: 'text', text: window.PAYLOAD}]}}, '*');
  } else if (m.method === 'tools/call') {
    window.calls.push(m.params);
    var a = m.params.arguments, text = window.PARTS[String(a.part)];
    var out = text === undefined ? {error: 'part must be between 0 and ' + (Object.keys(window.PARTS).length - 1) + '.'}
                                 : {result_id: a.result_id, part: a.part, parts: Object.keys(window.PARTS).length, text: text};
    e.source.postMessage({jsonrpc: '2.0', id: m.id, result: {content: [{type: 'text', text: JSON.stringify(out)}]}}, '*');
  }
});
</script></body>"""


@pytest.fixture(scope="module")
def held():
    import server
    full = server.cx_dashboard_suggest.suggest(_table(rows=400))
    short = server._hold_dashboard_result(full)
    assert "spec" not in short and short["parts"] >= 1 and short["result_id"] in server._DASH_RESULTS
    parts = {str(i): server.get_canvasxpress_dashboard_result(short["result_id"], i)["text"] for i in range(short["parts"])}
    return full, short, parts


def _open(context, payload_text, parts):
    page = context.new_page()
    page.set_content(HOST)
    page.evaluate("a => { window.PAYLOAD = a[0]; window.PARTS = a[1]; }", [payload_text, parts])
    page.evaluate("h => { document.getElementById('f').srcdoc = h; }", open(VIEW, encoding="utf-8").read())
    return page, page.frame_locator("#f")


def test_the_short_result_is_fetched_in_pieces_and_drawn(browser_context, held):
    full, short, parts = held
    page, frame = _open(browser_context, json.dumps(short), parts)
    frame.locator("#cx-dashboard .cxd-filters").first.wait_for(timeout=30000)
    page.wait_for_timeout(2500)
    assert page.evaluate("window.calls.length") == short["parts"]
    assert page.evaluate("window.calls[0].name") == "get_canvasxpress_dashboard_result"
    assert frame.locator("#cx-dashboard canvas").count() > 0
    spec = page.frames[1].evaluate("currentPayload.spec")
    assert spec == full["spec"]                                             # Save / Builder carry the full spec
    assert frame.locator("#cx-error").is_hidden()
    page.close()


def test_a_result_the_host_replaced_with_an_error_text_still_shows_a_line(browser_context, held):
    _full, short, parts = held
    page, frame = _open(browser_context, "Error: result (239,748 characters) exceeds maximum allowed tokens. Output has been saved to /x.txt", parts)
    frame.locator("#cx-error").wait_for(state="visible", timeout=15000)
    assert "did not pass the dashboard result" in frame.locator("#cx-error").inner_text()
    page.close()


def test_a_result_id_inside_a_mangled_text_is_still_used(browser_context, held):
    _full, short, parts = held
    page, frame = _open(browser_context, 'Error: truncated {"result_id":"%s","parts":%d' % (short["result_id"], short["parts"]), parts)
    frame.locator("#cx-dashboard .cxd-filters").first.wait_for(timeout=30000)
    assert frame.locator("#cx-dashboard canvas").count() > 0
    page.close()


def test_an_expired_id_is_said_on_screen(browser_context, held):
    _full, short, _parts = held
    page, frame = _open(browser_context, json.dumps(dict(short, result_id="gone")), {})
    frame.locator("#cx-error").wait_for(state="visible", timeout=15000)
    assert "Could not fetch the dashboard" in frame.locator("#cx-error").inner_text()
    page.close()


def test_the_held_result_tool_and_the_short_result_shape():
    import server
    full = server.cx_dashboard_suggest.suggest(_table())
    short = server._hold_dashboard_result(full)
    assert set(short) >= {"result_id", "parts", "size", "template", "panels", "valid", "builder_url", "usage"}
    assert len(json.dumps(short)) < 6000
    pieces = [server.get_canvasxpress_dashboard_result(short["result_id"], i)["text"] for i in range(short["parts"])]
    assert json.loads("".join(pieces)) == full
    assert "error" in server.get_canvasxpress_dashboard_result("nope")
    assert "error" in server.get_canvasxpress_dashboard_result(short["result_id"], short["parts"])
    declined = server._hold_dashboard_result({"suggested": False, "reason": "x"})
    assert declined == {"suggested": False, "reason": "x"}
