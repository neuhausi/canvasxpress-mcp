"""The drop zone: a suggest call with no data returns {drop: true}; the view shows "Drag your file here",
uploads a dropped file within its origin, calls suggest_canvasxpress_dashboard(url=...) through the host and
draws the dashboard. Nothing goes through the model.

Runs when Playwright is installed and CX_TEST_ENGINE / CX_TEST_DASHBOARDS are set; otherwise it skips.
"""

import csv
import io
import json
import os
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from test_dashboard_view_insights import browser_context, _table  # noqa: E402,F401
import server  # noqa: E402

VIEW = os.path.join(ROOT, "src", "ui", "cx_dashboard_view.html")

# A host that delivers window.PAYLOAD, then answers tools/call for suggest (from window.SUGGEST) and for the
# held parts (from window.PARTS).
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
    var text, a = m.params.arguments;
    if (m.params.name === 'suggest_canvasxpress_dashboard') text = window.SUGGEST;
    else if (m.params.name === 'upload_canvasxpress_file') {
      window.pieces = window.pieces || {}; window.pieces[a.part] = a.text;
      text = JSON.stringify(Object.keys(window.pieces).length < a.parts ? {upload_id: 'u1', received: Object.keys(window.pieces).length, parts: a.parts}
                            : {file_id: 'abc', url: 'https://canvasxpress.org/mcp/file/abc', name: a.name, rows: 120, columns: 4});
    } else { text = JSON.stringify({result_id: a.result_id, part: a.part,
             parts: Object.keys(window.PARTS).length, text: window.PARTS[String(a.part)]}); }
    e.source.postMessage({jsonrpc: '2.0', id: m.id, result: {content: [{type: 'text', text: text}]}}, '*');
  }
});
</script></body>"""


def _csv(table):
    out = io.StringIO()
    csv.writer(out).writerows(table)
    return out.getvalue()


def test_no_data_call_returns_the_drop_instruction():
    out = server.suggest_canvasxpress_dashboard()
    assert out["drop"] is True and out["suggested"] is False and out["upload_url"].endswith("/mcp/upload")
    assert "drag the file" in out["say"]


def test_a_dropped_file_becomes_the_dashboard(browser_context):
    table = _table()
    short = server.suggest_canvasxpress_dashboard(data=table, title="dropped")
    parts = {str(i): server.get_canvasxpress_dashboard_result(short["result_id"], i)["text"] for i in range(short["parts"])}
    page = browser_context.new_page()
    page.set_content(HOST)
    page.evaluate("p => { window.PAYLOAD = p.drop; window.SUGGEST = p.suggest; window.PARTS = p.parts; }",
                  {"drop": json.dumps(server.suggest_canvasxpress_dashboard()), "suggest": json.dumps(short), "parts": parts})
    page.evaluate("h => { document.getElementById('f').srcdoc = h; }", open(VIEW, encoding="utf-8").read())
    frame = page.frame_locator("#f")
    frame.locator("#cx-drop.on").wait_for(timeout=30000)
    assert "Drag your file here" in frame.locator("#cx-drop").inner_text()

    text = _csv(table)
    frame.locator("#cx-drop-input").set_input_files({"name": "qc.csv", "mimeType": "text/csv", "buffer": text.encode()})
    frame.locator("#cx-dashboard .cxd-filters").wait_for(timeout=30000)
    page.wait_for_timeout(1500)
    calls = page.evaluate("() => window.calls")
    # the file went through the host in one piece (it is small), never over the network from the iframe
    assert calls[0]["name"] == "upload_canvasxpress_file" and calls[0]["arguments"]["text"] == text and calls[0]["arguments"]["parts"] == 1
    assert calls[1] == {"name": "suggest_canvasxpress_dashboard", "arguments": {"url": "https://canvasxpress.org/mcp/file/abc"}}
    assert frame.locator("#cx-drop").evaluate("el => el.classList.contains('on')") is False
    assert frame.locator("#cx-dashboard canvas").count() >= 3
    page.close()


def test_a_big_dropped_file_goes_through_the_host_in_pieces(browser_context):
    table = _table(rows=3000)
    short = server.suggest_canvasxpress_dashboard(data=table, title="big")
    parts = {str(i): server.get_canvasxpress_dashboard_result(short["result_id"], i)["text"] for i in range(short["parts"])}
    page = browser_context.new_page()
    page.set_content(HOST)
    page.evaluate("p => { window.PAYLOAD = p.drop; window.SUGGEST = p.suggest; window.PARTS = p.parts; }",
                  {"drop": json.dumps(server.suggest_canvasxpress_dashboard()), "suggest": json.dumps(short), "parts": parts})
    page.evaluate("h => { document.getElementById('f').srcdoc = h; }", open(VIEW, encoding="utf-8").read())
    frame = page.frame_locator("#f")
    frame.locator("#cx-drop.on").wait_for(timeout=30000)
    text = _csv(table)
    frame.locator("#cx-drop-input").set_input_files({"name": "big.csv", "mimeType": "text/csv", "buffer": text.encode()})
    frame.locator("#cx-dashboard .cxd-filters").wait_for(timeout=60000)
    calls = page.evaluate("() => window.calls")
    ups = [c["arguments"] for c in calls if c["name"] == "upload_canvasxpress_file"]
    assert len(ups) > 1 and "".join(u["text"] for u in ups) == text and all(u["parts"] == len(ups) for u in ups)
    assert all(u.get("upload_id") == "u1" for u in ups[1:]) and "upload_id" not in ups[0]
    page.close()


def test_a_file_sent_in_pieces_through_the_host_becomes_a_link():
    text = _csv(_table())
    pieces = [text[i:i + 100] for i in range(0, len(text), 100)]
    out = server.upload_canvasxpress_file("qc.csv", pieces[0], 0, len(pieces))
    assert out["received"] == 1 and out["parts"] == len(pieces)
    for i in range(1, len(pieces)):
        out = server.upload_canvasxpress_file("qc.csv", pieces[i], i, len(pieces), out["upload_id"])
    assert out["rows"] == 120 and out["url"] == server._FILE_URL_BASE + out["file_id"]
    assert server.suggest_canvasxpress_dashboard(url=out["url"])["suggested"]
    assert "expired" in server.upload_canvasxpress_file("x.csv", "a", 1, 2, "nope")["error"]


def test_several_links_build_one_joined_dashboard(monkeypatch):
    patients = [["patient_id", "site_id", "age", "bmi", "OS_months", "OS_event"]] + [
        ["P%03d" % i, "S%d" % (i % 3), 40 + i % 30, 20 + i % 12, 6 + i % 40, i % 2] for i in range(120)]
    sites = [["site_id", "country"], ["S0", "US"], ["S1", "DE"], ["S2", "JP"]]
    files = {"https://x.org/patients.csv": ("patients.csv", _csv(patients)), "https://x.org/sites.csv": ("sites.csv", _csv(sites))}
    monkeypatch.setattr(server, "_public_host", lambda host: True)
    monkeypatch.setattr(server, "_open_url", lambda u: (files[u][0], files[u][1].encode()))
    out = server.suggest_canvasxpress_dashboard(urls=list(files))
    assert out["suggested"], out
    full = json.loads("".join(server.get_canvasxpress_dashboard_result(out["result_id"], i)["text"] for i in range(out["parts"])))
    assert full["joined"]["base"] == "patients" and [j["table"] for j in full["joined"]["joins"]] == ["sites"]
