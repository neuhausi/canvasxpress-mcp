"""The url path: suggest_canvasxpress_dashboard(url=...) makes the SERVER fetch the file, so no client has to
move the rows (the model types one link). A file dropped at GET /mcp/upload becomes a link under /mcp/file/."""

import csv
import io
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import server  # noqa: E402

ROWS = [["sample", "lane", "batch", "total_reads", "pct_mapped", "pct_duplication", "pct_q30"]] + [
    ["S%03d" % i, "L%d" % (1 + i % 4), "Run%s" % "ABC"[i % 3], 30000000 + i * 1000, 90 + i % 8 / 2, 10 + i % 9, 88 + i % 6]
    for i in range(1, 121)]


def _text():
    out = io.StringIO()
    csv.writer(out).writerows(ROWS)
    return out.getvalue()


def _spec_title(out):
    import json
    full = "".join(server.get_canvasxpress_dashboard_result(out["result_id"], i)["text"] for i in range(out["parts"]))
    return json.loads(full)["spec"]["title"]


@pytest.fixture
def client():
    from starlette.testclient import TestClient
    return TestClient(server.mcp.http_app())


def test_a_dropped_file_is_a_link_that_builds_the_dashboard(client):
    assert "Drop a CSV" in client.get("/mcp/upload").text                        # the page a person uses
    body = client.post("/mcp/upload?name=run7.csv", content=_text().encode()).json()
    assert body["url"] == server._FILE_URL_BASE + body["file_id"]
    fetched = client.get("/mcp/file/" + body["file_id"])
    assert fetched.status_code == 200 and fetched.text == _text() and "run7.csv" in fetched.headers["content-disposition"]
    out = server.suggest_canvasxpress_dashboard(url=body["url"])
    assert out["suggested"] and out["template"] == "ngs-qc" and _spec_title(out) == "run7"
    assert client.get("/mcp/file/nope").status_code == 404
    assert "expired" in server.suggest_canvasxpress_dashboard(url=server._FILE_URL_BASE + "nope")["error"]


def test_a_public_link_is_fetched_by_the_server(monkeypatch):
    seen = []

    def fake_open(url):
        seen.append(url)
        return "qc-from-header.csv", _text().encode()

    monkeypatch.setattr(server, "_open_url", fake_open)
    monkeypatch.setattr(server, "_public_host", lambda host: True)
    out = server.suggest_canvasxpress_dashboard(url="https://github.com/me/repo/blob/main/data/qc.csv")
    assert seen == ["https://raw.githubusercontent.com/me/repo/main/data/qc.csv"]
    assert out["suggested"] and out["template"] == "ngs-qc" and _spec_title(out) == "qc from header"


@pytest.mark.parametrize("given, direct", [
    ("https://www.dropbox.com/scl/fi/abc/qc.csv?rlkey=k&dl=0", "https://www.dropbox.com/scl/fi/abc/qc.csv?rlkey=k&dl=1"),
    ("https://www.dropbox.com/s/abc/qc.csv", "https://www.dropbox.com/s/abc/qc.csv?dl=1"),
    ("https://drive.google.com/file/d/1AbC/view?usp=sharing", "https://drive.google.com/uc?export=download&id=1AbC"),
    ("https://docs.google.com/spreadsheets/d/1AbC/edit#gid=42", "https://docs.google.com/spreadsheets/d/1AbC/export?format=csv&gid=42"),
    ("https://example.org/data/qc.csv", "https://example.org/data/qc.csv"),
])
def test_share_links_become_direct_download_links(given, direct):
    assert server._share_url(given) == direct


def test_private_hosts_and_bad_links_are_refused():
    assert "public host" in server.suggest_canvasxpress_dashboard(url="http://127.0.0.1:8100/health")["error"]
    assert "public host" in server.suggest_canvasxpress_dashboard(url="http://localhost/x.csv")["error"]
    assert "http(s)" in server.suggest_canvasxpress_dashboard(url="ftp://example.org/x.csv")["error"]
    assert "http(s)" in server.suggest_canvasxpress_dashboard(url="not a link")["error"]


def test_the_tool_tells_the_model_the_link_comes_first():
    import asyncio
    text = asyncio.run(server.mcp.get_tool("suggest_canvasxpress_dashboard")).description
    assert text.index("pass url") < text.index("file_id") < text.index("pass csv")
    assert "https://canvasxpress.org/mcp/upload" in text and "without asking the user which way" in text
