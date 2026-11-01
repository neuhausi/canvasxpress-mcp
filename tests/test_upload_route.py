"""The upload path: a file posted to /mcp/upload is held and suggest_canvasxpress_dashboard(file_id=...) reads it
on the server, so the model never types the rows. ``csv`` (raw text) is the fallback without a shell."""

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


def _text(delim=","):
    out = io.StringIO()
    csv.writer(out, delimiter=delim).writerows(ROWS)
    return out.getvalue()


@pytest.fixture
def client():
    from starlette.testclient import TestClient
    return TestClient(server.mcp.http_app())


def test_upload_then_suggest_by_file_id_gives_the_same_dashboard_as_the_rows(client):
    r = client.post("/mcp/upload?name=run7.csv", content=_text().encode())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["rows"] == 120 and body["columns"] == 7 and body["name"] == "run7.csv"
    via_id = server.suggest_canvasxpress_dashboard(file_id=body["file_id"])
    via_rows = server.suggest_canvasxpress_dashboard(data=ROWS, title="run7")
    assert via_id["suggested"] and via_id["template"] == "ngs-qc"
    full_id = "".join(server.get_canvasxpress_dashboard_result(via_id["result_id"], i)["text"] for i in range(via_id["parts"]))
    full_rows = "".join(server.get_canvasxpress_dashboard_result(via_rows["result_id"], i)["text"] for i in range(via_rows["parts"]))
    import json
    assert json.loads(full_id)["spec"] == json.loads(full_rows)["spec"]
    assert len(json.dumps(via_id)) < 3000                                   # the tool call and result stay tiny


def test_tsv_and_a_multipart_upload_work(client):
    r = client.post("/mcp/upload", files={"file": ("qc.tsv", _text("\t").encode(), "text/tab-separated-values")})
    assert r.status_code == 200 and r.json()["rows"] == 120 and r.json()["name"] == "qc.tsv"


def test_bad_uploads_and_unknown_ids_say_so(client):
    assert client.post("/mcp/upload", content=b"onlyheader\n").status_code == 400
    assert "No uploaded file" in server.suggest_canvasxpress_dashboard(file_id="nope")["error"]


def test_raw_csv_text_is_the_fallback():
    out = server.suggest_canvasxpress_dashboard(csv=_text())
    assert out["suggested"] and out["template"] == "ngs-qc"


def test_cells_are_read_like_a_careful_model_would_send_them():
    assert [server._cell(v) for v in ["12", "1.5", "NA", "", "x", " 7 "]] == [12, 1.5, None, None, "x", 7]


def test_the_tool_text_puts_the_upload_first():
    import asyncio
    tool = asyncio.run(server.mcp.get_tool("suggest_canvasxpress_dashboard"))
    assert "/mcp/upload" in tool.description and tool.description.index("file_id") < tool.description.index("csv:")
    assert {"file_id", "csv"} <= set(tool.parameters["properties"])
