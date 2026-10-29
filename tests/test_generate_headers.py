"""generate_canvasxpress_config: how ``headers`` and ``data`` combine.

With ``headers`` the caller has named the columns, so every row of ``data`` is a data row.
Without it, the first row of ``data`` is the header row. The LLM call is stubbed: the tests
are offline and check only which names the tool treats as the columns.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import server  # noqa: E402

_tool = server.generate_canvasxpress_config
_generate = getattr(_tool, "fn", _tool)  # FastMCP wraps the function in a tool object

HEADERS = ["species", "bill_length_mm", "bill_depth_mm"]
ROWS = [["Adelie", 39.1, 18.7], ["Gentoo", 46.1, 13.2]]
SCATTER = {"graphType": "Scatter2D", "xAxis": ["bill_length_mm"], "yAxis": ["bill_depth_mm"],
           "colorBy": "species"}


@pytest.fixture(autouse=True)
def stub_llm(monkeypatch):
    seen = {}

    def fake_generate_config(description, headers=None, column_types=None, temperature=0.0):
        seen["headers"] = headers
        return dict(SCATTER), []

    monkeypatch.setattr(server, "generate_config", fake_generate_config)
    return seen


def test_headers_are_the_header_row_and_every_data_row_is_data(stub_llm):
    out = _generate("scatter of bill length vs depth", headers=HEADERS, data=ROWS)
    assert out["headers_used"] == HEADERS
    assert stub_llm["headers"] == HEADERS            # not ["Adelie", 39.1, 18.7]
    assert out["valid"] and out["warnings"] == []    # the config's columns all exist
    assert out["data"] == ROWS                       # the first row is not consumed


def test_without_headers_the_first_data_row_is_the_header_row(stub_llm):
    out = _generate("scatter of bill length vs depth", data=[HEADERS] + ROWS)
    assert out["headers_used"] == HEADERS
    assert out["valid"]


def test_headers_alone_still_work(stub_llm):
    out = _generate("scatter of bill length vs depth", headers=HEADERS)
    assert out["headers_used"] == HEADERS and out["valid"]


def test_data_rows_of_the_wrong_width_are_warned_about(stub_llm):
    out = _generate("scatter", headers=HEADERS, data=ROWS + [["Chinstrap", 49.0]])
    assert out["headers_used"] == HEADERS
    assert any("do not have 3 values" in w for w in out["warnings"])


def test_data_that_is_not_an_array_of_arrays_is_rejected(stub_llm):
    out = _generate("scatter", headers=HEADERS, data=["Adelie", 39.1])
    assert out["valid"] is False and out["config"] == {}
    assert out["headers_used"] == []


def test_the_old_behaviour_is_gone(stub_llm):
    # Previously data won over headers: its first ROW became the column names.
    out = _generate("scatter", headers=HEADERS, data=ROWS)
    assert out["headers_used"] != [str(cell) for cell in ROWS[0]]
