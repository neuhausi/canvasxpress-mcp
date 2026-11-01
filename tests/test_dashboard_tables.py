"""Several related files -> one dashboard: the largest file is the base, the others are looked up on a shared key."""

import copy
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cx_dashboard_suggest as sg  # noqa: E402


def _orders(n=60):
    rows = [["order_id", "customer_id", "product_id", "quantity", "discount"]]
    for i in range(1, n + 1):
        rows.append(["O%03d" % i, "C%d" % (1 + i % 5), "P%d" % (1 + i % 3), 1 + i % 7, [0, 0.05, 0.1][i % 3]])
    return rows


CUSTOMERS = [["customer_id", "segment", "country"], ["C1", "SMB", "USA"], ["C2", "Enterprise", "UK"],
             ["C3", "Consumer", "USA"], ["C4", "SMB", "Japan"], ["C5", "Consumer", "UK"]]
PRODUCTS = [["product_id", "category", "unit_price"], ["P1", "Hardware", 19.5], ["P2", "Software", 49.0],
            ["P3", "Services", 120.0]]


def _tables(order=(0, 1, 2)):
    t = [{"name": "orders", "data": _orders()}, {"name": "customers", "data": copy.deepcopy(CUSTOMERS)},
         {"name": "products", "data": copy.deepcopy(PRODUCTS)}]
    return [t[i] for i in order]


def test_the_largest_file_is_the_base_and_keeps_every_row_whatever_the_order():
    for order in [(0, 1, 2), (2, 1, 0), (1, 2, 0)]:
        rows, info, error = sg.join_tables(_tables(order))
        assert error is None and info["base"] == "orders" and len(rows) == 61
        assert rows[0] == ["order_id", "customer_id", "product_id", "quantity", "discount",
                           "segment", "country", "category", "unit_price"] or set(rows[0]) >= {"segment", "category"}
        by = dict(zip(rows[0], rows[1]))
        assert (by["segment"], by["category"], by["unit_price"]) == ("Enterprise", "Software", 49.0)    # O001: C2, P2


def test_unmatched_rows_are_kept_with_missing_values_and_counted():
    tables = _tables()
    tables[1]["data"] = [r for r in tables[1]["data"] if r[0] != "C3"]
    rows, info, error = sg.join_tables(tables)
    assert error is None and len(rows) == 61
    assert [j for j in info["joins"] if j["table"] == "customers"][0]["unmatched"] == len([r for r in _orders()[1:] if r[1] == "C3"])
    index = rows[0].index("segment")
    assert [r[index] for r in rows[1:] if r[1] == "C3"] == [None] * len([r for r in _orders()[1:] if r[1] == "C3"])


def test_a_column_that_already_exists_is_renamed_with_the_file_name():
    tables = _tables()
    tables[2]["data"][0][2] = "quantity"
    rows, _info, error = sg.join_tables(tables)
    assert error is None and "quantity (products)" in rows[0]


def test_errors_say_what_to_correct():
    assert "two to six files" in sg.join_tables([{"name": "a", "data": _orders()}])[2]
    unrelated = [{"name": "orders", "data": _orders()}, {"name": "notes", "data": [["topic", "text"], ["x", "y"]]}]
    assert 'Cannot relate "notes"' in sg.join_tables(unrelated)[2]
    mismatch = [{"name": "orders", "data": _orders()}, {"name": "customers", "data": [["customer_id", "segment"], ["Z9", "SMB"]]}]
    assert "No row of" in sg.join_tables(mismatch)[2]
    many = [{"name": "orders", "data": _orders()},
            {"name": "visits", "data": [["customer_id", "note"], ["C1", "a"], ["C1", "b"]]}]
    assert "Cannot relate" in sg.join_tables(many)[2]                  # the key repeats: not a lookup table
    assert "its own name" in sg.join_tables([{"name": "a", "data": _orders()}, {"name": "a", "data": CUSTOMERS}])[2]


def test_suggest_builds_one_dashboard_from_the_joined_files():
    result = sg.suggest(tables=_tables())
    assert result["suggested"] and result["valid"], result.get("errors")
    assert result["joined"]["rows"] == 60 and result["joined"]["base"] == "orders"
    html = result["spec"]["panels"]["intro"]["html"]
    assert "Built from 3 related files: orders (60 rows), customers (5 rows), products (3 rows)" in html
    assert len(result["spec"]["data"]["data"]["value"]["y"]["smps"]) == 60            # every row of the base
    fields = [f for p in result["spec"]["panels"].values() if p.get("type") == "filters" for f in p["fields"]]
    assert "segment" in fields or "category" in fields


def test_tables_cannot_be_combined_with_data():
    result = sg.suggest(data=_orders(), tables=_tables())
    assert result["suggested"] is False and "on its own" in result["error"]


def test_the_tool_offers_tables():
    import asyncio
    import server
    tool = asyncio.run(server.mcp.get_tool("suggest_canvasxpress_dashboard"))
    assert "tables" in tool.parameters["properties"] and "several related files" in tool.description


def test_the_header_is_a_line_taller_when_it_lists_the_files_and_lookup_columns_are_named():
    result = sg.suggest(tables=_tables())
    item = next(i for i in result["spec"]["layout"]["items"] if i["panel"] == "intro")
    assert item["h"] == 4 and all(i["y"] >= 4 for i in result["spec"]["layout"]["items"] if i["panel"] != "intro")
    assert result["joined"]["lookup_columns"] == ["segment", "country", "category", "unit_price"] or \
        set(result["joined"]["lookup_columns"]) == {"segment", "country", "category", "unit_price"}


def test_a_date_that_takes_a_handful_of_values_is_not_a_time_axis():
    runs = [["batch", "run_date"], ["A", "2025-03-04"], ["B", "2025-04-22"], ["C", "2025-06-10"]]
    rows = [["sample", "batch", "total_reads", "pct_mapped", "pct_duplication", "pct_q30", "lane"]]
    for i in range(1, 61):
        rows.append(["S%d" % i, "ABC"[i % 3], 4e7 + i * 1e5, 90 + i % 5, 10 + i % 7, 91 + i % 4, "L%d" % (1 + i % 3)])
    result = sg.suggest(tables=[{"name": "samples", "data": rows}, {"name": "runs", "data": runs}])
    assert result["suggested"] and "time" not in result["spec"]["panels"] and "derived" not in result
