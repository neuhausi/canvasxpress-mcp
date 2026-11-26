"""
Tests for cx_semantics: deterministic semantic-type inference and the
CanvasXpress config fragment each type derives.

Offline: pure functions, no LLM / server.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cx_semantics as sem  # noqa: E402


# --------------------------------------------------------------------------- #
# infer_semantic_types — name-based                                           #
# --------------------------------------------------------------------------- #

def test_infer_percentage_change_by_name():
    out = sem.infer_semantic_types(["gene", "pct_change"])
    assert out["pct_change"] == "PercentageChange"
    out2 = sem.infer_semantic_types(["delta"])
    assert out2["delta"] == "PercentageChange"


def test_infer_rank_by_name():
    out = sem.infer_semantic_types(["rank", "label"])
    assert out["rank"] == "Rank"


def test_infer_percentage_by_name():
    out = sem.infer_semantic_types(["market_percent"])
    assert out["market_percent"] == "Percentage"


def test_infer_date_by_name_and_coarse_type():
    out = sem.infer_semantic_types(["order_date", "value"])
    assert out["order_date"] == "Date"
    out2 = sem.infer_semantic_types(["t0"], column_types={"t0": "date"})
    assert out2["t0"] == "Date"


def test_infer_count_by_name():
    out = sem.infer_semantic_types(["read_count"])
    assert out["read_count"] == "Count"


def test_price_is_not_a_v1_type():
    # Price / Currency were dropped in P0 (no expressible fragment).
    out = sem.infer_semantic_types(["revenue_usd", "price"])
    assert "revenue_usd" not in out or out["revenue_usd"] != "Price"
    assert "Price" not in sem.SEMANTIC_TYPES


def test_untagged_column_absent():
    out = sem.infer_semantic_types(["gene"])
    assert "gene" not in out


# --------------------------------------------------------------------------- #
# infer_semantic_types — value-based                                          #
# --------------------------------------------------------------------------- #

def test_infer_straddling_zero_is_delta():
    headers = ["score"]
    data = [["score"], [-3], [2], [-1], [5]]
    out = sem.infer_semantic_types(headers, data=data)
    assert out["score"] == "PercentageChange"


def test_infer_percent_range_is_percentage():
    headers = ["share"]
    data = [[12.5], [88.3], [40.0], [55.1]]  # true percentages (non-integer), no header row
    out = sem.infer_semantic_types(headers, data=data)
    assert out["share"] == "Percentage"


def test_infer_rank_permutation():
    headers = ["pos"]
    data = [[3], [1], [4], [2]]
    out = sem.infer_semantic_types(headers, data=data)
    assert out["pos"] == "Rank"


def test_infer_nonneg_integers_is_count():
    headers = ["widgets"]
    data = [[0], [5], [12], [3]]
    out = sem.infer_semantic_types(headers, data=data)
    assert out["widgets"] == "Count"


def test_probability_0_1_not_tagged_percentage():
    headers = ["prob"]
    data = [[0.2], [0.8], [0.5]]
    out = sem.infer_semantic_types(headers, data=data)
    # all <= 1, so NOT a 0-100 percentage; non-integers so NOT count/rank either.
    assert out.get("prob") != "Percentage"


# --------------------------------------------------------------------------- #
# derive_config_fragment — channel-aware                                      #
# --------------------------------------------------------------------------- #

def test_delta_colour_gets_diverging_spectrum():
    config = {"graphType": "Scatter2D", "colorBy": "chg"}
    frag = sem.derive_config_fragment({"chg": "PercentageChange"}, config)
    assert frag["colorSpectrumZeroValue"] == 0
    assert isinstance(frag["colorSpectrum"], list) and len(frag["colorSpectrum"]) == 7


def test_delta_not_colour_channel_emits_nothing():
    config = {"graphType": "Scatter2D", "colorBy": "group"}
    frag = sem.derive_config_fragment({"chg": "PercentageChange"}, config)
    assert frag == {}


def test_percentage_value_axis_bounds():
    config = {"graphType": "Bar"}
    frag = sem.derive_config_fragment({"share": "Percentage"}, config)
    assert frag["setMin"] == 0 and frag["setMax"] == 100


def test_percentage_as_aesthetic_emits_nothing():
    config = {"graphType": "Scatter2D", "sizeBy": "share"}
    frag = sem.derive_config_fragment({"share": "Percentage"}, config)
    assert frag == {}


def test_rank_colour_scheme():
    config = {"graphType": "Bar", "colorBy": "rank"}
    frag = sem.derive_config_fragment({"rank": "Rank"}, config)
    assert frag["colorScheme"] == "Viridis"


def test_date_time_scale():
    config = {"graphType": "Line"}
    frag = sem.derive_config_fragment({"order_date": "Date"}, config)
    assert frag["smpTimeScale"] is True and frag["smpLabelTimeAuto"] is True


def test_count_colour_vs_axis():
    frag_color = sem.derive_config_fragment({"n": "Count"}, {"colorBy": "n"})
    assert frag_color["colorScheme"] == "Viridis"
    frag_axis = sem.derive_config_fragment({"n": "Count"}, {"graphType": "Bar"})
    assert frag_axis["setMin"] == 0 and "setMax" not in frag_axis


def test_alias_accepted():
    frag = sem.derive_config_fragment({"chg": "Delta"}, {"colorBy": "chg"})
    assert frag["colorSpectrumZeroValue"] == 0
