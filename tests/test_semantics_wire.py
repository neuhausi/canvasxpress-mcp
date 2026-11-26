"""generate_canvasxpress_config: semantic-type merge (P2 wiring).

The LLM call is stubbed so these stay offline and exercise only the deterministic
semantic merge: explicit types applied, explicit LLM config respected, auto-inference
when omitted, and the extended return payload.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import server  # noqa: E402

_tool = server.generate_canvasxpress_config
_generate = getattr(_tool, "fn", _tool)


def _stub(monkeypatch, config):
    def fake_generate_config(description, headers=None, column_types=None, temperature=0.0):
        return dict(config), []
    monkeypatch.setattr(server, "generate_config", fake_generate_config)


def test_delta_colour_gets_diverging_spectrum(monkeypatch):
    _stub(monkeypatch, {"graphType": "Scatter2D", "xAxis": ["x"], "yAxis": ["y"], "colorBy": "chg"})
    out = _generate(
        "scatter coloured by change",
        headers=["x", "y", "chg"],
        data=[[1, 2, -3], [3, 4, 5]],
        semantic_types={"chg": "PercentageChange"},
    )
    cfg = out["config"]
    assert cfg["colorSpectrumZeroValue"] == 0
    assert isinstance(cfg["colorSpectrum"], list) and len(cfg["colorSpectrum"]) == 7
    assert out["semantic_types_used"] == {"chg": "PercentageChange"}


def test_explicit_llm_choice_is_respected(monkeypatch):
    # LLM already set colorSpectrum — the semantic default must NOT overwrite it.
    existing = ["#000000", "#ffffff"]
    _stub(monkeypatch, {"graphType": "Scatter2D", "xAxis": ["x"], "yAxis": ["y"],
                        "colorBy": "chg", "colorSpectrum": existing})
    out = _generate(
        "scatter coloured by change",
        headers=["x", "y", "chg"],
        data=[[1, 2, -3], [3, 4, 5]],
        semantic_types={"chg": "PercentageChange"},
    )
    assert out["config"]["colorSpectrum"] == existing          # unchanged
    assert out["config"]["colorSpectrumZeroValue"] == 0        # the unset key still filled


def test_auto_infer_when_omitted(monkeypatch):
    _stub(monkeypatch, {"graphType": "Bar", "xAxis": ["gene"], "yAxis": ["market_percent"]})
    out = _generate(
        "bar of market percent by gene",
        headers=["gene", "market_percent"],
        data=[["BRCA1", 12.5], ["TP53", 88.0]],
    )
    # inferred Percentage on the value axis → setMin/setMax bounds, and reported back.
    assert out["config"]["setMin"] == 0 and out["config"]["setMax"] == 100
    assert out["semantic_types_used"].get("market_percent") == "Percentage"


def test_unknown_semantic_type_dropped(monkeypatch):
    _stub(monkeypatch, {"graphType": "Bar", "xAxis": ["gene"], "yAxis": ["val"]})
    out = _generate(
        "bar",
        headers=["gene", "val"],
        data=[["BRCA1", 1], ["TP53", 2]],
        semantic_types={"val": "NotAType"},
    )
    assert out["semantic_types_used"] == {}            # invalid dropped
    assert "setMin" not in out["config"]
