"""annotate_semantic_types MCP tool (P3): suggested types + rationale + preview."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import server  # noqa: E402
import cx_semantics  # noqa: E402

_tool = server.annotate_semantic_types
_annotate = getattr(_tool, "fn", _tool)


def test_tags_names():
    out = _annotate(["gene", "pct_change", "rank"])
    assert out["semantic_types"]["pct_change"] == "PercentageChange"
    assert out["semantic_types"]["rank"] == "Rank"
    assert "gene" not in out["semantic_types"]


def test_rationale_and_vocabulary():
    out = _annotate(["pct_change", "rank"])
    assert out["rationale"]["pct_change"]
    assert out["rationale"]["rank"]
    assert out["vocabulary"] == sorted(cx_semantics.SEMANTIC_TYPES)
    assert out["vocabulary"]  # non-empty


def test_fragment_preview_shows_colour_and_axis_types():
    out = _annotate(
        ["pct_change", "share"],
        data=[["pct_change", "share"], [-2, 12.5], [3, 88.0]],
    )
    prev = out["fragment_preview"]
    # Delta (colour-channel probe) contributes the diverging spectrum + zero value.
    assert prev["colorSpectrumZeroValue"] == 0
    assert isinstance(prev["colorSpectrum"], list)
    # Percentage (axis probe) contributes the 0–100 bounds.
    assert prev["setMin"] == 0 and prev["setMax"] == 100
