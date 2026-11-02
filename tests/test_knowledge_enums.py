"""
Tests for how cx_knowledge turns config-schema enums into validation lists.

Offline: runs against a small inline schema, plus the bundled config schema that
ships in data/schema/.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cx_knowledge as ck  # noqa: E402

_BUNDLED = os.path.join(os.path.dirname(__file__), "..", "data", "schema",
                        "canvasxpress-config-latest.schema.json")


def _entries(props):
    return ck._config_schema_to_entries({"properties": props})


def test_false_disable_sentinel_keeps_enum_closed():
    e = _entries({"colorScheme": {
        "type": ["string", "boolean"], "enum": ["Dark2", "Paired", False],
        "x-cx-false-disables": True, "x-cx-nl": "Use the color scheme {_option_}"}})
    assert e["colorScheme"]["valid_values"] == ["Dark2", "Paired"]


def test_data_reference_template_stays_open():
    e = _entries({"colorBy": {
        "type": ["string", "boolean"], "enum": [False, "variable"],
        "x-cx-nl": "Color by {_factor_}"}})
    assert e["colorBy"]["valid_values"] == []
    assert e["colorBy"]["suggested_values"] == ["variable"]


def test_other_sentinels_stay_open():
    e = _entries({"x": {"type": ["string", "number"], "enum": ["a", 0]}})
    assert e["x"]["valid_values"] == []


def test_overlay_does_not_refill_open_entry():
    base = _entries({"colorBy": {"enum": [False, "variable"], "x-cx-nl": "Color by {_factor_}"}})
    ck._overlay_graph_knowledge(base, {"colorBy": {"valid_values": ["stale"]}})
    assert base["colorBy"]["valid_values"] == []


def test_bundled_schema_accepts_real_color_schemes_and_false():
    with open(_BUNDLED) as f:
        entries = ck._config_schema_to_entries(json.load(f))
    schemes = entries["colorScheme"]["valid_values"]
    for name in ("Dark2", "Paired", "WallStreetJournal3", "Tableau"):
        assert name in schemes
    assert False not in schemes
    assert entries["colorBy"]["valid_values"] == []


def test_validate_param_values_real_configs():
    # Pin to the bundled config schema that ships in data/schema/ so this test
    # is deterministic and offline (see module docstring). Without this,
    # load_schema() fetches the live canvasxpress.org schema, whose colorScheme
    # enum can lag the engine release and drop the closed value list — leaving
    # the overlay to backfill from the smaller _BUNDLED_SCHEMA list (no Dark2)
    # and failing the assertion for reasons unrelated to this repo.
    prev_skip = ck.SKIP_FETCH
    prev_cached = ck._cached_file
    ck.SKIP_FETCH = True
    # Ignore any on-disk cache so the offline layer reads the committed bundled
    # schema (data/schema/) rather than a possibly-stale cached live copy.
    ck._cached_file = lambda *_a, **_k: None
    try:
        ck.load_schema(force=True)
        ok = ck.validate_param_values({"colorScheme": "Dark2", "theme": "cxdark",
                                       "colorBy": "Treatment", "legendPosition": False})
        assert ok["invalid_values"] == {}
        bad = ck.validate_param_values({"legendPosition": "nowhere"})
        assert "legendPosition" in bad["invalid_values"]
    finally:
        ck.SKIP_FETCH = prev_skip
        ck._cached_file = prev_cached
        ck.load_schema(force=True)
