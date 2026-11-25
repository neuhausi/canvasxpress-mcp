"""
Tests for the intent-synonym layer (cx_synonyms): plain-English phrase -> canonical
config seed, applied before the unknown-param strip.

Offline: exercises the pure functions in cx_synonyms against the shipped
data/intent_synonyms.json map. No LLM / server needed.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cx_synonyms as syn  # noqa: E402
import cx_knowledge as ck  # noqa: E402


def test_map_parses_and_is_nonempty():
    entries = syn._load()
    assert isinstance(entries, list) and len(entries) >= 10
    for e in entries:
        assert isinstance(e.get("phrases"), list) and e["phrases"]
        assert isinstance(e.get("seed"), dict) and e["seed"].get("graphType")


def test_loess_fit_seeds_showloessfit_and_no_fittype():
    cfg = {"graphType": "Scatter2D", "xAxis": ["x"], "yAxis": ["y"]}
    syn.apply(cfg, "loess fit of x and y")
    assert cfg.get("showLoessFit") is True
    assert "fitType" not in cfg


def test_linear_regression_seeds_regression_params():
    cfg = {"graphType": "Scatter2D"}
    syn.apply(cfg, "linear regression of x vs y")
    assert cfg.get("showRegressionFit") is True
    assert cfg.get("regressionType") == "linear"


def test_manhattan_seeds_scatter2d_scattertype():
    cfg = {}
    syn.apply(cfg, "manhattan plot of my GWAS p-values")
    assert cfg.get("graphType") == "Scatter2D"
    assert cfg.get("scatterType") == "manhattan"


def test_bell_curve_seeds_density():
    cfg = {"graphType": "Bar"}
    syn.apply(cfg, "bell curve of mpg")
    assert cfg.get("graphType") == "Density"


def test_pyramid_and_tornado_identical_except_graphtype():
    pyr = {}
    tor = {}
    syn.apply(pyr, "population pyramid by age and sex")
    syn.apply(tor, "tornado plot of risk factors")
    assert pyr.get("graphType") == "Pyramid"
    assert tor.get("graphType") == "Tornado"
    pyr_rest = {k: v for k, v in pyr.items() if k != "graphType"}
    tor_rest = {k: v for k, v in tor.items() if k != "graphType"}
    assert pyr_rest == tor_rest  # same body, differ only in graphType


def test_bogus_fittype_key_is_canonicalized_not_dropped():
    # The reported bug: the model emits fitType (not a real param). Canonicalize
    # it to the real param BEFORE the strip rather than letting it vanish.
    cfg = {"graphType": "Scatter2D", "fitType": "loess"}
    changed = syn.canonicalize_params(cfg)
    assert "fitType" in changed
    assert cfg.get("showLoessFit") is True
    assert "fitType" not in cfg


def test_genuinely_unknown_param_is_stripped_so_handler_can_warn():
    # cx_synonyms does not recover a truly unknown key; it stays for the strip,
    # which is what lets generate_config surface a 'removed'/warning entry.
    cfg = {"graphType": "Scatter2D", "totallyMadeUpParam": 1}
    syn.canonicalize_params(cfg)
    _filtered, removed = ck.filter_unknown_params(cfg)
    assert "totallyMadeUpParam" in removed


def test_longest_phrase_wins():
    # "population pyramid" must not be shadowed by a shorter "pyramid" phrase.
    seed = syn.seed_from_prompt("show me a population pyramid")
    assert seed and seed["graphType"] == "Pyramid"
    assert seed["phrase"] == "population pyramid"


def test_no_match_returns_none():
    assert syn.seed_from_prompt("a plain bar chart of sales by region") is None
    cfg = {"graphType": "Bar"}
    assert syn.apply(cfg, "a plain bar chart") is None
    assert cfg == {"graphType": "Bar"}  # untouched
