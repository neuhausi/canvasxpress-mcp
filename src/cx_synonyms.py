"""
Intent-synonym layer for the canvasxpress-mcp server.

Turns a plain-English phrase in the prompt into a canonical config *seed*
(`graphType` + fixed params) that is applied to the generated config BEFORE the
unknown-param strip, so the user's intent is guaranteed rather than left to
retrieval luck. One editable map, `data/intent_synonyms.json`, drives it:

  - pure synonyms         : "bell curve"          -> {graphType: Density}
  - structural type pairs : "pyramid" / "tornado" -> same body, differ only in graphType
  - scatterType variants  : "manhattan"/"miami"/"clustree" -> Scatter2D + scatterType
  - fit intents           : "loess fit"/"<linear|log|...> regression" -> fit params

Adding a synonym is one line in the JSON map. Pure functions; no server state.
"""

import json
from pathlib import Path

_MAP_PATH = Path(__file__).parent.parent / "data" / "intent_synonyms.json"

# Foreign-tool fit vocabulary the model sometimes emits as a bogus `fitType`
# key (Plotly/ggplot words). `fitType` is NOT a real CanvasXpress param — the
# validator would strip it silently, dropping the user's fit (the reported bug).
# Canonicalize each value onto the real params BEFORE the strip so it survives.
_PARAM_ALIASES = {
    "fitType": {
        "loess":      {"showLoessFit": True},
        "lowess":     {"showLoessFit": True},
        "lm":         {"showRegressionFit": True, "regressionType": "linear"},
        "linear":     {"showRegressionFit": True, "regressionType": "linear"},
        "reg":        {"showRegressionFit": True, "regressionType": "linear"},
        "ols":        {"showRegressionFit": True, "regressionType": "linear"},
        "log":        {"showRegressionFit": True, "regressionType": "logarithmic"},
        "logarithmic":{"showRegressionFit": True, "regressionType": "logarithmic"},
        "exp":        {"showRegressionFit": True, "regressionType": "exponential"},
        "exponential":{"showRegressionFit": True, "regressionType": "exponential"},
        "power":      {"showRegressionFit": True, "regressionType": "power"},
        "poly":       {"showRegressionFit": True, "regressionType": "polynomial", "regressionOrder": 2},
        "polynomial": {"showRegressionFit": True, "regressionType": "polynomial", "regressionOrder": 2},
    },
}

_SYNONYMS = None  # lazy-loaded list of {"phrases": [...], "seed": {...}}


def _load():
    """Load the synonym map once; a bad/missing map degrades to no-op, never crashes the server."""
    global _SYNONYMS
    if _SYNONYMS is None:
        try:
            _SYNONYMS = json.loads(_MAP_PATH.read_text(encoding="utf-8"))
            if not isinstance(_SYNONYMS, list):
                _SYNONYMS = []
        except (OSError, ValueError):
            _SYNONYMS = []
    return _SYNONYMS


def seed_from_prompt(description):
    """Longest-matching phrase in the lowercased prompt wins.

    Returns {"graphType": str|None, "params": dict, "phrase": str} for the best
    match, or None if no phrase matches. Longest-phrase-wins keeps a specific
    phrase ("population pyramid") from being shadowed by a shorter one ("pyramid").
    """
    text = (description or "").lower()
    best_phrase = None
    best_seed = None
    for entry in _load():
        for phrase in entry.get("phrases", []):
            if phrase in text and (best_phrase is None or len(phrase) > len(best_phrase)):
                best_phrase = phrase
                best_seed = entry.get("seed", {})
    if best_phrase is None:
        return None
    return {
        "graphType": best_seed.get("graphType"),
        "params": dict(best_seed.get("params") or {}),
        "phrase": best_phrase,
    }


def canonicalize_params(config):
    """Rewrite foreign fit vocabulary (a bogus `fitType` key) onto real params, in place.

    Returns the list of keys that were canonicalized (so a caller can see a bogus
    key was recovered rather than silently dropped).
    """
    changed = []
    if not isinstance(config, dict):
        return changed
    for key, value_map in _PARAM_ALIASES.items():
        if key not in config:
            continue
        value = config.get(key)
        replacement = value_map.get(str(value).lower()) if value is not None else None
        if replacement:
            for real_key, real_value in replacement.items():
                config.setdefault(real_key, real_value)
            config.pop(key, None)
            changed.append(key)
    return changed


def apply(config, description):
    """Canonicalize bogus fit vocab, then seed the matched synonym into the config.

    `graphType` from a matched synonym OVERRIDES whatever the model chose (a named
    chart type like "manhattan plot" is an explicit request, not a heuristic); the
    seed's params are set as the explicit intent. Mutates `config` in place and
    returns the matched phrase (for observability), or None.
    """
    if not isinstance(config, dict):
        return None
    canonicalize_params(config)
    seed = seed_from_prompt(description)
    if seed is None:
        return None
    if seed.get("graphType"):
        config["graphType"] = seed["graphType"]
    for param_key, param_value in seed["params"].items():
        config[param_key] = param_value
    return seed["phrase"]
