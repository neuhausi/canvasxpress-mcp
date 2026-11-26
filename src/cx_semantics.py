"""
cx_semantics.py — deterministic semantic-type → CanvasXpress presentation defaults.

Flint's field-meaning idea applied to the MCP: instead of leaving palette / baseline /
scale choices to the LLM (which the hallucinated-param strip then has to clean up), tag
each column with a *semantic type* (what the field means) and derive an explicit
CanvasXpress config fragment from that type deterministically.

Two public functions:
  infer_semantic_types(headers, data, column_types) -> {column: semantic_type}
  derive_config_fragment(semantic_types, config, headers, data) -> {config_key: value}

No LLM call. No API key. Every emitted config key is a real engine parameter present in
`cx_knowledge.KNOWN_PARAMS` / the engine config (verified in the plan's P0 design), so the
downstream hallucinated-param strip never removes fragment output. Fragments are merged by
the caller **only where the LLM left a key unset** (never overwriting an explicit choice).

v1 vocabulary (closed list, 5 types): PercentageChange, Rank, Percentage, Date, Count.
`Price`/`Currency` (no engine currency-format param) and `Temperature` (needs a
domain-specific midpoint) are deliberately out of v1.

Reuses the shared name-regex layer from cx_selector (`_col_matches` / `_PATTERNS`).
"""

from __future__ import annotations

import re
from typing import Any, Callable, Optional

try:  # package / module dual-import, mirroring the rest of src/
    from . import cx_selector  # type: ignore
except Exception:  # pragma: no cover - direct-import fallback
    import cx_selector  # type: ignore


# ---------------------------------------------------------------------------
# New name-regex groups specific to semantic typing
# (cx_selector already carries rank / proportion / frequency / time groups we reuse)
# ---------------------------------------------------------------------------

_LOCAL_PATTERNS_RAW: dict[str, list[str]] = {
    "pctchange": [
        r"pct[\s_]?change", r"delta", r"change$", r"_chg$", r"chg$",
        r"log2fc", r"lfc", r"logfc", r"_diff$", r"fold[\s_]?change",
    ],
    "date_sem": [
        r"date", r"year$", r"month$", r"_ym$", r"yearmonth", r"timestamp", r"quarter",
    ],
    "percentage": [r"pct$", r"percent", r"proportion", r"^prop$", r"ratio$", r"fraction"],
    "count_sem": [r"count$", r"^n$", r"_n$", r"freq", r"quantity", r"qty", r"num_", r"reads", r"umi"],
}

_LOCAL_PATTERNS: dict[str, list[re.Pattern]] = {
    group: [re.compile(p, re.IGNORECASE) for p in pats]
    for group, pats in _LOCAL_PATTERNS_RAW.items()
}


def _local_matches(col: str, group: str) -> bool:
    """True if a column name matches any regex in a cx_semantics-local group."""
    return any(p.search(col) for p in _LOCAL_PATTERNS.get(group, []))


# ---------------------------------------------------------------------------
# Value-based column analysis (only runs when `data` is supplied)
# ---------------------------------------------------------------------------

def _column_values(col: str, headers: list[str], data: Optional[list]) -> list:
    """Extract the raw cell values for one column from a row-oriented 2-D array.

    `data` may or may not carry a header row; if its first row equals `headers`
    it is treated as a header row and skipped.
    """
    if not data or not headers or col not in headers:
        return []
    idx = headers.index(col)
    rows = data
    if rows and isinstance(rows[0], (list, tuple)) and list(rows[0]) == list(headers):
        rows = rows[1:]
    values = []
    for row in rows:
        if isinstance(row, (list, tuple)) and idx < len(row):
            values.append(row[idx])
    return values


def _numeric(values: list) -> list[float]:
    """Return the subset of values that parse as finite numbers, as floats."""
    out = []
    for v in values:
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            out.append(float(v))
            continue
        if isinstance(v, str):
            s = v.strip().replace(",", "")
            try:
                out.append(float(s))
            except ValueError:
                pass
    return out


def _all_numeric(values: list) -> bool:
    """True when every non-empty value parses as a number."""
    nonempty = [v for v in values if not (v is None or (isinstance(v, str) and v.strip() == ""))]
    if not nonempty:
        return False
    return len(_numeric(nonempty)) == len(nonempty)


def _straddles_zero(values: list) -> bool:
    nums = _numeric(values)
    return any(n < 0 for n in nums) and any(n > 0 for n in nums)


def _in_percent_range(values: list) -> bool:
    """Value-based Percentage: finite, all in [0,100], max > 1, and NOT all integers.

    Integer-only columns in [0,100] are indistinguishable from counts/ranks by value
    alone, so they fall through to Rank/Count; a Percentage name match still wins.
    """
    nums = _numeric(values)
    if not nums or not _all_numeric(values):
        return False
    if all(n == int(n) for n in nums):
        return False
    return all(0.0 <= n <= 100.0 for n in nums) and max(nums) > 1.0


def _is_rank_permutation(values: list) -> bool:
    nums = _numeric(values)
    if not nums or not _all_numeric(values) or len(nums) < 2:
        return False
    if any(n != int(n) for n in nums):
        return False
    ints = sorted(int(n) for n in nums)
    return ints == list(range(1, len(ints) + 1))


def _is_nonneg_integers(values: list) -> bool:
    nums = _numeric(values)
    if not nums or not _all_numeric(values):
        return False
    return all(n >= 0 and n == int(n) for n in nums)


# ---------------------------------------------------------------------------
# Vocabulary (channel + fragment). Fragments are callables so data-dependent
# values (the diverging ramp, the rank/count scheme) stay in one place.
# ---------------------------------------------------------------------------

# The engine default diverging ramp (config.js:1183): blue-low → red-high.
_DIVERGING_SPECTRUM = [
    "#4575b4", "#91bfdb", "#e0f3f8", "#ffffbf", "#fee090", "#fc8d59", "#d73027",
]

SEMANTIC_TYPES: dict[str, dict[str, Any]] = {
    "PercentageChange": {"aliases": ["Delta", "Change"], "channel": "color"},
    "Rank": {"aliases": [], "channel": "color"},
    "Percentage": {"aliases": ["Proportion"], "channel": "axis"},
    "Date": {"aliases": ["YearMonth"], "channel": "axis"},
    "Count": {"aliases": ["Quantity"], "channel": "any"},
}

# alias (lower-case) -> canonical type name, for validating caller-supplied values
_ALIAS_TO_TYPE: dict[str, str] = {}
for _t, _meta in SEMANTIC_TYPES.items():
    _ALIAS_TO_TYPE[_t.lower()] = _t
    for _a in _meta["aliases"]:
        _ALIAS_TO_TYPE[_a.lower()] = _t


def canonical_type(name: str) -> Optional[str]:
    """Map a type name or alias (case-insensitive) to its canonical name, or None."""
    if not isinstance(name, str):
        return None
    return _ALIAS_TO_TYPE.get(name.strip().lower())


# ---------------------------------------------------------------------------
# Inference — name-regex + value checks, with precedence (first match wins)
# ---------------------------------------------------------------------------

def infer_semantic_types(
    headers: list[str],
    data: Optional[list] = None,
    column_types: Optional[dict[str, str]] = None,
) -> dict[str, str]:
    """Tag each column with a v1 semantic type, deterministically.

    Precedence (first to match wins): Date, PercentageChange, Percentage, Rank, Count.
    Value-based checks run only when `data` is supplied; otherwise name-regex alone.
    A column that matches nothing is left untagged (absent from the result).

    @param headers      ordered column names
    @param data         optional row-oriented 2-D array (header row tolerated)
    @param column_types optional coarse types ({"date","numeric","factor","string"})
    @returns {column_name: semantic_type} for every column that matched a type
    @public
    """
    column_types = column_types or {}
    result: dict[str, str] = {}

    for col in headers or []:
        values = _column_values(col, headers, data)
        ctype = (column_types.get(col) or "").lower()

        # 1. Date — coarse date type or a date-ish name.
        if ctype == "date" or _local_matches(col, "date_sem") or cx_selector._col_matches(col, "time"):
            result[col] = "Date"
            continue

        # 2. PercentageChange — signed change: name or values straddling 0.
        if _local_matches(col, "pctchange") or (values and _straddles_zero(values)):
            result[col] = "PercentageChange"
            continue

        # 3. Percentage — 0-100 proportion: name or value range.
        if (
            _local_matches(col, "percentage")
            or cx_selector._col_matches(col, "proportion")
            or (values and _in_percent_range(values))
        ):
            result[col] = "Percentage"
            continue

        # 4. Rank — ordinal position: name or a 1..n permutation.
        if cx_selector._col_matches(col, "rank") or (values and _is_rank_permutation(values)):
            result[col] = "Rank"
            continue

        # 5. Count — non-negative magnitude: name or non-negative integers.
        if _local_matches(col, "count_sem") or (values and _is_nonneg_integers(values)):
            result[col] = "Count"
            continue

    return result


# ---------------------------------------------------------------------------
# Fragment derivation — honours each type's channel against the actual config
# ---------------------------------------------------------------------------

_AESTHETIC_KEYS = ("colorBy", "sizeBy", "shapeBy", "lineBy", "ridgeBy", "ellipseBy")


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _is_color_channel(col: str, config: dict) -> bool:
    return col in _as_list(config.get("colorBy"))


def _aesthetic_columns(config: dict) -> set:
    cols: set = set()
    for key in _AESTHETIC_KEYS:
        for c in _as_list(config.get(key)):
            cols.add(c)
    return cols


def derive_config_fragment(
    semantic_types: dict[str, str],
    config: dict,
    headers: Optional[list[str]] = None,
    data: Optional[list] = None,
) -> dict:
    """Derive the CanvasXpress config keys implied by the columns' semantic types.

    Only keys whose channel condition is satisfied by `config` are emitted (e.g. a
    diverging spectrum only when the column is actually the colour channel). Every key
    is a real engine parameter. The caller merges the result respecting explicit config
    (sets a key only when it is absent), so earlier columns win over later ones here too.

    @param semantic_types {column_name: semantic_type} (canonical or alias names)
    @param config         the (LLM-produced) config being post-processed
    @param headers        ordered column names (unused today; kept for signature parity)
    @param data           optional row-oriented 2-D array (unused today)
    @returns {config_key: value} fragment to merge
    @public
    """
    fragment: dict = {}
    if not semantic_types or not isinstance(config, dict):
        return fragment

    aesthetics = _aesthetic_columns(config)

    def put(key: str, value: Any) -> None:
        # First column to claim a global key wins (merge is additive, no clobber).
        if key not in fragment:
            fragment[key] = value

    for col, raw_type in semantic_types.items():
        stype = canonical_type(raw_type)
        if stype is None:
            continue

        if stype == "PercentageChange":
            if _is_color_channel(col, config):
                put("colorSpectrum", list(_DIVERGING_SPECTRUM))
                put("colorSpectrumZeroValue", 0)

        elif stype == "Rank":
            if _is_color_channel(col, config):
                put("colorScheme", "Viridis")

        elif stype == "Percentage":
            if col not in aesthetics:
                put("setMin", 0)
                put("setMax", 100)

        elif stype == "Date":
            if col not in aesthetics:
                put("smpTimeScale", True)
                put("smpLabelTimeAuto", True)

        elif stype == "Count":
            if _is_color_channel(col, config):
                put("colorScheme", "Viridis")
            elif col not in aesthetics:
                put("setMin", 0)

    return fragment


# ---------------------------------------------------------------------------
# Rationale — a short, human-readable reason a column got its type (for the
# annotate_semantic_types tool). Deterministic, mirrors the infer precedence.
# ---------------------------------------------------------------------------

def rationale_for(
    col: str,
    stype: str,
    headers: Optional[list[str]] = None,
    data: Optional[list] = None,
    column_types: Optional[dict[str, str]] = None,
) -> str:
    """Explain which signal tagged `col` as `stype` (name match vs value pattern)."""
    column_types = column_types or {}
    values = _column_values(col, headers or [], data)
    ctype = (column_types.get(col) or "").lower()

    if stype == "Date":
        if ctype == "date":
            return "column_types marks this column as a date"
        return "name looks temporal (date/year/month/quarter/timestamp)"
    if stype == "PercentageChange":
        if _local_matches(col, "pctchange"):
            return "name denotes a signed change (delta/pct_change/fold-change)"
        return "values straddle zero (both negative and positive)"
    if stype == "Percentage":
        if _local_matches(col, "percentage") or cx_selector._col_matches(col, "proportion"):
            return "name denotes a proportion/percentage"
        return "values are non-integer and fall within 0–100"
    if stype == "Rank":
        if cx_selector._col_matches(col, "rank"):
            return "name denotes an ordinal position (rank/order/position)"
        return "values form a 1..n permutation"
    if stype == "Count":
        if _local_matches(col, "count_sem"):
            return "name denotes a count/quantity"
        return "values are non-negative integers"
    return "matched the semantic-type signals"
