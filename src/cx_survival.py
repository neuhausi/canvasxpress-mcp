#!/usr/bin/env python3
"""
cx_survival.py
==============
Kaplan-Meier survival analysis skill for the CanvasXpress MCP server.

Exposes one MCP tool — generate_km_config — that handles all four
capabilities in a single call, based on what inputs are provided:

  1. GENERATE   Plain English description → full KM config
  2. VALIDATE   Existing config → structured report of errors and fixes
  3. DETECT     Dataset (headers or data array) → identifies time, event,
                and grouping columns using heuristic + LLM fallback
  4. ANNOTATE   Computes median survival and log-rank p-value from data
                (pure Python, no scipy required) and embeds decorations
                into the config

Input routing:
  - description only           → generate from scratch
  - config only                → validate existing config
  - headers / data only        → detect columns, then generate
  - description + headers/data → generate with column guidance
  - config + data              → validate + recompute annotations
  - any combination            → all applicable capabilities run in sequence

KM-specific CanvasXpress parameters handled:
  graphType     : always "KaplanMeier"
  xAxis         : [time_column]            — numeric survival/follow-up time
  yAxis         : [event_column]           — 0/1 event indicator (1 = event occurred)
  groupingFactors: [group_column]          — categorical grouping variable
  xAxisTitle    : human-readable time label
  yAxisTitle    : "Survival Probability"
  colorScheme   : user-specified or "Tableau" default
  showLegend    : true
  decorations   : median survival lines + optional p-value annotation
"""

import logging
import json
import re
from typing import Optional

log = logging.getLogger("cx-mcp.survival")

# ---------------------------------------------------------------------------
# KM-specific knowledge
# ---------------------------------------------------------------------------

KM_SYSTEM_PROMPT = """You are a CanvasXpress survival analysis expert specialising in
Kaplan-Meier plots.

## OUTPUT FORMAT
Return ONLY a valid JSON object. No markdown, no backticks, no explanations.

## KAPLAN-MEIER RULES (CRITICAL)
- graphType MUST be exactly "KaplanMeier"
- xAxis     MUST be a list containing the time column (numeric, e.g. days/months/years)
- yAxis     MUST be a list containing the event/status column (0=censored, 1=event)
- groupingFactors MUST be a list of grouping columns (treatment arm, stage, etc.)
- xAxisTitle should describe the time units (e.g. "Time (months)", "Days since diagnosis")
- yAxisTitle should be "Survival Probability" or "Survival Function"
- showLegend should be true when grouping is present
- colorScheme default: "Tableau"
- DO NOT include yAxis-related axis range params unless explicitly requested
- decorations: use "line" type with "value" (not x/y) for horizontal median lines

## REQUIRED MINIMAL PARAMETERS
graphType, xAxis, yAxis, groupingFactors (when groups present)

## VALID COLOR SCHEMES
YlGn, YlGnBu, GnBu, BuGn, PuBuGn, PuBu, BuPu, RdPu, PuRd, OrRd, YlOrRd, YlOrBr,
Purples, Blues, Greens, Oranges, Reds, Greys, PuOr, BrBG, PRGn, PiYG, RdBu, RdGy,
RdYlBu, Spectral, RdYlGn, Bootstrap, Economist, Excel, GGPlot, Solarized, PaulTol,
ColorBlind, Tableau, WallStreetJournal, Stata, BlackAndWhite, CanvasXpress

## VALID THEMES
bw, classic, cx, dark, economist, excel, ggblanket, ggplot, gray, grey,
highcharts, igray, light, linedraw, minimal, none, ptol, solarized, stata, tableau, void0, wsj
"""

# Heuristic patterns for column role detection
_TIME_PATTERNS = [
    r'\btime\b', r'\bsurvival[_\s]?time\b', r'\bdays?\b', r'\bmonths?\b',
    r'\byears?\b', r'\bweeks?\b', r'\bfollow[_\s]?up\b', r'\bfu[_\s]?time\b',
    r'\bos[_\s]?time\b', r'\bpfs[_\s]?time\b', r'\bdfs[_\s]?time\b',
    r'\bduration\b', r'\btime_to\b', r'\btte\b',
]
_EVENT_PATTERNS = [
    r'\bevent\b', r'\bstatus\b', r'\bdead\b', r'\bdeath\b', r'\bcensored?\b',
    r'\boccurred?\b', r'\bindic(ator)?\b', r'\bos[_\s]?status\b',
    r'\bpfs[_\s]?status\b', r'\bdfs[_\s]?status\b', r'\bcens\b',
    r'\bfailure\b', r'\boutcome\b',
]
_GROUP_PATTERNS = [
    r'\bgroup\b', r'\barm\b', r'\btreat(ment)?\b', r'\bcohort\b',
    r'\bstage\b', r'\bgrade\b', r'\bsubgroup\b', r'\bcategory\b',
    r'\bstrat(um|a|ify)?\b', r'\bcondition\b', r'\btherapy\b',
]

# Decoration colours for median lines (up to 6 groups)
_GROUP_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

# ---------------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------------

def _score_col(col: str, patterns: list[str]) -> int:
    col_lower = col.lower()
    return sum(1 for p in patterns if re.search(p, col_lower))


def detect_km_columns(headers: list[str]) -> dict:
    """
    Heuristically identify time, event, and grouping columns from a list
    of column names.

    Returns:
        {
          "time_col":    str | None,
          "event_col":   str | None,
          "group_cols":  list[str],
          "unassigned":  list[str],
          "confidence":  "high" | "medium" | "low",
          "notes":       list[str],
        }
    """
    notes: list[str] = []
    candidates: dict[str, dict] = {}

    for col in headers:
        candidates[col] = {
            "time_score":  _score_col(col, _TIME_PATTERNS),
            "event_score": _score_col(col, _EVENT_PATTERNS),
            "group_score": _score_col(col, _GROUP_PATTERNS),
        }

    def _best(role_key: str, exclude: set[str]) -> Optional[str]:
        scored = [
            (col, candidates[col][role_key])
            for col in headers
            if col not in exclude and candidates[col][role_key] > 0
        ]
        if not scored:
            return None
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]

    used: set[str] = set()

    time_col  = _best("time_score",  used)
    if time_col:  used.add(time_col)

    event_col = _best("event_score", used)
    if event_col: used.add(event_col)

    # Group columns: all cols with any group score not already used
    group_cols = [
        col for col in headers
        if col not in used and candidates[col]["group_score"] > 0
    ]
    used.update(group_cols)

    unassigned = [col for col in headers if col not in used]

    # Confidence scoring
    found = sum([time_col is not None, event_col is not None, bool(group_cols)])
    confidence = "high" if found == 3 else ("medium" if found == 2 else "low")

    if not time_col:
        notes.append("Could not detect a time column. Look for a numeric column "
                     "representing follow-up duration (days, months, years).")
    if not event_col:
        notes.append("Could not detect an event/status column. Look for a 0/1 "
                     "indicator column (1=event occurred, 0=censored).")
    if not group_cols:
        notes.append("No grouping column detected. A KM plot without groups "
                     "shows a single overall survival curve.")
    if unassigned:
        notes.append(f"Unassigned columns (may be IDs or covariates): {unassigned}")

    return {
        "time_col":   time_col,
        "event_col":  event_col,
        "group_cols": group_cols,
        "unassigned": unassigned,
        "confidence": confidence,
        "notes":      notes,
    }


# ---------------------------------------------------------------------------
# Survival statistics (pure Python — no scipy dependency)
# ---------------------------------------------------------------------------

def _km_estimator(times: list[float], events: list[int]) -> list[tuple[float, float]]:
    """
    Kaplan-Meier estimator.
    Returns list of (time, survival_probability) step points.
    """
    from collections import defaultdict
    at_risk   = len(times)
    n_events: dict[float, int]  = defaultdict(int)
    n_censor: dict[float, int]  = defaultdict(int)

    for t, e in zip(times, events):
        if e == 1:
            n_events[t] += 1
        else:
            n_censor[t] += 1

    event_times = sorted(n_events.keys())
    surv  = 1.0
    steps = [(0.0, 1.0)]

    for t in event_times:
        d = n_events[t]
        surv *= (at_risk - d) / at_risk
        steps.append((t, surv))
        at_risk -= d + n_censor.get(t, 0)

    return steps


def _median_survival(steps: list[tuple[float, float]]) -> Optional[float]:
    """Time at which survival probability first drops to or below 0.5."""
    for t, s in steps:
        if s <= 0.5:
            return t
    return None  # median not reached


def _logrank_pvalue(
    times_a: list[float], events_a: list[int],
    times_b: list[float], events_b: list[int],
) -> float:
    """
    Two-group log-rank test (pure Python).
    Returns an approximate p-value using the chi-squared distribution
    approximation (1 degree of freedom).
    """
    import math

    all_times = sorted(set(t for t, e in zip(times_a, events_a) if e == 1) |
                       set(t for t, e in zip(times_b, events_b) if e == 1))

    O1, E1 = 0.0, 0.0

    def _at_risk_and_events(times, events, t):
        n = sum(1 for ti in times if ti >= t)
        d = sum(1 for ti, ei in zip(times, events) if ti == t and ei == 1)
        return n, d

    for t in all_times:
        n1, d1 = _at_risk_and_events(times_a, events_a, t)
        n2, d2 = _at_risk_and_events(times_b, events_b, t)
        n  = n1 + n2
        d  = d1 + d2
        if n < 2:
            continue
        e1 = n1 * d / n
        E1 += e1
        O1 += d1

    if E1 == 0:
        return 1.0

    # Variance estimate (simplified)
    V = E1 * (1 - E1 / max(O1 + (d - O1), 1))
    if V <= 0:
        return 1.0

    chi2 = (O1 - E1) ** 2 / V

    # Approximate p-value from chi-squared(df=1) using complementary error function
    # p ≈ erfc(sqrt(chi2/2))
    x = math.sqrt(chi2 / 2)
    # erfc approximation (Horner's method, accurate to ~1e-7)
    t_val = 1.0 / (1.0 + 0.3275911 * x)
    poly  = t_val * (0.254829592 + t_val * (-0.284496736 + t_val * (
            1.421413741 + t_val * (-1.453152027 + t_val * 1.061405429))))
    p = poly * math.exp(-(x * x))
    return float(max(0.0, min(1.0, p)))


def compute_km_stats(
    data: list[list],
    time_col: str,
    event_col: str,
    group_col: Optional[str] = None,
) -> dict:
    """
    Compute KM statistics from a data array.

    Returns:
        {
          "groups": {
            group_name: {
              "n":               int,
              "n_events":        int,
              "median_survival": float | None,
              "steps":           [(time, survival), ...]
            }
          },
          "logrank_pvalue": float | None,   # only if exactly 2 groups
          "pvalue_str":     str,            # formatted: "p = 0.032" or "p < 0.001"
          "warnings":       list[str],
        }
    """
    if not data or len(data) < 2:
        return {"groups": {}, "logrank_pvalue": None, "pvalue_str": "", "warnings": ["Empty dataset"]}

    headers  = [str(h) for h in data[0]]
    warnings: list[str] = []

    try:
        time_idx  = headers.index(time_col)
    except ValueError:
        return {"groups": {}, "logrank_pvalue": None, "pvalue_str": "",
                "warnings": [f"Time column '{time_col}' not found in data"]}

    try:
        event_idx = headers.index(event_col)
    except ValueError:
        return {"groups": {}, "logrank_pvalue": None, "pvalue_str": "",
                "warnings": [f"Event column '{event_col}' not found in data"]}

    group_idx = None
    if group_col:
        try:
            group_idx = headers.index(group_col)
        except ValueError:
            warnings.append(f"Group column '{group_col}' not found — treating as single group")

    # Parse rows
    rows_by_group: dict[str, list[tuple[float, int]]] = {}
    skip = 0
    for row in data[1:]:
        try:
            t = float(row[time_idx])
            e = int(float(row[event_idx]))
            g = str(row[group_idx]) if group_idx is not None else "All"
        except (ValueError, IndexError, TypeError):
            skip += 1
            continue
        rows_by_group.setdefault(g, []).append((t, e))

    if skip:
        warnings.append(f"Skipped {skip} rows with unparseable values")

    if not rows_by_group:
        return {"groups": {}, "logrank_pvalue": None, "pvalue_str": "",
                "warnings": warnings + ["No valid rows found"]}

    # KM per group
    groups: dict[str, dict] = {}
    for group, pairs in sorted(rows_by_group.items()):
        times  = [p[0] for p in pairs]
        events = [p[1] for p in pairs]
        steps  = _km_estimator(times, events)
        median = _median_survival(steps)
        groups[group] = {
            "n":               len(pairs),
            "n_events":        sum(events),
            "median_survival": median,
            "steps":           steps,
        }

    # Log-rank p-value (two groups only)
    pvalue: Optional[float] = None
    pvalue_str = ""
    group_names = list(groups.keys())
    if len(group_names) == 2:
        g1, g2 = group_names
        pairs1  = rows_by_group[g1]
        pairs2  = rows_by_group[g2]
        pvalue  = _logrank_pvalue(
            [p[0] for p in pairs1], [p[1] for p in pairs1],
            [p[0] for p in pairs2], [p[1] for p in pairs2],
        )
        pvalue_str = (
            f"p < 0.001" if pvalue < 0.001
            else f"p = {pvalue:.3f}"
        )
    elif len(group_names) > 2:
        warnings.append(
            f"Log-rank p-value computed for two-group comparisons only "
            f"({len(group_names)} groups found)."
        )

    return {
        "groups":           groups,
        "logrank_pvalue":   pvalue,
        "pvalue_str":       pvalue_str,
        "warnings":         warnings,
    }


# ---------------------------------------------------------------------------
# Decoration builder
# ---------------------------------------------------------------------------

def build_km_decorations(
    stats:          dict,
    add_median:     bool = True,
    add_pvalue:     bool = True,
    time_col_label: str  = "Time",
) -> list[dict]:
    """
    Build CanvasXpress decoration objects from KM statistics.

    Median survival → horizontal dashed lines at y=0.5 per group
    Log-rank p-value → text annotation in the upper-right area
    """
    decorations: list[dict] = []
    groups = stats.get("groups", {})

    if add_median:
        for i, (group_name, gstats) in enumerate(groups.items()):
            med = gstats.get("median_survival")
            if med is None:
                continue  # median not reached — skip line
            color = _GROUP_COLORS[i % len(_GROUP_COLORS)]
            # Vertical line at median time
            decorations.append({
                "type":  "line",
                "value": med,
                "color": color,
                "width": 1,
                "label": f"Median {group_name}: {med:.1f} {time_col_label}",
            })

    if add_pvalue and stats.get("pvalue_str"):
        decorations.append({
            "type":  "text",
            "value": 0,       # x position (ignored for text decorations in KM)
            "color": "#333333",
            "label": f"Log-rank {stats['pvalue_str']}",
        })

    return decorations


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

_KM_REQUIRED = ["graphType", "xAxis", "yAxis"]
_KM_RECOMMENDED = ["groupingFactors", "xAxisTitle", "yAxisTitle", "colorScheme", "showLegend"]


def validate_km_config(config: dict, headers: Optional[list[str]] = None) -> dict:
    """
    Validate a KM config for correctness.

    Checks:
      - graphType is "KaplanMeier"
      - xAxis and yAxis are present and non-empty lists
      - column references exist in headers (if provided)
      - no forbidden single-dimensional-only params (e.g. smpTitle instead of yAxisTitle)
      - recommends missing but useful params

    Returns:
        {
          "valid":        bool,
          "errors":       list[str],   # must-fix issues
          "warnings":     list[str],   # should-fix issues
          "suggestions":  list[str],   # nice-to-have improvements
          "fixed_config": dict,        # auto-corrected config (best-effort)
        }
    """
    errors:      list[str] = []
    warnings:    list[str] = []
    suggestions: list[str] = []
    fixed = dict(config)  # copy to auto-fix

    # graphType
    gt = config.get("graphType")
    if gt != "KaplanMeier":
        if gt and gt.lower() in ("kaplanmeier", "kaplan_meier", "km", "survival"):
            errors.append(f"graphType must be exactly 'KaplanMeier' (found '{gt}')")
            fixed["graphType"] = "KaplanMeier"
        else:
            errors.append(f"graphType must be 'KaplanMeier' (found {repr(gt)})")
            fixed["graphType"] = "KaplanMeier"

    # xAxis
    xaxis = config.get("xAxis")
    if not xaxis:
        errors.append("xAxis is required — must list the time/duration column e.g. ['Time']")
    elif not isinstance(xaxis, list):
        errors.append(f"xAxis must be a list, got {type(xaxis).__name__}")
        fixed["xAxis"] = [xaxis]
    elif len(xaxis) != 1:
        warnings.append(f"xAxis should contain exactly one time column (found {len(xaxis)})")

    # yAxis
    yaxis = config.get("yAxis")
    if not yaxis:
        errors.append("yAxis is required — must list the event/status column e.g. ['Event']")
    elif not isinstance(yaxis, list):
        errors.append(f"yAxis must be a list, got {type(yaxis).__name__}")
        fixed["yAxis"] = [yaxis]
    elif len(yaxis) != 1:
        warnings.append(f"yAxis should contain exactly one event column (found {len(yaxis)})")

    # Forbidden single-dim param
    if "smpTitle" in config:
        warnings.append(
            "smpTitle is for single-dimensional charts. Use yAxisTitle for KM plots."
        )
        if "yAxisTitle" not in fixed:
            fixed["yAxisTitle"] = fixed.pop("smpTitle", "Survival Probability")

    # Column references against headers
    if headers:
        header_set = set(headers)
        for key in ["xAxis", "yAxis", "groupingFactors"]:
            val = config.get(key)
            if not val:
                continue
            cols = [val] if isinstance(val, str) else val
            missing = [c for c in cols if isinstance(c, str) and c not in header_set]
            if missing:
                errors.append(
                    f"'{key}' references column(s) not found in headers: {missing}"
                )

    # Recommendations
    if "groupingFactors" not in config:
        suggestions.append(
            "Add groupingFactors to compare survival curves between groups "
            "(e.g. treatment arms, disease stages)."
        )
    if "xAxisTitle" not in config:
        suggestions.append("Add xAxisTitle to label the time axis (e.g. 'Time (months)').")
    if "yAxisTitle" not in config:
        suggestions.append("Add yAxisTitle — recommend 'Survival Probability'.")
        fixed.setdefault("yAxisTitle", "Survival Probability")
    if "colorScheme" not in config:
        suggestions.append("Add colorScheme — 'Tableau' or 'ColorBlind' work well for KM.")
    if "showLegend" not in config and "groupingFactors" in config:
        suggestions.append("Add showLegend: true to display group labels.")
        fixed.setdefault("showLegend", True)

    return {
        "valid":        len(errors) == 0,
        "errors":       errors,
        "warnings":     warnings,
        "suggestions":  suggestions,
        "fixed_config": fixed,
    }


# ---------------------------------------------------------------------------
# LLM-based generation
# ---------------------------------------------------------------------------

def generate_km_config_llm(
    description:    str,
    headers:        Optional[list[str]],
    column_roles:   Optional[dict],
    llm_complete_fn,
    temperature:    float = 0.0,
) -> dict:
    """
    Use the LLM to generate a KM config, guided by the KM system prompt
    and detected column roles.
    """
    # Build user prompt
    parts = [f'Generate a CanvasXpress KaplanMeier config for:\n"{description}"']

    if column_roles:
        time_col  = column_roles.get("time_col")
        event_col = column_roles.get("event_col")
        groups    = column_roles.get("group_cols", [])
        if time_col:
            parts.append(f"\nTime column  : {time_col}  → use for xAxis")
        if event_col:
            parts.append(f"Event column : {event_col}  → use for yAxis")
        if groups:
            parts.append(f"Group column : {groups[0]}  → use for groupingFactors")
        if column_roles.get("notes"):
            for note in column_roles["notes"]:
                parts.append(f"Note: {note}")

    elif headers:
        parts.append(f"\nDataset columns: {', '.join(headers)}")
        parts.append("Identify which column is time, which is event/status, which is grouping.")

    parts.append("\nReturn ONLY the JSON config object.")
    user_prompt = "\n".join(parts)

    raw_text, _ = llm_complete_fn(
        system=KM_SYSTEM_PROMPT,
        user=user_prompt,
        temperature=temperature,
        max_tokens=800,
    )

    raw = raw_text.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0].strip()

    if not raw:
        return {}

    return json.loads(raw)


# ---------------------------------------------------------------------------
# Main handler (called from server.py MCP tool)
# ---------------------------------------------------------------------------

def handle_generate_km(
    description:    Optional[str],
    headers:        Optional[list[str]],
    data:           Optional[list[list]],
    config:         Optional[dict],
    add_annotations: bool,
    temperature:    float,
    llm_complete_fn,
) -> dict:
    """
    Orchestrates all four capabilities and returns a unified response.

    Returns:
        {
          "config":           dict,   # final KM config
          "valid":            bool,
          "errors":           list,
          "warnings":         list,
          "suggestions":      list,
          "column_detection": dict | None,
          "statistics":       dict | None,
          "decorations_added": bool,
        }
    """
    errors:   list[str] = []
    warnings: list[str] = []
    suggestions: list[str] = []
    column_detection = None
    statistics       = None
    decorations_added = False

    # ── Resolve headers from data ─────────────────────────────────────────────
    resolved_headers: Optional[list[str]] = headers
    if data and len(data) >= 1:
        resolved_headers = [str(h) for h in data[0]]

    # ── Step 1: Column detection ──────────────────────────────────────────────
    column_roles: Optional[dict] = None
    if resolved_headers:
        column_roles = detect_km_columns(resolved_headers)
        column_detection = {
            "time_col":   column_roles["time_col"],
            "event_col":  column_roles["event_col"],
            "group_cols": column_roles["group_cols"],
            "unassigned": column_roles["unassigned"],
            "confidence": column_roles["confidence"],
            "notes":      column_roles["notes"],
        }
        log.info(
            "KM column detection: time=%s event=%s groups=%s confidence=%s",
            column_roles["time_col"], column_roles["event_col"],
            column_roles["group_cols"], column_roles["confidence"],
        )
        warnings.extend(column_roles["notes"])

    # ── Step 2: Validate existing config (if provided) ────────────────────────
    working_config: dict = {}
    if config:
        validation = validate_km_config(config, resolved_headers)
        errors.extend(validation["errors"])
        warnings.extend(validation["warnings"])
        suggestions.extend(validation["suggestions"])
        working_config = validation["fixed_config"]
        log.info(
            "KM validation: valid=%s errors=%d warnings=%d",
            validation["valid"], len(validation["errors"]), len(validation["warnings"]),
        )

    # ── Step 3: Generate config via LLM ──────────────────────────────────────
    if not working_config or description:
        if description or resolved_headers:
            try:
                generated = generate_km_config_llm(
                    description  = description or "Kaplan-Meier survival curve",
                    headers      = resolved_headers,
                    column_roles = column_roles,
                    llm_complete_fn = llm_complete_fn,
                    temperature  = temperature,
                )
                if generated:
                    # Merge: generated values fill in anything missing from fixed config
                    for k, v in generated.items():
                        working_config.setdefault(k, v)
                    # Override graphType to always be correct
                    working_config["graphType"] = "KaplanMeier"
            except Exception as e:
                warnings.append(f"LLM generation failed: {e}. Config may be incomplete.")
                log.warning("KM LLM generation failed: %s", e)

    # Ensure graphType is always set
    working_config["graphType"] = "KaplanMeier"

    # ── Step 4: Compute statistics and add decorations ────────────────────────
    if data and add_annotations:
        time_col  = (column_roles or {}).get("time_col") or (
            working_config.get("xAxis", [None])[0]
        )
        event_col = (column_roles or {}).get("event_col") or (
            working_config.get("yAxis", [None])[0]
        )
        group_cols = (column_roles or {}).get("group_cols", [])
        group_col  = group_cols[0] if group_cols else None

        if time_col and event_col:
            try:
                stats = compute_km_stats(data, time_col, event_col, group_col)
                statistics = {
                    "groups": {
                        name: {
                            "n":               g["n"],
                            "n_events":        g["n_events"],
                            "median_survival": g["median_survival"],
                        }
                        for name, g in stats["groups"].items()
                    },
                    "logrank_pvalue": stats["logrank_pvalue"],
                    "pvalue_str":     stats["pvalue_str"],
                }
                warnings.extend(stats.get("warnings", []))

                # Build time axis label from xAxisTitle or column name
                time_label = working_config.get("xAxisTitle", time_col)

                decors = build_km_decorations(
                    stats          = stats,
                    add_median     = True,
                    add_pvalue     = bool(stats.get("pvalue_str")),
                    time_col_label = time_label,
                )
                if decors:
                    working_config["decorations"] = decors
                    decorations_added = True
                    log.info("KM: added %d decorations (medians + p-value)", len(decors))

            except Exception as e:
                warnings.append(f"Could not compute KM statistics: {e}")
                log.warning("KM stats computation failed: %s", e)
        else:
            warnings.append(
                "Skipping statistical annotations — could not identify time and event columns."
            )

    # ── Step 5: Final validation on the assembled config ─────────────────────
    final_validation = validate_km_config(working_config, resolved_headers)

    # Merge any new errors/warnings discovered during final validation
    for err in final_validation["errors"]:
        if err not in errors:
            errors.append(err)
    for sug in final_validation["suggestions"]:
        if sug not in suggestions:
            suggestions.append(sug)

    return {
        "config":             working_config,
        "valid":              len(errors) == 0,
        "errors":             errors,
        "warnings":           warnings,
        "suggestions":        suggestions,
        "column_detection":   column_detection,
        "statistics":         statistics,
        "decorations_added":  decorations_added,
    }
