#!/usr/bin/env python3
"""
cx_survival.py
==============
Kaplan-Meier survival analysis skill for the CanvasXpress MCP server.

Exposes one MCP tool — generate_km_config — that handles three
capabilities in a single call, based on what inputs are provided:

  1. GENERATE   Plain English description → full KM config
  2. VALIDATE   Existing config → structured report of errors and fixes
  3. DETECT     Dataset (headers or data array) → identifies time, event,
                and color columns using heuristic + LLM fallback

Note: KM statistics (median survival, confidence intervals, etc.) are
computed and rendered by CanvasXpress itself — this tool does not
replicate them.

Input routing:
  - description only           → generate from scratch
  - config only                → validate existing config
  - headers / data only        → detect columns, then generate
  - description + headers/data → generate with column guidance
  - any combination            → all applicable capabilities run in sequence

KM-specific CanvasXpress parameters handled:
  graphType                  : always "KaplanMeier"
  xAxis                      : [time_column]   — numeric survival/follow-up time
  yAxis                      : [event_column]  — 0/1 event indicator (1 = event occurred)
  colorBy                    : "color_column"  — categorical variable used to color the curves (string, not a list)
  xAxisTitle                 : human-readable time label
  yAxisTitle                 : "Survival Probability"
  colorScheme                : user-specified or "Tableau" default
  showLegend                 : true
  kmRiskTable                : show at-risk counts table below the plot
  showKMConfidenceIntervals  : show 95% confidence bands around each curve
  showKMMedianSurvivalTime   : annotate each curve with its median survival time
  decorations                : median survival lines + optional p-value annotation
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
- colorBy   MUST be a plain string (NOT a list) naming the color column (treatment arm, stage, etc.) — NEVER use groupingFactors
- xAxisTitle should describe the time units (e.g. "Time (months)", "Days since diagnosis")
- yAxisTitle should be "Survival Probability" or "Survival Function"
- showLegend should be true when colorBy is present
- colorScheme default: "Tableau"
- DO NOT include yAxis-related axis range params unless explicitly requested
- decorations: use "line" type with "value" (not x/y) for horizontal median lines

## VISUAL PARAMETERS (include when relevant)
- kmRiskTable               : true to show at-risk counts table below the plot
- showKMConfidenceIntervals : true to show 95% confidence bands around each curve
- showKMMedianSurvivalTime  : true to annotate each curve with its median survival time

## REQUIRED MINIMAL PARAMETERS
graphType, xAxis, yAxis, colorBy (when a color column is present)

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
_COLOR_PATTERNS = [
    r'\bgroup\b', r'\barm\b', r'\btreat(ment)?\b', r'\bcohort\b',
    r'\bstage\b', r'\bgrade\b', r'\bsubgroup\b', r'\bcategory\b',
    r'\bstrat(um|a|ify)?\b', r'\bcondition\b', r'\btherapy\b',
]

# ---------------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------------

def _score_col(col: str, patterns: list[str]) -> int:
    col_lower = col.lower()
    return sum(1 for p in patterns if re.search(p, col_lower))


def detect_km_columns(headers: list[str]) -> dict:
    """
    Heuristically identify time, event, and color columns from a list
    of column names.

    Returns:
        {
          "time_col":   str | None,
          "event_col":  str | None,
          "color_cols": list[str],
          "unassigned": list[str],
          "confidence": "high" | "medium" | "low",
          "notes":      list[str],
        }
    """
    notes: list[str] = []
    candidates: dict[str, dict] = {}

    for col in headers:
        candidates[col] = {
            "time_score":  _score_col(col, _TIME_PATTERNS),
            "event_score": _score_col(col, _EVENT_PATTERNS),
            "color_score": _score_col(col, _COLOR_PATTERNS),
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

    # Color columns: all cols with any group score not already used
    color_cols = [
        col for col in headers
        if col not in used and candidates[col]["color_score"] > 0
    ]
    used.update(color_cols)

    unassigned = [col for col in headers if col not in used]

    # Confidence scoring
    found = sum([time_col is not None, event_col is not None, bool(color_cols)])
    confidence = "high" if found == 3 else ("medium" if found == 2 else "low")

    if not time_col:
        notes.append("Could not detect a time column. Look for a numeric column "
                     "representing follow-up duration (days, months, years).")
    if not event_col:
        notes.append("Could not detect an event/status column. Look for a 0/1 "
                     "indicator column (1=event occurred, 0=censored).")
    if not color_cols:
        notes.append("No color column detected. A KM plot without colorBy "
                     "shows a single overall survival curve.")
    if unassigned:
        notes.append(f"Unassigned columns (may be IDs or covariates): {unassigned}")

    return {
        "time_col":   time_col,
        "event_col":  event_col,
        "color_cols": color_cols,
        "unassigned": unassigned,
        "confidence": confidence,
        "notes":      notes,
    }


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

_KM_REQUIRED = ["graphType", "xAxis", "yAxis"]
_KM_RECOMMENDED = ["colorBy", "xAxisTitle", "yAxisTitle", "colorScheme", "showLegend",
                   "kmRiskTable", "showKMConfidenceIntervals", "showKMMedianSurvivalTime"]


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

    # groupingFactors is wrong for KM — should be colorBy
    if "groupingFactors" in config:
        warnings.append(
            "groupingFactors is not valid for KaplanMeier. Use colorBy (a plain string) instead."
        )
        if "colorBy" not in fixed:
            gf = fixed.pop("groupingFactors")
            # groupingFactors was a list — unwrap to a string
            fixed["colorBy"] = gf[0] if isinstance(gf, list) and gf else gf
        else:
            fixed.pop("groupingFactors", None)

    # colorBy must be a string, not a list
    if "colorBy" in config and isinstance(config.get("colorBy"), list):
        warnings.append("colorBy must be a plain string, not a list. Using first element.")
        if config["colorBy"]:
            fixed["colorBy"] = fixed["colorBy"][0]
        else:
            fixed.pop("colorBy", None)

    # Column references against headers
    if headers:
        header_set = set(headers)
        for key in ["xAxis", "yAxis"]:
            val = config.get(key)
            if not val:
                continue
            cols = [val] if isinstance(val, str) else val
            missing = [c for c in cols if isinstance(c, str) and c not in header_set]
            if missing:
                errors.append(
                    f"'{key}' references column(s) not found in headers: {missing}"
                )
        color_by = fixed.get("colorBy")
        if color_by and isinstance(color_by, str) and color_by not in header_set:
            errors.append(f"'colorBy' references column '{color_by}' not found in headers")

    # Recommendations
    if "colorBy" not in config:
        suggestions.append(
            "Add colorBy to color survival curves by a column "
            "(e.g. treatment arm, disease stage)."
        )
    if "xAxisTitle" not in config:
        suggestions.append("Add xAxisTitle to label the time axis (e.g. 'Time (months)').")
    if "yAxisTitle" not in config:
        suggestions.append("Add yAxisTitle — recommend 'Survival Probability'.")
        fixed.setdefault("yAxisTitle", "Survival Probability")
    if "colorScheme" not in config:
        suggestions.append("Add colorScheme — 'Tableau' or 'ColorBlind' work well for KM.")
    if "showLegend" not in config and "colorBy" in config:
        suggestions.append("Add showLegend: true to display color column labels.")
        fixed.setdefault("showLegend", True)
    if "kmRiskTable" not in config:
        suggestions.append("Consider kmRiskTable: true to show at-risk counts below the plot.")
    if "showKMConfidenceIntervals" not in config:
        suggestions.append("Consider showKMConfidenceIntervals: true to display 95% confidence bands.")
    if "showKMMedianSurvivalTime" not in config:
        suggestions.append("Consider showKMMedianSurvivalTime: true to annotate median survival on each curve.")

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
        event_col  = column_roles.get("event_col")
        color_cols = column_roles.get("color_cols", [])
        if time_col:
            parts.append(f"\nTime column  : {time_col}  → use for xAxis")
        if event_col:
            parts.append(f"Event column : {event_col}  → use for yAxis")
        if color_cols:
            parts.append(f"Color column : {color_cols[0]}  → use for colorBy (string, e.g. colorBy: \"{color_cols[0]}\")")
        if column_roles.get("notes"):
            for note in column_roles["notes"]:
                parts.append(f"Note: {note}")

    elif headers:
        parts.append(f"\nDataset columns: {', '.join(headers)}")
        parts.append("Identify which column is time, which is event/status, which is the color column.")

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
    temperature:    float,
    llm_complete_fn,
) -> dict:
    """
    Orchestrates generate, validate, and detect capabilities and returns a unified response.

    Returns:
        {
          "config":           dict,        # final KM config
          "valid":            bool,
          "errors":           list,
          "warnings":         list,
          "suggestions":      list,
          "column_detection": dict | None,
        }
    """
    errors:   list[str] = []
    warnings: list[str] = []
    suggestions: list[str] = []
    column_detection = None

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
            "color_cols": column_roles["color_cols"],
            "unassigned": column_roles["unassigned"],
            "confidence": column_roles["confidence"],
            "notes":      column_roles["notes"],
        }
        log.info(
            "KM column detection: time=%s event=%s color=%s confidence=%s",
            column_roles["time_col"], column_roles["event_col"],
            column_roles["color_cols"], column_roles["confidence"],
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

    # ── Step 4: Final validation on the assembled config ─────────────────────
    final_validation = validate_km_config(working_config, resolved_headers)

    for err in final_validation["errors"]:
        if err not in errors:
            errors.append(err)
    for sug in final_validation["suggestions"]:
        if sug not in suggestions:
            suggestions.append(sug)

    return {
        "config":           working_config,
        "valid":            len(errors) == 0,
        "errors":           errors,
        "warnings":         warnings,
        "suggestions":      suggestions,
        "column_detection": column_detection,
    }
