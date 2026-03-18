#!/usr/bin/env python3
"""
CanvasXpress MCP Server — HTTP Transport
=============================================
All v3 improvements plus fast local vector search:
  - sqlite-vec + sentence-transformers for semantic retrieval
  - Scales to 3,000+ few-shot examples with ~10ms retrieval
  - Falls back to SequenceMatcher if index not built yet
  - Complete canvasxpress-LLM knowledge base in system prompt

Run build_index.py once before starting to build the vector index:
    python build_index.py
    python src/server.py

Runs at http://0.0.0.0:8100/mcp
"""

import json
import os
import sys
import struct
import logging
import sqlite3
from pathlib import Path
from difflib import SequenceMatcher
from typing import Optional

import numpy as np
import sqlite_vec
import anthropic
from fastmcp import FastMCP
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
DEBUG = os.environ.get("CX_DEBUG", "").lower() in ("1", "true", "yes")

logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("cx-mcp")

def _sep(title: str = "") -> None:
    """Print a debug separator to stderr."""
    if DEBUG:
        bar = "─" * 60
        print("", file=sys.stderr)
        print(bar, file=sys.stderr)
        if title:
            print("  " + title, file=sys.stderr)
            print(bar, file=sys.stderr)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
EXAMPLES_FILE = DATA_DIR / "few_shot_examples.json"
DB_FILE = DATA_DIR / "embeddings.db"

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2; change to 768 for all-mpnet-base-v2

HOST = os.environ.get("MCP_HOST", "0.0.0.0")
PORT = int(os.environ.get("MCP_PORT", "8100"))

# ---------------------------------------------------------------------------
# Few-shot examples (fallback for when vector index is not built)
# ---------------------------------------------------------------------------
def load_examples() -> list[dict]:
    if not EXAMPLES_FILE.exists():
        log.warning("few_shot_examples.json not found at %s", EXAMPLES_FILE)
        return []
    with open(EXAMPLES_FILE) as f:
        data = json.load(f)
    log.info("Loaded %d examples from JSON", len(data))
    return data

EXAMPLES: list[dict] = load_examples()

# ---------------------------------------------------------------------------
# Vector index — sqlite-vec + sentence-transformers
# ---------------------------------------------------------------------------

_embed_model: Optional[SentenceTransformer] = None
_use_vector_index: bool = False


def _serialize(vector: list[float]) -> bytes:
    """Pack float list to little-endian bytes for sqlite-vec."""
    return struct.pack(f"{len(vector)}f", *vector)


def _load_vector_index() -> bool:
    """Try to load the sqlite-vec index. Returns True if successful."""
    global _embed_model, _use_vector_index

    if not DB_FILE.exists():
        log.warning(
            "Vector index not found at %s. "
            "Run build_index.py to build it for faster/better retrieval. "
            "Falling back to SequenceMatcher.", DB_FILE
        )
        return False

    try:
        log.info("Loading embedding model: %s", EMBEDDING_MODEL)
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
        log.info("Vector index ready: %s", DB_FILE)
        _use_vector_index = True
        return True
    except Exception as e:
        log.warning("Failed to load embedding model (%s). Falling back to SequenceMatcher.", e)
        return False


def _vector_retrieve(query: str, top_k: int) -> list[dict]:
    """Retrieve top-k examples using sqlite-vec cosine similarity."""
    query_emb = _embed_model.encode([query], normalize_embeddings=True)[0]
    query_bytes = _serialize(query_emb.tolist())

    db = sqlite3.connect(str(DB_FILE))
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    rows = db.execute(
        """
        SELECT e.description, e.config, v.distance
        FROM vec_examples v
        JOIN examples e ON e.id = v.rowid
        WHERE v.embedding MATCH ?
          AND k = ?
        ORDER BY v.distance
        """,
        [query_bytes, top_k]
    ).fetchall()
    db.close()

    return [
        {"description": row[0], "config": json.loads(row[1])}
        for row in rows
    ]


def _fallback_retrieve(query: str, top_k: int) -> list[dict]:
    """Retrieve top-k examples using SequenceMatcher (no index required)."""
    scored = [
        (ex, SequenceMatcher(None, query.lower(), ex["description"].lower()).ratio())
        for ex in EXAMPLES
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [ex for ex, _ in scored[:top_k]]


def retrieve_examples(query: str, top_k: int = 6) -> list[dict]:
    """Retrieve the most relevant few-shot examples for the query."""
    if _use_vector_index:
        try:
            return _vector_retrieve(query, top_k)
        except Exception as e:
            log.warning("Vector retrieval failed (%s), falling back to SequenceMatcher", e)
    return _fallback_retrieve(query, top_k)


# Load vector index at startup
_load_vector_index()

# ---------------------------------------------------------------------------
# Knowledge base — sourced from neuhausi/canvasxpress-LLM
# ---------------------------------------------------------------------------

GRAPH_TYPE_CATEGORIES = """
SINGLE-DIMENSIONAL GRAPH TYPES (use xAxis only, never yAxis):
  Alluvial, Area, Bar, Boxplot, Bin, Binplot, Bubble, Bullet, CDF, Chord,
  Circular, Cleveland, Correlation, Density, Distribution, Donut, Dotplot,
  Dumbbell, Gantt, Heatmap, Hex, Hexplot, Histogram, Line, Lollipop, Meter,
  ParallelCoordinates, Pie, QQ, Quantile, Radar, Ribbon, Ridgeline, Sankey,
  Stacked, StackedPercent, TagCloud, Tornado, Tree, Treemap, Violin, Venn, Waterfall, WordCloud

COMBINED GRAPH TYPES (use xAxis + xAxis2, never yAxis):
  AreaLine, BarLine, DotLine, Pareto, StackedLine, StackedPercentLine

MULTI-DIMENSIONAL GRAPH TYPES (require both xAxis and yAxis):
  Bump, Contour, Scatter2D, Scatter3D, ScatterBubble2D, Spaghetti, Streamgraph, Volcano

GRAPH TYPES REQUIRING x/y DECORATION PARAMS:
  Bin, Binplot, CDF, Contour, Density, Hex, Hexplot, Histogram, KaplanMeier,
  QQ, Quantile, Ridgeline, Scatter2D, ScatterBubble2D, Spaghetti, Streamgraph, Volcano

GRAPH TYPES REQUIRING VALUE DECORATION PARAMS:
  Area, AreaLine, Bar, BarLine, Boxplot, DotLine, Dotplot, Line, Lollipop,
  Pareto, Stacked, StackedLine, StackedPercent, StackedPercentLine, Violin, Waterfall
"""

MINIMAL_PARAMETERS = """
MINIMAL REQUIRED PARAMETERS PER GRAPH TYPE:
  Alluvial: graphType, sankeyAxes, xAxis
  Area: graphType, xAxis
  AreaLine: graphType, xAxis, xAxis2
  Bar: graphType, xAxis
  BarLine: graphType, xAxis, xAxis2
  Bin: graphType, xAxis
  Binplot: graphType, xAxis
  Boxplot: graphType, groupingFactors, xAxis
  Bubble: graphType, hierarchy, xAxis
  Bump: graphType, lineBy, xAxis, yAxis
  CDF: graphType, xAxis
  Chord: graphType, xAxis
  Circular: graphType, xAxis
  Cleveland: graphType, xAxis
  Contour: graphType, xAxis, yAxis
  Correlation: graphType, xAxis
  Density: graphType, xAxis
  Distribution: graphType, xAxis
  Donut: graphType, xAxis
  DotLine: graphType, xAxis, xAxis2
  Dotplot: graphType, xAxis
  Dumbbell: graphType, xAxis
  Heatmap: graphType, xAxis
  Histogram: graphType, xAxis
  KaplanMeier: graphType, xAxis, yAxis
  Line: graphType, xAxis
  Lollipop: graphType, xAxis
  Network: graphType
  ParallelCoordinates: graphType, xAxis
  Pareto: graphType, xAxis, xAxis2
  Pie: graphType, xAxis
  QQ: graphType, xAxis
  Quantile: graphType, xAxis
  Radar: graphType, xAxis
  Ribbon: graphType, sankeyAxes, xAxis
  Ridgeline: graphType, ridgeBy, xAxis
  Sankey: graphType, sankeyAxes, xAxis
  Scatter2D: graphType, xAxis, yAxis
  Scatter3D: graphType, xAxis, yAxis, zAxis
  ScatterBubble2D: graphType, xAxis, yAxis, zAxis
  Spaghetti: colorBy, graphType, xAxis, yAxis
  Stacked: graphType, xAxis
  StackedLine: graphType, xAxis, xAxis2
  StackedPercent: graphType, xAxis
  StackedPercentLine: graphType, xAxis, xAxis2
  Streamgraph: graphType, xAxis, yAxis
  Sunburst: graphType, hierarchy
  TagCloud: colorBy, graphType, xAxis
  TimeSeries: graphType, xAxis, yAxis
  Tornado: graphType, xAxis
  Tree: graphType, hierarchy, xAxis
  Treemap: graphType, groupingFactors, xAxis
  Venn: graphType, vennGroups, xAxis
  Violin: graphType, groupingFactors, xAxis
  Volcano: graphType, xAxis, yAxis
  Waterfall: graphType, xAxis
  WordCloud: colorBy, graphType, xAxis
"""

DECISION_TREE = """
GRAPH TYPE SELECTION DECISION TREE:

ONE-DIMENSIONAL DATA:
  Compare categories → Bar (default), Lollipop (emphasis on points), Cleveland (horizontal), Dumbbell (change), Waterfall (cumulative)
  Show distribution (single) → Histogram, Density, CDF, QQ
  Show distribution (multiple groups) → Boxplot, Violin, Dotplot, Ridgeline
  Part-to-whole (circular) → Pie, Donut, Sunburst
  Part-to-whole (rectangular) → Treemap, Stacked, StackedPercent
  Ranking → Bar, Pareto (with threshold)
  Time series → Line, Area, Streamgraph, Bump (rank changes)

TWO-DIMENSIONAL DATA:
  Correlation/relationship (continuous) → Scatter2D, ScatterBubble2D (with size), Bin/Hexplot (density), Contour
  Correlation (categorical+continuous) → Scatter2D with colorBy, Boxplot, Violin
  Time series multi-variable → Line, Spaghetti, Ribbon (with confidence)
  Compare multiple metrics → BarLine, AreaLine, DotLine, Radar, ParallelCoordinates

THREE-DIMENSIONAL → Scatter3D
NETWORK → Network, Tree, Treemap, Sankey, Chord, Alluvial
GEOGRAPHICAL → Map
SET RELATIONSHIPS → Venn, Upset
SPECIAL PURPOSE → Correlation (matrix), Volcano (differential expression), KaplanMeier (survival), Gantt (scheduling), Tornado
"""

VALID_COLOR_SCHEMES = (
    "YlGn, YlGnBu, GnBu, BuGn, PuBuGn, PuBu, BuPu, RdPu, PuRd, OrRd, YlOrRd, YlOrBr, "
    "Purples, Blues, Greens, Oranges, Reds, Greys, PuOr, BrBG, PRGn, PiYG, RdBu, RdGy, "
    "RdYlBu, Spectral, RdYlGn, Bootstrap, Economist, Excel, GGPlot, Solarized, PaulTol, "
    "ColorBlind, Tableau, WallStreetJournal, Stata, BlackAndWhite, CanvasXpress"
)

VALID_THEMES = (
    "bw, classic, cx, dark, economist, excel, ggblanket, ggplot, gray, grey, "
    "highcharts, igray, light, linedraw, minimal, none, ptol, solarized, stata, tableau, void0, wsj"
)

GRAPH_SPECIFIC_RULES = """
GRAPH-TYPE-SPECIFIC REQUIRED PARAMETERS:
  Area: must include areaType ("overlapping"|"stacked"|"percent")
  Contour: must include xAxis (col 1) AND yAxis (col 2)
  Density: must include densityPosition ("normal"|"stacked"|"filled")
  Dumbbell: must include dumbbellType ("arrow"|"bullet"|"cleveland"|"connected"|"line"|"lineConnected"|"stacked")
  Histogram: must include histogramType ("dodged"|"staggered"|"stacked")
  Ridgeline: use ridgeBy instead of groupingFactors

AXIS RULES (CRITICAL):
  Single-Dimensional types: use xAxis ONLY — NEVER include yAxis or any y-axis params
    (yAxisTitle, yAxisTextColor, etc.). Use smpTitle/smpTextColor instead.
  Combined types: use xAxis + xAxis2 — NEVER include yAxis
  Multi-Dimensional types: MUST include both xAxis and yAxis; xAxis always listed first

DECORATION RULES:
  decorations keys: only "line", "point", or "text" for 1D; or scatter-specific keys
  Each decoration array item: only color, value, x, y, width, label (scalar values only)
  Only one of x/y/value per item; only include x/y for x/y-decoration graph types

SORTING: Never sort for: Bin, Binplot, CDF, Contour, Density, Hex, Hexplot, Histogram,
  KaplanMeier, QQ, Quantile, Ridgeline, Scatter2D, ScatterBubble2D, Streamgraph

FILTER: filterData is array of arrays: ["guess", columnName, "like"|"different", value]
SORT:   sortData is array of arrays: ["var"|"smp"|"cat", "var"|"smp", columnName]

AXIS MIN/MAX:
  Both xAxis+yAxis present: use setMinX/setMaxX and setMinY/setMaxY
  Only xAxis present: use setMinX/setMaxX only — NEVER use setMinY/setMaxY
"""

# ---------------------------------------------------------------------------
# System prompt — integrates full canvasxpress-LLM knowledge base
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = f"""You are an expert CanvasXpress data visualization assistant.

Your task is to generate a valid CanvasXpress JSON configuration object from a natural
language description and optional column headers.

## OUTPUT FORMAT
- Return ONLY a valid JSON object. No markdown, no backticks, no explanations.
- If you cannot generate a valid config, return an empty string.

## VALID GRAPH TYPES (use exactly these names)
Alluvial, Area, AreaLine, Bar, BarLine, Boxplot, Bin, Binplot, Bubble, Bullet, Bump,
CDF, Chord, Circular, Cleveland, Contour, Correlation, Density, Distribution, Donut,
DotLine, Dotplot, Dumbbell, Gantt, Heatmap, Hex, Hexplot, Histogram, KaplanMeier, Line,
Lollipop, Map, Meter, Network, ParallelCoordinates, Pareto, Pie, QQ, Quantile, Radar,
Ribbon, Ridgeline, Sankey, Scatter2D, Scatter3D, ScatterBubble2D, Spaghetti, Stacked,
StackedLine, StackedPercent, StackedPercentLine, Streamgraph, Sunburst, TagCloud,
TimeSeries, Tornado, Tree, Treemap, Upset, Violin, Volcano, Venn, Waterfall, WordCloud

Default to "Bar" if the type is ambiguous.

## VALID COLOR SCHEMES
{VALID_COLOR_SCHEMES}

## VALID THEMES
{VALID_THEMES}

## GRAPH TYPE CATEGORIES
{GRAPH_TYPE_CATEGORIES}

## DECISION TREE — CHOOSING GRAPH TYPE
{DECISION_TREE}

## MINIMAL REQUIRED PARAMETERS
{MINIMAL_PARAMETERS}

## CRITICAL RULES
{GRAPH_SPECIFIC_RULES}

## STEPS
1. Select graphType from the valid list using the decision tree
2. Set xAxis (and yAxis/zAxis if required) using provided column names
3. Set decorations if specified (follow decoration rules strictly)
4. Set filterData if filtering is requested
5. Set sortData if sorting is requested (skip for density/scatter types)
6. Set colorScheme from valid list if colors mentioned
7. Set theme from valid list if style mentioned
8. Set graph-type-specific required parameters (areaType, densityPosition, etc.)
9. Add any additional parameters from description
10. Validate: ensure minimal required parameters are present; return empty string if not valid
"""

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_config(
    description: str,
    headers: list[str] | None = None,
    column_types: dict[str, str] | None = None,
    temperature: float = 0.0,
) -> dict | str:
    import time

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set.")

    client = anthropic.Anthropic(api_key=api_key)

    # ── Step 1: Retrieval ────────────────────────────────────────────────────
    t0 = time.perf_counter()
    examples = retrieve_examples(description)
    t_retrieval = (time.perf_counter() - t0) * 1000

    if DEBUG:
        bar = "─" * 64
        print(f"\n{bar}\n  STEP 1 — RETRIEVAL\n{bar}", file=sys.stderr)
        print(f"  Query    : {description}", file=sys.stderr)
        print(f"  Method   : {'vector (sqlite-vec)' if _use_vector_index else 'SequenceMatcher (fallback)'}", file=sys.stderr)
        print(f"  Results  : {len(examples)} examples in {t_retrieval:.1f}ms", file=sys.stderr)
        for i, ex in enumerate(examples, 1):
            print(f"  [{i}] {ex['description'][:80]}", file=sys.stderr)

    # ── Step 2: Build prompt ─────────────────────────────────────────────────
    ex_text = "\n\n".join(
        f'Description: "{ex["description"]}"\nConfig: {json.dumps(ex["config"])}'
        for ex in examples
    )

    header_hint = ""
    if headers:
        if column_types:
            col_desc = ", ".join(
                f"{col} ({column_types.get(col, 'unknown')})" for col in headers
            )
            type_rules = (
                "\n   Column type rules:"
                "\n   - numeric : use for xAxis (scatter), yAxis, zAxis, value axes"
                "\n   - factor  : use for groupingFactors, colorBy, shapeBy, segregateSamplesBy"
                "\n   - string  : use for xAxis labels, smpOverlays, annotation overlays"
                "\n   - date    : use for xAxis in time series (set xAxisTitle to the date column)"
            )
            header_hint = (
                f"\n\nDataset columns with types: {col_desc}.{type_rules}"
                f"\n   Only assign columns to axes and parameters matching their type."
            )
        else:
            header_hint = (
                f"\n\nThe dataset has these column names: {', '.join(headers)}. "
                f"Use them for xAxis, yAxis, groupingFactors, colorBy, etc. as appropriate."
            )

    prompt = (
        f"Similar CanvasXpress examples for reference:\n\n{ex_text}\n\n"
        f"---\nGenerate the CanvasXpress config for:\n\"{description}\"{header_hint}\n\n"
        f"Return ONLY the JSON config object."
    )

    if DEBUG:
        bar = "─" * 64
        print(f"\n{bar}\n  STEP 2 — PROMPT\n{bar}", file=sys.stderr)
        print(f"  System prompt : {len(SYSTEM_PROMPT)} chars", file=sys.stderr)
        print(f"  User prompt   : {len(prompt)} chars", file=sys.stderr)
        print(f"  Headers       : {headers}", file=sys.stderr)
        if column_types:
            print(f"  Column types  : {column_types}", file=sys.stderr)
        print(f"  Temperature   : {temperature}", file=sys.stderr)
        print(f"\n  ── User prompt (first 600 chars) ──", file=sys.stderr)
        print("  " + prompt[:600].replace("\n", "\n  "), file=sys.stderr)

    # ── Step 3: LLM call ─────────────────────────────────────────────────────
    if DEBUG:
        bar = "─" * 64
        print(f"\n{bar}\n  STEP 3 — LLM CALL\n{bar}", file=sys.stderr)
        print(f"  Model    : claude-sonnet-4-20250514", file=sys.stderr)
        print(f"  Calling Anthropic API...", file=sys.stderr)

    t1 = time.perf_counter()
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        temperature=temperature,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    t_llm = (time.perf_counter() - t1) * 1000

    if DEBUG:
        usage = message.usage
        print(f"  Latency       : {t_llm:.0f}ms", file=sys.stderr)
        print(f"  Input tokens  : {usage.input_tokens}", file=sys.stderr)
        print(f"  Output tokens : {usage.output_tokens}", file=sys.stderr)
        print(f"  Stop reason   : {message.stop_reason}", file=sys.stderr)

    # ── Step 4: Parse response ───────────────────────────────────────────────
    raw = message.content[0].text.strip()

    if DEBUG:
        bar = "─" * 64
        print(f"\n{bar}\n  STEP 4 — RAW LLM RESPONSE\n{bar}", file=sys.stderr)
        print(f"  {raw}", file=sys.stderr)

    # Strip markdown fences
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0].strip()

    if not raw or raw == "''":
        if DEBUG:
            print("\n  ⚠️  Model returned empty string — could not generate valid config", file=sys.stderr)
        return {}

    config = json.loads(raw)

    if DEBUG:
        bar = "─" * 64
        print(f"\n{bar}\n  STEP 5 — PARSED CONFIG\n{bar}", file=sys.stderr)
        print(f"  graphType : {config.get('graphType', 'NOT SET')}", file=sys.stderr)
        print(f"  Keys      : {list(config.keys())}", file=sys.stderr)
        print(f"\n  Full config:", file=sys.stderr)
        print("  " + json.dumps(config, indent=2).replace("\n", "\n  "), file=sys.stderr)
        print(f"\n  ── Timing ──────────────────────────────────", file=sys.stderr)
        print(f"  Retrieval : {t_retrieval:.1f}ms", file=sys.stderr)
        print(f"  LLM       : {t_llm:.0f}ms", file=sys.stderr)
        print(f"  Total     : {t_retrieval + t_llm:.0f}ms", file=sys.stderr)
        print(f"{'─' * 64}\n", file=sys.stderr)

    return config


# ---------------------------------------------------------------------------
# Header validation
# ---------------------------------------------------------------------------

# All config keys that reference column names from the dataset
COLUMN_REF_KEYS = [
    "xAxis", "xAxis2", "yAxis", "zAxis",
    "groupingFactors", "segregateSamplesBy", "segregateVariablesBy",
    "smpOverlays", "varOverlays", "sankeyAxes",
    "colorBy", "shapeBy", "sizeBy", "stackBy", "pivotBy",
    "ridgeBy", "splitSamplesBy", "splitVariablesBy",
]


def extract_headers_from_data(data: list[list]) -> list[str]:
    """
    Extract column headers from a flat CSV-style array of arrays.
    The first row must contain the column names.

    Example input:
        [["Gene", "Sample1", "Sample2", "Treatment"],
         ["BRCA1", 1.2, 3.4, "Control"],
         ["TP53",  2.1, 0.9, "Treated"]]

    Returns: ["Gene", "Sample1", "Sample2", "Treatment"]
    """
    if not data or not isinstance(data[0], list):
        raise ValueError(
            "data must be an array of arrays where the first row contains column headers. "
            "Example: [[col1,col2,col3],[val1,val2,val3]]"
        )
    return [str(h).strip() for h in data[0]]


def validate_config_headers(config: dict, headers: list[str]) -> dict:
    """
    Check that every column name referenced in the config actually exists
    in the provided headers list.

    Returns a dict with:
        valid (bool)       - True if all referenced columns are found
        warnings (list)    - list of warning strings for missing columns
        invalid_refs (dict)- map of config key -> [missing column names]
    """
    header_set = {h.strip() for h in headers}
    warnings = []
    invalid_refs = {}

    for key in COLUMN_REF_KEYS:
        if key not in config:
            continue

        value = config[key]

        # Normalise to a list of strings
        if isinstance(value, str):
            candidates = [value]
        elif isinstance(value, list):
            candidates = [v for v in value if isinstance(v, str)]
        else:
            continue

        missing = [c for c in candidates if c not in header_set]
        if missing:
            invalid_refs[key] = missing
            warnings.append(
                f"'{key}' references column(s) not found in headers: {missing}"
            )

    return {
        "valid": len(warnings) == 0,
        "warnings": warnings,
        "invalid_refs": invalid_refs,
    }


# ---------------------------------------------------------------------------
# FastMCP server
# ---------------------------------------------------------------------------
mcp = FastMCP(
    name="canvasxpress-mcp",
    instructions=(
        "Generate accurate CanvasXpress visualization configs from plain English. "
        "Uses sqlite-vec semantic vector search over few-shot examples plus the full "
        "canvasxpress-LLM knowledge base (RULES, SCHEMA, DECISION-TREE, MINIMAL-PARAMETERS) "
        "for highly accurate, validated output. Scales to 3000+ examples."
    ),
)


@mcp.tool(
    description=(
        "Generate a CanvasXpress visualization config from a plain English description. "
        "Accepts headers, a CSV-style data array, and optional column_types metadata "
        "(string/numeric/factor/date) to guide axis assignment and grouping. "
        "All column references in the generated config are validated against the provided columns. "
        "Examples: 'clustered heatmap with RdBu colors', 'volcano plot with fold change on x-axis', "
        "'violin plot of gene expression by cell type', 'survival curve for two treatment groups'. "
        "Returns a validated JSON config object ready to pass to new CanvasXpress()."
    )
)
def generate_canvasxpress_config(
    description: str,
    headers: list[str] | None = None,
    data: list[list] | None = None,
    column_types: dict[str, str] | None = None,
    temperature: float = 0.0,
) -> dict:
    """
    Args:
        description: Plain English chart description.
        headers: Optional list of column names from your dataset.
                 e.g. ["Gene", "Sample1", "Sample2", "Treatment"]
        data: Optional flat CSV-style array of arrays where the first row
              contains column headers and subsequent rows contain data values.
              e.g. [["Gene","Sample1","Treatment"],["BRCA1",1.2,"Control"]]
              When provided, headers are extracted from row 0 automatically.
              If both headers and data are provided, data takes precedence.
        column_types: Optional dict mapping column name to type.
                      Valid types: "string", "numeric", "factor", "date".
                      e.g. {"Gene": "string", "Expression": "numeric", "Treatment": "factor"}
                      Guides axis assignment: numerics to yAxis, factors to groupingFactors.
        temperature: LLM creativity 0.0-1.0 (default 0.0 = deterministic).

    Returns:
        Dict with keys:
          config       (dict)  - the CanvasXpress JSON config object
          valid        (bool)  - True if all column refs are valid (or no headers given)
          warnings     (list)  - column validation warnings (empty if valid)
          invalid_refs (dict)  - map of config key to list of missing column names
          headers_used (list)  - the column names actually used for validation
          types_used   (dict)  - the column types passed in (if provided)
    """
    # ── Resolve headers ──────────────────────────────────────────────────────
    # data takes precedence over headers if both are supplied
    resolved_headers: list[str] | None = None

    if data is not None:
        try:
            resolved_headers = extract_headers_from_data(data)
            log.info("Extracted %d headers from data array", len(resolved_headers))
            if DEBUG:
                bar = "─" * 64
                print(f"\n{bar}\n  DATA INPUT\n{bar}", file=sys.stderr)
                print(f"  Rows (incl. header) : {len(data)}", file=sys.stderr)
                print(f"  Columns extracted   : {resolved_headers}", file=sys.stderr)
        except ValueError as e:
            return {
                "config": {},
                "valid": False,
                "warnings": [str(e)],
                "invalid_refs": {},
                "headers_used": [],
            }
    elif headers is not None:
        resolved_headers = headers

    # Validate column_types values
    VALID_TYPES = {"string", "numeric", "factor", "date"}
    if column_types:
        bad = {k: v for k, v in column_types.items() if v not in VALID_TYPES}
        if bad:
            log.warning("Unknown column types ignored: %s. Valid: %s", bad, VALID_TYPES)
            column_types = {k: v for k, v in column_types.items() if v in VALID_TYPES}

    if DEBUG and column_types:
        bar = "─" * 64
        print(f"\n{bar}\n  COLUMN TYPES\n{bar}", file=sys.stderr)
        for col, typ in column_types.items():
            print(f"  {col:25s} → {typ}", file=sys.stderr)

    log.info("Generating config for: %s", description)
    config = generate_config(description, resolved_headers, column_types, temperature)
    graph_type = config.get("graphType", "unknown") if config else "none"
    log.info("Generated graphType: %s", graph_type)

    # ── Validate column references ───────────────────────────────────────────
    if resolved_headers and config:
        validation = validate_config_headers(config, resolved_headers)
        if DEBUG:
            bar = "─" * 64
            print(f"\n{bar}\n  STEP 6 — HEADER VALIDATION\n{bar}", file=sys.stderr)
            print(f"  Headers used : {resolved_headers}", file=sys.stderr)
            print(f"  Source       : {'data array' if data is not None else 'headers list'}", file=sys.stderr)
            print(f"  Valid        : {validation['valid']}", file=sys.stderr)
            if validation["warnings"]:
                for w in validation["warnings"]:
                    print(f"  ⚠️  {w}", file=sys.stderr)
            else:
                print("  ✅ All column references match provided columns", file=sys.stderr)
            print(f"{'─' * 64}\n", file=sys.stderr)
        if not validation["valid"]:
            log.warning("Header validation warnings: %s", validation["warnings"])
    else:
        validation = {"valid": True, "warnings": [], "invalid_refs": {}}
        if DEBUG:
            log.debug("No headers or data provided — skipping column validation")

    return {
        "config": config,
        "valid": validation["valid"],
        "warnings": validation["warnings"],
        "invalid_refs": validation["invalid_refs"],
        "headers_used": resolved_headers or [],
        "types_used": column_types or {},
    }


@mcp.tool(description="List all supported CanvasXpress chart types with descriptions and categories.")
def list_chart_types() -> dict:
    """Returns chart types organized by category."""
    return {
        "single_dimensional": {
            "Bar": "Vertical or horizontal bars; grouped, stacked, diverging",
            "Line": "Trends over time or categories",
            "Area": "Filled line; overlapping, stacked, or percent",
            "Boxplot": "Box-and-whisker distribution summary",
            "Violin": "Kernel density distribution",
            "Heatmap": "Color matrix with optional clustering/dendrograms",
            "Pie": "Part-to-whole circular",
            "Donut": "Pie with center hole",
            "Stacked": "Stacked bars (absolute)",
            "StackedPercent": "100% stacked bars",
            "Histogram": "Binned frequency",
            "Density": "Smooth kernel density curve",
            "Dotplot": "Individual data points",
            "Lollipop": "Dot + stem for rankings",
            "Waterfall": "Cumulative change",
            "Cleveland": "Horizontal dot plot",
            "Dumbbell": "Before/after comparison",
            "Ridgeline": "Overlapping density curves by group",
            "Treemap": "Hierarchical nested rectangles",
            "Sankey": "Flow diagram",
            "Chord": "Circular flow between categories",
            "Alluvial": "Multi-level flow",
            "Venn": "Set overlaps",
            "Radar": "Spider/radar chart",
            "WordCloud": "Text frequency visualization",
            "CDF": "Cumulative distribution function",
        },
        "multi_dimensional": {
            "Scatter2D": "2D scatter; PCA, UMAP, tSNE, MA, volcano-style",
            "Scatter3D": "3D scatter",
            "ScatterBubble2D": "Bubble chart (size = 3rd variable)",
            "Volcano": "Volcano plot for differential expression",
            "Contour": "2D density contour",
            "Spaghetti": "Connected scatter for longitudinal data",
            "Streamgraph": "Flowing stacked area over time",
        },
        "combined": {
            "BarLine": "Bar + line overlay",
            "AreaLine": "Area + line overlay",
            "DotLine": "Dot + line overlay",
            "Pareto": "Bar + cumulative line",
        },
        "network_hierarchy": {
            "Network": "Force-directed node-edge graph",
            "Tree": "Hierarchical tree",
            "Sunburst": "Radial hierarchy",
        },
        "special": {
            "KaplanMeier": "Survival curves",
            "Correlation": "Correlation matrix",
            "Gantt": "Project scheduling",
            "Tornado": "Sensitivity analysis",
            "TimeSeries": "Time series with irregular intervals",
        },
    }


@mcp.tool(description="Get an explanation of any CanvasXpress configuration property.")
def explain_config_property(property: str) -> str:
    """
    Args:
        property: The config property name to explain.
    """
    explanations = {
        "graphType": "The chart type. One of 70+ supported types (Bar, Heatmap, Scatter2D, etc.)",
        "xAxis": "Array of column names for the x-axis. For single-dimensional graphs this is the only axis.",
        "yAxis": "Array of column names for the y-axis. Only for multi-dimensional graph types.",
        "zAxis": "Array of column names for the z-axis. Required for Scatter3D and ScatterBubble2D.",
        "xAxis2": "Secondary x-axis for combined graph types (BarLine, AreaLine, etc.)",
        "groupingFactors": "Array of column names used to group/color data. e.g. ['Treatment', 'CellType']",
        "colorBy": "Column name to color data points by. e.g. 'Species'",
        "shapeBy": "Column name to assign different shapes to data points.",
        "sizeBy": "Column name to scale data point sizes.",
        "colorScheme": f"Color palette. Valid options: {VALID_COLOR_SCHEMES}",
        "theme": f"Visual theme. Valid options: {VALID_THEMES}",
        "title": "Chart title string.",
        "xAxisTitle": "X-axis label (for multi-dimensional graphs).",
        "yAxisTitle": "Y-axis label (for multi-dimensional graphs).",
        "smpTitle": "Sample axis title for single-dimensional graphs (replaces yAxisTitle).",
        "samplesClustered": "Hierarchically cluster columns with a dendrogram. Use for heatmaps.",
        "variablesClustered": "Hierarchically cluster rows with a dendrogram. Use for heatmaps.",
        "showLegend": "Show/hide legend. Boolean.",
        "legendPosition": "Legend position: topRight, right, bottomRight, bottom, bottomLeft, left, topLeft, top.",
        "graphOrientation": "Bar/chart direction: 'horizontal' or 'vertical'.",
        "showRegressionFit": "Show regression line on scatter plots. Boolean.",
        "regressionType": "Regression type: linear, exponential, logarithmic, power, polynomial.",
        "showLoessFit": "Show LOESS/lowess smooth fit on scatter plots. Boolean.",
        "showConfidenceIntervals": "Show confidence bands. Boolean.",
        "transformData": "Data transformation: log2, log10, -log2, -log10, exp2, exp10, sqrt, percentile, zscore.",
        "xAxisTransform": "X-axis transform: log2, log10, -log2, -log10, etc.",
        "yAxisTransform": "Y-axis transform: log2, log10, -log2, -log10, etc.",
        "segregateSamplesBy": "Array of columns to facet/split samples into subplots.",
        "segregateVariablesBy": "Array of columns to facet/split variables into subplots.",
        "filterData": 'Array of filter arrays: [["guess", "colName", "like", "value"]]',
        "sortData": 'Array of sort arrays: [["smp", "smp", "colName"]]',
        "areaType": "Area chart subtype: overlapping, stacked, percent.",
        "densityPosition": "Density chart layout: normal (overlapping), stacked, filled.",
        "histogramType": "Histogram style with multiple series: dodged, staggered, stacked.",
        "dumbbellType": "Dumbbell style: arrow, bullet, cleveland, connected, line, lineConnected, stacked.",
        "boxplotNotched": "Show notched boxplots. Boolean.",
        "showBoxplotOriginalData": "Overlay original data points on boxplots. Boolean.",
        "jitter": "Jitter data points in dotplots/boxplots/scatter. Boolean.",
        "showViolinBoxplot": "Show embedded boxplot inside violin. Boolean.",
        "decorations": "Visual annotations: lines, points, or text overlaid on the chart.",
        "smpOverlays": "Sample metadata columns to overlay as annotation tracks on 1D plots.",
        "varOverlays": "Variable metadata columns to overlay on heatmaps.",
        "setMinX": "Set minimum x-axis value.",
        "setMaxX": "Set maximum x-axis value.",
        "setMinY": "Set minimum y-axis value.",
        "setMaxY": "Set maximum y-axis value.",
        "ridgeBy": "Column name for creating ridgeline plots (replaces groupingFactors for Ridgeline).",
        "sankeyAxes": "Array of column names for Sankey/Alluvial/Ribbon flow axes.",
        "pivotBy": "Column to pivot data with (reshape from wide to long).",
        "stackBy": "Column to stack samples in bar graphs.",
    }
    if property in explanations:
        return f"**`{property}`** — {explanations[property]}"
    return (
        f"No built-in explanation for `{property}`. "
        f"See the full API: https://canvasxpress.org/api/general.html"
    )


@mcp.tool(description="Get the minimal required parameters for a specific CanvasXpress graph type.")
def get_minimal_parameters(graph_type: str) -> dict:
    """
    Args:
        graph_type: The CanvasXpress graph type (e.g. 'Scatter2D', 'Heatmap').

    Returns:
        Dict with required parameters and their descriptions.
    """
    minimal = {
        "Alluvial": ["graphType", "sankeyAxes", "xAxis"],
        "Area": ["graphType", "xAxis"],
        "AreaLine": ["graphType", "xAxis", "xAxis2"],
        "Bar": ["graphType", "xAxis"],
        "BarLine": ["graphType", "xAxis", "xAxis2"],
        "Boxplot": ["graphType", "groupingFactors", "xAxis"],
        "CDF": ["graphType", "xAxis"],
        "Chord": ["graphType", "xAxis"],
        "Circular": ["graphType", "xAxis"],
        "Cleveland": ["graphType", "xAxis"],
        "Contour": ["graphType", "xAxis", "yAxis"],
        "Correlation": ["graphType", "xAxis"],
        "Density": ["graphType", "xAxis"],
        "Distribution": ["graphType", "xAxis"],
        "Donut": ["graphType", "xAxis"],
        "DotLine": ["graphType", "xAxis", "xAxis2"],
        "Dotplot": ["graphType", "xAxis"],
        "Dumbbell": ["graphType", "xAxis"],
        "Heatmap": ["graphType", "xAxis"],
        "Histogram": ["graphType", "xAxis"],
        "KaplanMeier": ["graphType", "xAxis", "yAxis"],
        "Line": ["graphType", "xAxis"],
        "Lollipop": ["graphType", "xAxis"],
        "Network": ["graphType"],
        "ParallelCoordinates": ["graphType", "xAxis"],
        "Pareto": ["graphType", "xAxis", "xAxis2"],
        "Pie": ["graphType", "xAxis"],
        "QQ": ["graphType", "xAxis"],
        "Radar": ["graphType", "xAxis"],
        "Ridgeline": ["graphType", "ridgeBy", "xAxis"],
        "Sankey": ["graphType", "sankeyAxes", "xAxis"],
        "Scatter2D": ["graphType", "xAxis", "yAxis"],
        "Scatter3D": ["graphType", "xAxis", "yAxis", "zAxis"],
        "ScatterBubble2D": ["graphType", "xAxis", "yAxis", "zAxis"],
        "Spaghetti": ["colorBy", "graphType", "xAxis", "yAxis"],
        "Stacked": ["graphType", "xAxis"],
        "StackedLine": ["graphType", "xAxis", "xAxis2"],
        "StackedPercent": ["graphType", "xAxis"],
        "StackedPercentLine": ["graphType", "xAxis", "xAxis2"],
        "Streamgraph": ["graphType", "xAxis", "yAxis"],
        "Sunburst": ["graphType", "hierarchy"],
        "TagCloud": ["colorBy", "graphType", "xAxis"],
        "TimeSeries": ["graphType", "xAxis", "yAxis"],
        "Tornado": ["graphType", "xAxis"],
        "Tree": ["graphType", "hierarchy", "xAxis"],
        "Treemap": ["graphType", "groupingFactors", "xAxis"],
        "Venn": ["graphType", "vennGroups", "xAxis"],
        "Violin": ["graphType", "groupingFactors", "xAxis"],
        "Volcano": ["graphType", "xAxis", "yAxis"],
        "Waterfall": ["graphType", "xAxis"],
        "WordCloud": ["colorBy", "graphType", "xAxis"],
    }

    gt = graph_type.strip()
    if gt in minimal:
        return {"graphType": gt, "required_parameters": minimal[gt]}

    return {
        "error": f"Unknown graph type '{gt}'.",
        "tip": "Use list_chart_types to see all valid graph types.",
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    log.info("Starting CanvasXpress MCP HTTP server on %s:%d", HOST, PORT)
    log.info("MCP endpoint: http://%s:%d/mcp", HOST if HOST != "0.0.0.0" else "localhost", PORT)
    if DEBUG:
        log.info("Debug mode ON  — set CX_DEBUG=0 to disable")
        log.info("Retrieval : %s", "vector (sqlite-vec)" if _use_vector_index else "SequenceMatcher (fallback)")
        log.info("Examples  : %d loaded", len(EXAMPLES))
    mcp.run(transport="http", host=HOST, port=PORT)
