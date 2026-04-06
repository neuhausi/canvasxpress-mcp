#!/usr/bin/env python3
"""
CanvasXpress MCP Server — HTTP Transport
=============================================
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

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")  # load .env before any os.environ.get calls

from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, HTMLResponse, Response

import numpy as np
import sqlite_vec
from fastmcp import FastMCP
from llm_providers import complete as llm_complete, provider_info, PROVIDER, MODEL
import cx_knowledge
import cx_survival
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
CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get("CORS_ORIGINS", "*").split(",")
    if o.strip()
]

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
# Knowledge DB — SQLite-backed, graph-type-aware prompt context
# ---------------------------------------------------------------------------

# Graph type keyword detection
_GRAPH_TYPE_KEYWORDS: dict[str, str] = {
    "heatmap": "Heatmap", "heat map": "Heatmap",
    "scatter": "Scatter2D", "scatter plot": "Scatter2D", "scatterplot": "Scatter2D",
    "pca": "Scatter2D", "umap": "Scatter2D", "tsne": "Scatter2D", "t-sne": "Scatter2D",
    "volcano": "Volcano",
    "bar chart": "Bar", "bar graph": "Bar", "barplot": "Bar", "bar plot": "Bar",
    "boxplot": "Boxplot", "box plot": "Boxplot",
    "violin": "Violin",
    "line chart": "Line", "line graph": "Line", "line plot": "Line",
    "3d scatter": "Scatter3D", "scatter 3d": "Scatter3D",
    "histogram": "Histogram",
    "density": "Density",
    "sankey": "Sankey", "alluvial": "Alluvial",
    "network": "Network",
    "venn": "Venn",
    "treemap": "Treemap", "tree map": "Treemap",
    "survival": "KaplanMeier", "kaplan": "KaplanMeier",
    "pie chart": "Pie", "donut": "Donut",
    "area chart": "Area", "area plot": "Area",
    "lollipop": "Lollipop", "waterfall": "Waterfall",
    "correlation": "Correlation",
    "bubble": "ScatterBubble2D",
    "ridgeline": "Ridgeline", "ridge": "Ridgeline",
    "gantt": "Gantt", "tornado": "Tornado",
}

# Contradiction keywords - 2+ hits triggers Tier 3
_CONTRADICTION_KEYWORDS = [
    "pie", "correlation", "regression", "3d", "bubble",
    "survival", "kaplan", "gantt", "sankey", "network",
    "venn", "treemap", "volcano", "scatter", "heatmap",
]


def detect_graph_type(description: str) -> Optional[str]:
    """Infer likely graph type from description using keyword matching."""
    desc_lower = description.lower()
    for kw in sorted(_GRAPH_TYPE_KEYWORDS, key=len, reverse=True):
        if kw in desc_lower:
            return _GRAPH_TYPE_KEYWORDS[kw]
    return None


def detect_tier(
    description: str,
    headers: list[str] | None,
    data: list[list] | None,
) -> int:
    """Detect which prompt tier to use (1, 2, or 3)."""
    has_data = headers is not None or data is not None
    keyword_hits = sum(kw in description.lower() for kw in _CONTRADICTION_KEYWORDS)
    if keyword_hits >= 2:
        return 3
    if has_data:
        return 2
    return 1


_SYSTEM_PROMPT_HEADER = """You are an expert CanvasXpress data visualization assistant.
Your task is to generate a valid CanvasXpress JSON configuration object from a natural
language description and optional column headers and column types.

## OUTPUT FORMAT
Return ONLY a valid JSON object. No markdown, no backticks, no explanations.
If you cannot generate a valid config, return an empty string.

## STEP 1 — SELECT GRAPH TYPE
Choose graphType from this exact list (default to "Bar" if ambiguous):
  Alluvial, Area, AreaLine, Bar, BarLine, Boxplot, Bin, Binplot, Bubble, Bullet, Bump,
  CDF, Chord, Circular, Cleveland, Contour, Correlation, Density, Distribution, Donut,
  DotLine, Dotplot, Dumbbell, Gantt, Heatmap, Hex, Hexplot, Histogram, KaplanMeier, Line,
  Lollipop, Map, Meter, Network, ParallelCoordinates, Pareto, Pie, QQ, Quantile, Radar,
  Ribbon, Ridgeline, Sankey, Scatter2D, Scatter3D, ScatterBubble2D, Spaghetti, Stacked,
  StackedLine, StackedPercent, StackedPercentLine, Streamgraph, Sunburst, TagCloud,
  TimeSeries, Tornado, Tree, Treemap, Upset, Violin, Volcano, Venn, Waterfall, WordCloud

## STEP 2 — ASSIGN AXES (most critical structural decision)
First classify the graphType, then assign axes accordingly.

SINGLE-DIMENSIONAL (xAxis ONLY — NEVER yAxis):
  Bar, Boxplot, Violin, Heatmap, Line, Area, Histogram, Density, Dotplot, Lollipop,
  Waterfall, Ridgeline, Pie, Donut, Stacked, StackedPercent, Chord, Sankey, Alluvial,
  Ribbon, Treemap, Venn, Radar, CDF, QQ, Quantile, Cleveland, Dumbbell, Gantt,
  TagCloud, WordCloud, Sunburst, Bubble, Network, Correlation, and all others not below.

  CONCEPTUAL MODEL (critical — get this right first):
  - Single-dimensional graphs have TWO axes with fundamentally different roles:
      xAxis   = the NUMERIC variable being plotted (heights of bars, positions of points,
                values in the distribution). This is the data column with numbers.
      samples = the CATEGORICAL labels (gene names, patient IDs, time points, groups).
                Samples are NOT set via a parameter — they come from the data automatically.
  - smpTitle labels the sample/categorical axis. It is the equivalent of what yAxisTitle
    is for multi-dimensional graphs, but for the categorical dimension of 1D charts.
    Use smpTitle instead of yAxisTitle on all single-dimensional chart types; use
    smpText, smpTextColor, smpTextScaleFontFactor, smpTextRotate, etc, for sample axis labels.

  Examples of correct xAxis assignment:
    "Bar chart of Expression values per Gene"
      → xAxis: ["Expression"]   (numeric)   Gene is the sample (categorical) — not in xAxis
    "Violin plot of Score grouped by Treatment"
      → xAxis: ["Score"]        (numeric)   Treatment → groupingFactors, not xAxis
    "Heatmap of gene expression"
      → xAxis: ["Gene"]         (in heatmaps, Gene/variable names ARE the xAxis — exception)
    "Bar chart of Q1, Q2, Q3, Q4 revenue"
      → xAxis: ["Q1","Q2","Q3","Q4"]   (numeric columns — multiple values allowed)
    "Line chart of Sales over Month"
      → xAxis: ["Sales"]        (numeric)   Month is the sample axis (smpTitle: "Month")

  Rules:
  - xAxis must contain NUMERIC data column name(s). If the column is categorical
    (gene names, patient IDs, drug names, group labels), it belongs in the samples
    dimension, not xAxis — omit xAxis and let CanvasXpress auto-assign.
  - Exception: Heatmap — the variable names (genes, features) go in xAxis because
    that is how CanvasXpress structures heatmap data.
  - If no numeric column name is identifiable from the description, omit xAxis
    entirely (CanvasXpress will auto-assign from the data).
  - Multiple numeric columns on the same axis are allowed:
    "xAxis": ["Q1", "Q2", "Q3", "Q4"]
  - To label the categorical/sample axis: use smpTitle (NOT yAxisTitle).
  - NEVER use yAxisTitle, yAxisTextColor, yAxisTitleColor, yAxisLog,
    yAxisMinValue, yAxisMaxValue, yAxisTextFontStyle, yAxisTitleFontStyle.

COMBINED (xAxis + xAxis2 — NEVER yAxis):
  AreaLine, BarLine, DotLine, Pareto, StackedLine, StackedPercentLine.
  Rules:
  - xAxis for the primary numeric series, xAxis2 for the secondary numeric series.
  - If ambiguous: first numeric column → xAxis, second numeric column → xAxis2.
  - NEVER yAxis. Use smpTitle to label the categorical sample axis, never yAxisTitle.

MULTI-DIMENSIONAL (both xAxis AND yAxis required):
  Scatter2D, Scatter3D, ScatterBubble2D, Volcano, Spaghetti,
  Contour, Streamgraph, Bump, KaplanMeier, TimeSeries.
  Rules:
  - MUST include BOTH xAxis and yAxis. Always list xAxis before yAxis.
  - Scatter3D and ScatterBubble2D also require zAxis.
  - If no column names given, omit axis params (CanvasXpress auto-assigns).
  - Use xAxisTitle and yAxisTitle for axis labels (NOT smpTitle).

## STEP 3 — SET REQUIRED GRAPH-TYPE-SPECIFIC PARAMETERS
These are mandatory when the graph type is selected:
  Area:      areaType: "overlapping" | "stacked" | "percent"  (REQUIRED)
  Density:   densityPosition: "normal" | "stacked" | "filled"  (REQUIRED)
  Histogram: histogramType: "dodged" | "staggered" | "stacked"  (REQUIRED)
  Dumbbell:  dumbbellType: "arrow" | "bullet" | "cleveland" | "connected" | "line" | "lineConnected" | "stacked"
  Ridgeline: use ridgeBy (column name) instead of groupingFactors
  Spaghetti, TagCloud, WordCloud: must include colorBy
  KaplanMeier: xAxis = time column, yAxis = event/status column (0/1), use colorBy for grouping (treatment arms, etc.)

## STEP 4 — ASSIGN DATA COLUMNS TO PARAMETERS
Using column names from the description or provided headers:
  groupingFactors : factor/categorical columns for grouping and colouring (1D charts)
  colorBy         : column whose values determine colour (scatter, spaghetti)
  shapeBy         : column for point shapes
  sizeBy          : column for point size (ScatterBubble2D)
  ellipseBy       : column to draw confidence ellipses around groups (Scatter2D and Scatter3D only)
  segregateSamplesBy / segregateVariablesBy : columns for faceting into sub-plots
  smpOverlays / varOverlays : metadata columns for heatmap annotation tracks
  ridgeBy         : column for Ridgeline groups (NOT groupingFactors)
  sankeyAxes      : ordered list of flow columns (Sankey, Alluvial, Ribbon)
  hierarchy       : ordered list of hierarchy columns (Bubble, Tree, Sunburst)

## STEP 5 — APPLY DATA TRANSFORMS (if requested)
  transformData   : "log2" | "log10" | "-log2" | "-log10" | "zscore" | "percentile" | "sqrt"
  xAxisTransform  : "log2" | "log10" | "-log2" | "-log10" | "sqrt" | "percentile"
  yAxisTransform  : same options (only for multi-dimensional graphs)
  filterData FORMAT (use when description says "filter", "only show", "where", "limit to")
    filterData is an array of filter rule arrays. Each rule: ["guess", "columnName", "operator", "value"]
    operators: "like" (equals / contains), "different" (not equals)
    "guess" is always the literal string "guess" as the first element.
    Examples: [["guess", "Treatment", "like", "Control"]], [["guess", "Stage", "different", "IV"]],
    Multiple filters (AND logic — all must pass):
      "filterData": [["guess", "Treatment", "like", "Drug A"], ["guess", "Responder", "like", "Yes"]]
  sortData FORMAT (use when description says "sort", "order by", "ranked by", "ascending", "descending")
    sortData is an array of sort rule arrays. Each rule: ["sortType", "axis", "columnName"]
    sortType: "var" (sort variables/rows), "smp" (sort samples/columns), "cat" (sort by category)
    axis: "var" or "smp"
    columnName: the column to sort by
    Examples: [["var", "var", "Expression"]], [["smp", "smp", "Treatment"]]
    Multiple filters and sorts are allowed — apply in the order given.
      Never use sortData for: Bin, Binplot, CDF, Contour, Density, Hex, Hexplot,
        Histogram, KaplanMeier, QQ, Quantile, Ridgeline, Scatter2D, ScatterBubble2D, Streamgraph.
        For simple bar chart sorting use sortDir: "ascending" or "descending" instead.

## STEP 6 — ADD DECORATIONS (if requested)
decorations is an array of objects. Each object requires "type" and "color".
  types: "line" | "point" | "text"
CRITICAL — position key depends on graph category:
  1D graphs (Bar, Violin, Heatmap, Line, etc.):
    Use "value" (a number). NEVER "x" or "y".
    {"type": "line",  "value": 2.0,  "color": "#e74c3c", "width": 1, "label": "Threshold"}
    {"type": "point", "value": 8.5,  "color": "#e67e22", "label": "Marker"}
    {"type": "text",  "value": 100,  "color": "#2c3e50", "label": "Key event"}
  Multi-dim (Scatter2D, Volcano, etc.):
    Use "x" for vertical lines, "y" for horizontal lines, both for points/text.
    NEVER use "value" for multi-dimensional graphs.
    {"type": "line", "x":  2.0, "color": "#e74c3c", "width": 1, "label": "FC +2"}
    {"type": "line", "y":  1.3, "color": "#7f8c8d", "width": 1, "label": "p=0.05"}
    {"type": "point","x": 1.5, "y": 4.2, "color": "#e74c3c", "label": "Sample X"}
  Volcano standard: two vertical lines (x = ±threshold) + one horizontal line (y = significance)

## STEP 7 — SET VISUAL STYLING (if mentioned)
  colorScheme (use exactly one of):
    YlGn, YlGnBu, GnBu, BuGn, PuBuGn, PuBu, BuPu, RdPu, PuRd, OrRd, YlOrRd, YlOrBr,
    Purples, Blues, Greens, Oranges, Reds, Greys, PuOr, BrBG, PRGn, PiYG, RdBu, RdGy,
    RdYlBu, Spectral, RdYlGn, Bootstrap, Economist, Excel, GGPlot, Solarized, PaulTol,
    ColorBlind, Tableau, WallStreetJournal, Stata, BlackAndWhite, CanvasXpress
  theme (use exactly one of):
    bw, classic, cx, dark, economist, excel, ggblanket, ggplot, gray, grey,
    highcharts, igray, light, linedraw, minimal, none, ptol, solarized, stata, tableau, void0, wsj
  Other styling: title, showLegend, legendPosition, graphOrientation,
    xAxisTitle, yAxisTitle (multi-dim only), smpTitle (1D/combined only),
    setMinX, setMaxX, setMinY (multi-dim only), setMaxY (multi-dim only),
    background, dataPointSize, samplesClustered, variablesClustered, heatmapIndicator

## STEP 8 — PARAMETER DISCIPLINE (final check before output)
Only use parameter names that are known CanvasXpress parameters.
NEVER invent parameter names. If unsure whether a parameter exists, omit it.
Examples of hallucinated names to NEVER use:
  showRegressionEllipse, showEllipse, ellipseShow, showGroupEllipses — use ellipseBy instead.
  yAxisTitle on 1D charts — use smpTitle instead.
  yAxis on single-dimensional or combined charts — never valid.

## STEP 9 — VALIDATE
Ensure graphType and all required axis parameters are present.
Return empty string if the config cannot be made valid.
"""


def build_system_prompt(
    description: str,
    headers: list[str] | None,
    data: list[list] | None,
) -> tuple[str, int, Optional[str]]:
    """
    Build a graph-type-aware, tiered system prompt from the knowledge DB.
    Returns (prompt_string, tier_used, detected_graph_type).
    """
    tier       = detect_tier(description, headers, data)
    graph_type = detect_graph_type(description)
    prompt     = _SYSTEM_PROMPT_HEADER

    # Inject live parameter+valid-values snippet from cx_knowledge
    param_snippet = cx_knowledge.get_param_snippet(graph_type=graph_type)
    if param_snippet:
        prompt += "\n" + param_snippet

    return prompt, tier, graph_type


# Warm the cx_knowledge schema cache
cx_knowledge.warm_cache()

# Alias for legacy references
SYSTEM_PROMPT = _SYSTEM_PROMPT_HEADER
# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_config(
    description: str,
    headers: list[str] | None = None,
    column_types: dict[str, str] | None = None,
    temperature: float = 0.0,
) -> tuple[dict, list[str]]:
    import time

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

    # Keep reference to raw data for tier detection in build_system_prompt
    data_ref = None  # set below if caller passes data through headers

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
        print(f"  System prompt : (tiered — see TIERED PROMPT step)", file=sys.stderr)
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
        print(f"  Provider : {PROVIDER}", file=sys.stderr)
        print(f"  Model    : {MODEL}", file=sys.stderr)
        print(f"  Calling Anthropic API...", file=sys.stderr)

    # Build tiered, graph-type-aware system prompt from knowledge DB
    system_prompt, tier, graph_type = build_system_prompt(description, headers, data_ref)
    if DEBUG:
        bar = "─" * 64
        print("", file=sys.stderr)
        print(bar, file=sys.stderr)
        print(bar, file=sys.stderr)
        tier_labels = ["", "base only", "base+schema+data", "base+schema+data+contradictions"]
        print(f"  Tier      : {tier} ({tier_labels[tier]})", file=sys.stderr)
        print(f"  GraphType : {graph_type or 'not detected'}", file=sys.stderr)
        print(f"  Size      : {len(system_prompt)} chars (~{len(system_prompt)//4} tokens)", file=sys.stderr)

    t1 = time.perf_counter()
    raw_text, usage = llm_complete(
        system=system_prompt,
        user=prompt,
        temperature=temperature,
        max_tokens=1500,
    )
    t_llm = (time.perf_counter() - t1) * 1000

    if DEBUG:
        print(f"  Latency       : {t_llm:.0f}ms", file=sys.stderr)
        print(f"  Input tokens  : {usage.get('input_tokens', '?')}", file=sys.stderr)
        print(f"  Output tokens : {usage.get('output_tokens', '?')}", file=sys.stderr)
        print(f"  Stop reason   : {usage.get('stop_reason', '?')}", file=sys.stderr)

    # ── Step 4: Parse response ───────────────────────────────────────────────
    raw = raw_text.strip()

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

    # Strip any hallucinated parameter names not present in the known schema
    config, removed_keys = cx_knowledge.filter_unknown_params(config)
    if removed_keys and DEBUG:
        bar = "─" * 64
        print(f"\n{bar}\n  STEP 4b — PARAM FILTER\n{bar}", file=sys.stderr)
        print(f"  Removed unknown params: {removed_keys}", file=sys.stderr)

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

    return config, removed_keys


# ---------------------------------------------------------------------------
# Config modification
# ---------------------------------------------------------------------------

def modify_config(
    config: dict,
    instruction: str,
    headers: list[str] | None = None,
    column_types: dict[str, str] | None = None,
    temperature: float = 0.0,
) -> tuple[dict, list[str]]:
    """
    Apply a plain-English modification instruction to an existing CanvasXpress config.
    Returns the complete modified config as a dict.
    """
    import time

    # ── Retrieve relevant examples using the instruction as query ─────────────
    t0 = time.perf_counter()
    examples = retrieve_examples(instruction)
    t_retrieval = (time.perf_counter() - t0) * 1000

    if DEBUG:
        bar = "─" * 64
        print(f"\n{bar}\n  MODIFY — STEP 1 RETRIEVAL\n{bar}", file=sys.stderr)
        print(f"  Instruction : {instruction}", file=sys.stderr)
        print(f"  Results     : {len(examples)} examples in {t_retrieval:.1f}ms", file=sys.stderr)

    ex_parts = []
    for ex in examples:
        ex_parts.append('Description: "' + ex["description"] + '"\nConfig: ' + json.dumps(ex["config"]))
    ex_text = "\n\n".join(ex_parts)

    # ── Build header hint ─────────────────────────────────────────────────────
    header_hint = ""
    if headers:
        if column_types:
            col_desc = ", ".join(
                col + " (" + column_types.get(col, "unknown") + ")" for col in headers
            )
            header_hint = (
                "\n\nDataset columns with types: " + col_desc + "."
                "\n   - numeric : xAxis (scatter), yAxis, zAxis"
                "\n   - factor  : groupingFactors, colorBy, shapeBy"
                "\n   - string  : xAxis labels, smpOverlays"
                "\n   - date    : xAxis in time series"
                "\n   Only assign columns matching their type."
            )
        else:
            header_hint = (
                "\n\nDataset columns: " + ", ".join(headers) + ". "
                "Use them for xAxis, yAxis, groupingFactors, colorBy etc. as appropriate."
            )

    # ── Build the modification prompt ─────────────────────────────────────────
    config_json = json.dumps(config, indent=2)
    prompt = (
        "Similar CanvasXpress examples for reference:\n\n" + ex_text + "\n\n"
        "---\n"
        "EXISTING CONFIG (preserve all parameters unless the instruction explicitly removes them):\n"
        + config_json + "\n\n"
        "MODIFICATION INSTRUCTION:\n\"" + instruction + "\"" + header_hint + "\n\n"
        "Apply the instruction to the existing config. "
        "Return ONLY the complete modified JSON config object — no explanation, no markdown fences."
    )

    if DEBUG:
        bar = "─" * 64
        print(f"\n{bar}\n  MODIFY — STEP 2 PROMPT\n{bar}", file=sys.stderr)
        print(f"  Existing keys : {list(config.keys())}", file=sys.stderr)
        print(f"  Instruction   : {instruction}", file=sys.stderr)
        print(f"  Prompt length : {len(prompt)} chars", file=sys.stderr)

    # ── Build system prompt (tiered, reusing existing logic) ──────────────────
    system_prompt, tier, detected_gt = build_system_prompt(instruction, headers, None)

    modify_preamble = (
        "You are a CanvasXpress configuration editor. "
        "You will receive an EXISTING config and a plain-English instruction describing a modification. "
        "Your job is to apply that modification and return the COMPLETE updated config.\n"
        "Rules:\n"
        "- Keep ALL existing parameters unless the instruction explicitly says to remove one.\n"
        "- Add new parameters or change existing values as instructed.\n"
        "- Never remove graphType, xAxis, or other required parameters unless explicitly told to.\n"
        "- Return ONLY the JSON object. No markdown, no explanation.\n\n"
    )
    system_prompt = modify_preamble + system_prompt

    if DEBUG:
        bar = "─" * 64
        print(f"\n{bar}\n  MODIFY — TIERED PROMPT\n{bar}", file=sys.stderr)
        tier_labels = ["", "base only", "base+schema+data", "base+schema+data+contradictions"]
        print(f"  Tier      : {tier} ({tier_labels[tier]})", file=sys.stderr)
        print(f"  GraphType : {detected_gt or 'not detected'}", file=sys.stderr)
        print(f"  Size      : {len(system_prompt)} chars", file=sys.stderr)

    # ── LLM call ──────────────────────────────────────────────────────────────
    if DEBUG:
        bar = "─" * 64
        print(f"\n{bar}\n  MODIFY — STEP 3 LLM CALL\n{bar}", file=sys.stderr)
        print(f"  Provider : {PROVIDER}", file=sys.stderr)
        print(f"  Model    : {MODEL}", file=sys.stderr)

    t1 = time.perf_counter()
    raw_text, usage = llm_complete(
        system=system_prompt,
        user=prompt,
        temperature=temperature,
        max_tokens=1500,
    )
    t_llm = (time.perf_counter() - t1) * 1000

    if DEBUG:
        print(f"  Latency       : {t_llm:.0f}ms", file=sys.stderr)
        print(f"  Input tokens  : {usage.get('input_tokens', '?')}", file=sys.stderr)
        print(f"  Output tokens : {usage.get('output_tokens', '?')}", file=sys.stderr)

    # ── Parse response ─────────────────────────────────────────────────────────
    raw = raw_text.strip()

    if DEBUG:
        bar = "─" * 64
        print(f"\n{bar}\n  MODIFY — STEP 4 RAW RESPONSE\n{bar}", file=sys.stderr)
        print(f"  {raw[:400]}", file=sys.stderr)

    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0].strip()

    if not raw or raw in ("''", '""'):
        if DEBUG:
            print("\n  ⚠️  Model returned empty — returning original config unchanged", file=sys.stderr)
        return config

    modified = json.loads(raw)

    # Strip hallucinated parameter names
    modified, removed_keys = cx_knowledge.filter_unknown_params(modified)
    if removed_keys and DEBUG:
        bar = "─" * 64
        print(f"\n{bar}\n  MODIFY — PARAM FILTER\n{bar}", file=sys.stderr)
        print(f"  Removed unknown params: {removed_keys}", file=sys.stderr)

    if DEBUG:
        bar = "─" * 64
        print(f"\n{bar}\n  MODIFY — STEP 5 RESULT\n{bar}", file=sys.stderr)
        added   = [k for k in modified if k not in config]
        removed = [k for k in config   if k not in modified]
        changed = [k for k in config   if k in modified and config[k] != modified[k]]
        print(f"  Keys added   : {added   or 'none'}", file=sys.stderr)
        print(f"  Keys removed : {removed or 'none'}", file=sys.stderr)
        print(f"  Keys changed : {changed or 'none'}", file=sys.stderr)
        print(f"  Retrieval : {t_retrieval:.1f}ms   LLM : {t_llm:.0f}ms", file=sys.stderr)

    return modified, removed_keys


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

    # Value-level validation via cx_knowledge schema
    value_check = cx_knowledge.validate_param_values(config)
    for w in value_check["warnings"]:
        if w not in warnings:
            warnings.append(w)

    return {
        "valid":          len(warnings) == 0,
        "warnings":       warnings,
        "invalid_refs":   invalid_refs,
        "invalid_values": value_check["invalid_values"],
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

# Build the CORS ASGI middleware list — passed to mcp.run() below.
# CORS_ORIGINS is read from .env (CORS_ORIGINS env var).
_cors_middleware: list = [
    Middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-API-Key"],
    )
]


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
    config, removed_params = generate_config(description, resolved_headers, column_types, temperature)
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
        "config":         config,
        "valid":          validation["valid"],
        "warnings":       validation["warnings"],
        "invalid_refs":   validation["invalid_refs"],
        "headers_used":   resolved_headers or [],
        "types_used":     column_types or {},
        "removed_params": removed_params,
    }
# Note: tier info is logged in debug mode but not returned to keep response lean




@mcp.tool(
    description=(
        "Modify an existing CanvasXpress config using a plain English instruction. "
        "Pass in your current config and describe what you want to change — add parameters, "
        "remove parameters, change values, switch color scheme, update axis titles, etc. "
        "The full existing config is preserved except for the changes you request. "
        "Examples: 'add a title My Chart', 'change the color scheme to Tableau', "
        "'remove the legend', 'set the x-axis title to Fold Change', "
        "'add groupingFactors for the Treatment column', 'switch to dark theme'. "
        "Returns the complete modified config ready to pass to new CanvasXpress()."
    )
)
def modify_canvasxpress_config(
    config: dict,
    instruction: str,
    headers: list[str] | None = None,
    data: list[list] | None = None,
    column_types: dict[str, str] | None = None,
    temperature: float = 0.0,
) -> dict:
    """
    Args:
        config:       The existing CanvasXpress JSON config to modify.
                      e.g. {"graphType": "Heatmap", "xAxis": ["Gene"], "colorScheme": "RdBu"}
        instruction:  Plain English description of the modification to apply.
                      e.g. "add a title", "change colorScheme to Tableau", "remove the legend"
        headers:      Optional column names — used to validate any new column references
                      introduced by the instruction.
        data:         Optional CSV-style data array (first row = headers). Overrides headers.
        column_types: Optional map of column name → type (string/numeric/factor/date).
        temperature:  LLM creativity 0.0–1.0 (default 0.0 = deterministic).

    Returns:
        Dict with keys:
          config        (dict) - the complete modified CanvasXpress JSON config
          valid         (bool) - True if all column refs exist in the provided columns
          warnings      (list) - column validation warnings (empty if valid)
          invalid_refs  (dict) - map of config key → missing column names
          headers_used  (list) - column names used for validation
          types_used    (dict) - column types passed in (if provided)
          changes       (dict) - summary of keys added, removed, and changed
    """
    if not config:
        return {
            "config": {},
            "valid": False,
            "warnings": ["config parameter is empty — nothing to modify."],
            "invalid_refs": {},
            "headers_used": [],
            "types_used":   {},
            "changes":      {},
        }

    # ── Resolve headers ──────────────────────────────────────────────────────
    resolved_headers: list[str] | None = None
    if data is not None:
        try:
            resolved_headers = extract_headers_from_data(data)
            log.info("Extracted %d headers from data array", len(resolved_headers))
        except ValueError as e:
            return {
                "config": config,
                "valid": False,
                "warnings": [str(e)],
                "invalid_refs": {},
                "headers_used": [],
                "types_used":   {},
                "changes":      {},
            }
    elif headers is not None:
        resolved_headers = headers

    # Validate column_types
    VALID_TYPES = {"string", "numeric", "factor", "date"}
    if column_types:
        bad = {k: v for k, v in column_types.items() if v not in VALID_TYPES}
        if bad:
            log.warning("Unknown column types ignored: %s", bad)
            column_types = {k: v for k, v in column_types.items() if v in VALID_TYPES}

    if DEBUG:
        bar = "─" * 64
        print(f"\n{bar}\n  MODIFY REQUEST\n{bar}", file=sys.stderr)
        print(f"  Instruction   : {instruction}", file=sys.stderr)
        print(f"  Existing keys : {list(config.keys())}", file=sys.stderr)
        if resolved_headers:
            print(f"  Headers       : {resolved_headers}", file=sys.stderr)

    log.info("Modifying config — instruction: %s", instruction)
    modified, removed_params = modify_config(config, instruction, resolved_headers, column_types, temperature)

    # ── Build change summary ─────────────────────────────────────────────────
    changes = {
        "added":   [k for k in modified if k not in config],
        "removed": [k for k in config   if k not in modified],
        "changed": [k for k in config   if k in modified and config[k] != modified[k]],
    }
    log.info(
        "Modification complete — added: %s  removed: %s  changed: %s",
        changes["added"], changes["removed"], changes["changed"],
    )

    # ── Validate column references ───────────────────────────────────────────
    if resolved_headers and modified:
        validation = validate_config_headers(modified, resolved_headers)
        if DEBUG:
            bar = "─" * 64
            print(f"\n{bar}\n  MODIFY — HEADER VALIDATION\n{bar}", file=sys.stderr)
            print(f"  Valid : {validation['valid']}", file=sys.stderr)
            if validation["warnings"]:
                for w in validation["warnings"]:
                    print(f"  ⚠️  {w}", file=sys.stderr)
    else:
        validation = {"valid": True, "warnings": [], "invalid_refs": {}}

    return {
        "config":         modified,
        "valid":          validation["valid"],
        "warnings":       validation["warnings"],
        "invalid_refs":   validation["invalid_refs"],
        "headers_used":   resolved_headers or [],
        "types_used":     column_types or {},
        "changes":        changes,
        "removed_params": removed_params,
    }



@mcp.tool(
    description=(
        "Generate, validate, and annotate Kaplan-Meier survival plot configs for CanvasXpress. "
        "Accepts any combination of: a plain English description, column headers or a full data "
        "array, and/or an existing config to validate and fix. "
        "Automatically detects which columns are time, event, and grouping from the dataset. "
        "Computes median survival and log-rank p-value from data and embeds them as decorations. "
        "Examples: description='OS curve by treatment arm' with headers=['PatientID','OS_Time','OS_Status','Treatment']; "
        "or config={...} to validate an existing KM config; "
        "or data=[['ID','Time','Event','Arm'],[...]] to generate with annotations."
    )
)
def generate_km_config(
    description:     str | None = None,
    headers:         list[str] | None = None,
    data:            list[list] | None = None,
    config:          dict | None = None,
    add_annotations: bool = True,
    temperature:     float = 0.0,
) -> dict:
    """
    Args:
        description:     Plain English description of the KM plot you want.
                         e.g. "Overall survival by treatment arm with 95% CI"
        headers:         Column names from your dataset.
                         e.g. ["PatientID", "OS_Time", "OS_Status", "Treatment"]
        data:            Full data array — first row must be column headers.
                         e.g. [["ID","Time","Event","Arm"],["P1",24,1,"A"],...]
                         When provided, headers are extracted automatically and
                         statistics + decorations are computed from the data.
        config:          An existing KM config to validate, fix, and/or enrich.
                         e.g. {"graphType":"KaplanMeier","xAxis":["OS_Time"],...}
        add_annotations: Whether to compute median survival and log-rank p-value
                         from data and add them as decorations (default True).
        temperature:     LLM creativity 0.0–1.0 (default 0.0 = deterministic).

    Returns:
        Dict with keys:
          config            (dict)       - the CanvasXpress KM JSON config
          valid             (bool)       - True if config passes all KM validation rules
          errors            (list)       - must-fix issues (e.g. missing xAxis)
          warnings          (list)       - should-fix issues and notes
          suggestions       (list)       - optional improvements
          column_detection  (dict|None)  - detected time/event/group columns + confidence
          statistics        (dict|None)  - per-group n, n_events, median survival + log-rank p
          decorations_added (bool)       - whether median/p-value decorations were added
    """
    log.info(
        "KM skill: description=%s headers=%s data_rows=%s config_keys=%s",
        bool(description), bool(headers),
        len(data) - 1 if data else 0,
        list(config.keys()) if config else None,
    )

    if not any([description, headers, data, config]):
        return {
            "config":            {"graphType": "KaplanMeier"},
            "valid":             False,
            "errors":            ["At least one of description, headers, data, or config must be provided."],
            "warnings":          [],
            "suggestions":       ["Pass headers or data so columns can be detected automatically."],
            "column_detection":  None,
            "statistics":        None,
            "decorations_added": False,
        }

    return cx_survival.handle_generate_km(
        description     = description,
        headers         = headers,
        data            = data,
        config          = config,
        add_annotations = add_annotations,
        temperature     = temperature,
        llm_complete_fn = llm_complete,
    )


@mcp.tool(
    description=(
        "Query the CanvasXpress parameter knowledge base. "
        "Fetch parameters, their valid values, and descriptions from the "
        "canvasxpress-LLM schema — sourced live from GitHub with local cache fallback. "
        "Usage: pass graph_type to list all parameters for a chart type, "
        "param_name to look up a single parameter's valid values and description, "
        "or both to check whether a parameter applies to a specific chart type. "
        "Examples: graph_type='Heatmap', param_name='colorScheme', "
        "param_name='areaType' graph_type='Area'."
    )
)
def query_canvasxpress_params(
    graph_type: str | None = None,
    param_name: str | None = None,
    refresh: bool = False,
) -> dict:
    """
    Args:
        graph_type: CanvasXpress chart type e.g. "Heatmap", "Scatter2D", "Violin".
                    Returns all parameters that apply to this chart type.
        param_name: Parameter name e.g. "colorScheme", "areaType", "histogramType".
                    Returns full definition including valid values and applicable graph types.
        refresh:    If True, re-fetch SCHEMA.md from GitHub even if cache is fresh.

    Returns:
        Dict with:
          For a single param:  {found, param, description, type, valid_values, graph_types, schema_source}
          For a graph type:    {graph_type, param_count, params: {name: {description, type, valid_values}}, schema_source}
          For all params:      {param_count, params, schema_source, tip}
    """
    if refresh:
        cx_knowledge.load_schema(force=True)
        log.info("cx_knowledge schema refreshed on request")
    return cx_knowledge.handle_query_params(
        graph_type=graph_type,
        param_name=param_name,
    )

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
# REST helpers — parse query params or JSON body into tool kwargs
# ---------------------------------------------------------------------------

def _parse_col_types(raw: str) -> dict:
    """
    Accept either format:
      JSON object :  '{"Gene":"string","Sample1":"numeric"}'
      name=type   :  'Gene=string, Sample1=numeric, Treatment=factor'  (CanvasXpress JS format)
    Returns {} for blank input.
    """
    raw = raw.strip()
    if not raw:
        return {}
    if raw.startswith("{"):
        return json.loads(raw)
    result = {}
    for item in raw.split(","):
        item = item.strip()
        if "=" in item:
            k, _, v = item.partition("=")
            k, v = k.strip(), v.strip()
            if k and v:
                result[k] = v
    return result


async def _kwargs_from_request(request: Request, require_description: bool = True) -> tuple[dict, int, str]:
    """Extract generate/modify kwargs from a GET query string or POST JSON/form body.
    Also returns any CanvasXpress pass-through params (target, client_id) in the dict
    under the key '_cx' so callers can include them in JSONP responses.
    """
    if request.method == "GET":
        p = dict(request.query_params)
    else:
        ct = request.headers.get("content-type", "")
        if "application/json" in ct:
            p = await request.json()
        else:
            form = await request.form()
            p = dict(form)

    kwargs: dict = {}

    # description / prompt (aliases)
    desc = p.get("description") or p.get("prompt") or p.get("q") or ""
    desc = desc.strip()
    if require_description and not desc:
        return {}, 400, "'description' (or 'prompt') is required"
    if desc:
        kwargs["description"] = desc

    # instruction (modify only)
    if "instruction" in p:
        kwargs["instruction"] = p["instruction"]

    # config (modify only) — JSON string or object
    if "config" in p:
        v = p["config"]
        kwargs["config"] = json.loads(v) if isinstance(v, str) else v

    # headers — comma-separated string or JSON array; skip if empty
    if p.get("headers", "").strip():
        v = p["headers"].strip()
        kwargs["headers"] = json.loads(v) if v.startswith("[") else [h.strip() for h in v.split(",") if h.strip()]

    # data — JSON array of arrays
    if "data" in p:
        v = p["data"]
        kwargs["data"] = json.loads(v) if isinstance(v, str) else v

    # column_types — JSON object or "Col=type, Col2=type2"; skip if empty
    for key in ("column_types", "types"):
        if p.get(key, "").strip():
            parsed = _parse_col_types(p[key])
            if parsed:
                kwargs["column_types"] = parsed
            break

    # temperature
    if "temperature" in p:
        try:
            kwargs["temperature"] = float(p["temperature"])
        except (ValueError, TypeError):
            pass  # ignore bad values, use default

    # CanvasXpress pass-through params — stored under _cx, not forwarded to tools
    cx: dict = {}
    if p.get("target",    "").strip(): cx["target"]    = p["target"].strip()
    if p.get("client_id", "").strip(): cx["client"]    = p["client_id"].strip()
    if p.get("callback",  "").strip(): cx["callback"]  = p["callback"].strip()
    if desc:                           cx["prompt"]     = desc
    kwargs["_cx"] = cx

    return kwargs, 200, ""


# ---------------------------------------------------------------------------
# REST endpoints — /generate  /modify  /ui
# ---------------------------------------------------------------------------

_UI_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CanvasXpress MCP — Web UI</title>
<style>
  *{box-sizing:border-box}
  body{font-family:system-ui,sans-serif;max-width:860px;margin:40px auto;padding:0 20px;background:#f5f5f5;color:#222}
  h1{font-size:1.4rem;margin-bottom:4px}
  h1 span{color:#c0392b}
  .subtitle{color:#666;font-size:.85rem;margin-bottom:24px}
  label{display:block;font-weight:600;font-size:.85rem;margin-top:14px;margin-bottom:3px}
  label span{font-weight:400;color:#888}
  input[type=text],textarea,select{width:100%;padding:7px 10px;border:1px solid #ccc;border-radius:5px;font-size:.9rem;font-family:inherit;background:#fff}
  textarea{resize:vertical;min-height:80px}
  .row{display:grid;grid-template-columns:1fr 1fr;gap:12px}
  .actions{margin-top:18px;display:flex;align-items:center;gap:10px;flex-wrap:wrap}
  button{padding:8px 18px;border:none;border-radius:5px;cursor:pointer;font-size:.9rem;font-weight:600}
  .btn-primary{background:#c0392b;color:#fff}
  .btn-secondary{background:#555;color:#fff}
  #url-box{flex:1;min-width:200px;font-size:.78rem;color:#555;background:#fff;border:1px solid #ccc;border-radius:5px;padding:7px 10px;word-break:break-all;cursor:text;white-space:pre-wrap}
  #result{margin-top:24px;background:#fff;border:1px solid #ddd;border-radius:6px;padding:16px;display:none}
  #result h3{margin:0 0 10px;font-size:.95rem}
  pre{margin:0;font-size:.82rem;overflow:auto;max-height:460px;background:#f8f8f8;padding:10px;border-radius:4px}
  .badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:.75rem;font-weight:700;margin-left:6px}
  .valid{background:#d4edda;color:#155724}
  .invalid{background:#f8d7da;color:#721c24}
  .warn{background:#fff3cd;color:#856404}
  .meta{font-size:.8rem;color:#666;margin-bottom:8px}
  .tab-bar{display:flex;gap:4px;margin-bottom:16px}
  .tab{padding:7px 16px;border-radius:5px 5px 0 0;border:1px solid #ccc;border-bottom:none;cursor:pointer;font-weight:600;font-size:.85rem;background:#eee}
  .tab.active{background:#fff;border-bottom:1px solid #fff;margin-bottom:-1px}
  .panel{display:none}
  .panel.active{display:block}
  .card{background:#fff;border:1px solid #ddd;border-radius:6px;padding:16px}
</style>
</head>
<body>
<h1>CanvasXpress <span>MCP</span> — Web UI</h1>
<p class="subtitle">Generate or modify CanvasXpress configs. Parameters are encoded in the URL — bookmark or share it.</p>

<div class="tab-bar">
  <div class="tab active" data-tab="generate">Generate</div>
  <div class="tab" data-tab="modify">Modify</div>
</div>

<!-- ── GENERATE panel ─────────────────────────────────────────────── -->
<div class="panel card active" id="panel-generate">
  <label>Description / Prompt <span>(required)</span>
    <input type="text" id="g-desc" placeholder="e.g. Clustered heatmap with RdBu colors and dendrograms on both axes">
  </label>
  <div class="row">
    <div>
      <label>Headers <span>comma-separated</span>
        <input type="text" id="g-headers" placeholder="Gene, Sample1, Sample2, Treatment">
      </label>
    </div>
    <div>
      <label>Column types <span>Col=type,… or JSON</span>
        <input type="text" id="g-types" placeholder='Gene=string,Sample1=numeric,Treatment=factor'>
      </label>
    </div>
  </div>
  <label>Data <span>JSON array of arrays — first row is headers</span>
    <textarea id="g-data" placeholder='[["Gene","S1","Treatment"],["BRCA1",1.2,"Control"]]'></textarea>
  </label>
  <label>Temperature <span>0 = deterministic</span>
    <input type="text" id="g-temp" value="0" style="width:80px">
  </label>
</div>

<!-- ── MODIFY panel ───────────────────────────────────────────────── -->
<div class="panel card" id="panel-modify">
  <label>Existing config <span>JSON object — required</span>
    <textarea id="m-config" style="min-height:120px" placeholder='{"graphType":"Bar","xAxis":["Gene"]}'></textarea>
  </label>
  <label>Instruction <span>plain English modification — required</span>
    <input type="text" id="m-instr" placeholder="add a title My Chart and change colorScheme to Tableau">
  </label>
  <div class="row">
    <div>
      <label>Headers <span>optional</span>
        <input type="text" id="m-headers" placeholder="Gene, Sample1, Treatment">
      </label>
    </div>
    <div>
      <label>Column types <span>optional</span>
        <input type="text" id="m-types" placeholder="Gene=string,Sample1=numeric">
      </label>
    </div>
  </div>
  <label>Data <span>optional JSON array</span>
    <textarea id="m-data" placeholder='[["Gene","S1"],["BRCA1",1.2]]'></textarea>
  </label>
</div>

<div class="actions">
  <button class="btn-primary" onclick="submit()">&#9654; Run</button>
  <button class="btn-secondary" onclick="copyUrl()">&#128279; Copy URL</button>
  <div id="url-box">—</div>
</div>

<div id="result">
  <h3 id="result-title">Result</h3>
  <div class="meta" id="result-meta"></div>
  <pre id="result-pre"></pre>
</div>

<script>
const BASE = window.location.origin;
let activeTab = 'generate';

document.querySelectorAll('.tab').forEach(t => {
  t.addEventListener('click', () => {
    document.querySelectorAll('.tab, .panel').forEach(el => el.classList.remove('active'));
    t.classList.add('active');
    document.getElementById('panel-' + t.dataset.tab).classList.add('active');
    activeTab = t.dataset.tab;
    buildUrl();
  });
});

function v(id){ return document.getElementById(id).value.trim(); }

function buildUrl() {
  const p = new URLSearchParams();
  if (activeTab === 'generate') {
    const desc = v('g-desc'), hdrs = v('g-headers'), types = v('g-types'),
          data = v('g-data'), temp = v('g-temp');
    if (desc)  p.set('description', desc);
    if (hdrs)  p.set('headers', hdrs);
    if (types) p.set('column_types', types);
    if (data)  p.set('data', data);
    if (temp && temp !== '0') p.set('temperature', temp);
    const url = BASE + '/generate?' + p.toString();
    document.getElementById('url-box').textContent = url;
    return url;
  } else {
    const cfg = v('m-config'), instr = v('m-instr'), hdrs = v('m-headers'),
          types = v('m-types'), data = v('m-data');
    if (cfg)   p.set('config', cfg);
    if (instr) p.set('instruction', instr);
    if (hdrs)  p.set('headers', hdrs);
    if (types) p.set('column_types', types);
    if (data)  p.set('data', data);
    const url = BASE + '/modify?' + p.toString();
    document.getElementById('url-box').textContent = url;
    return url;
  }
}

['g-desc','g-headers','g-types','g-data','g-temp',
 'm-config','m-instr','m-headers','m-types','m-data'].forEach(id => {
  document.getElementById(id).addEventListener('input', buildUrl);
});

function copyUrl() {
  const url = buildUrl();
  navigator.clipboard.writeText(url).catch(() => {});
}

async function submit() {
  const url = buildUrl();
  const resultEl = document.getElementById('result');
  const preEl    = document.getElementById('result-pre');
  const metaEl   = document.getElementById('result-meta');
  const titleEl  = document.getElementById('result-title');
  resultEl.style.display = 'block';
  preEl.textContent = 'Loading…';
  metaEl.textContent = '';
  titleEl.innerHTML = activeTab === 'generate' ? 'Generated Config' : 'Modified Config';
  try {
    const resp = await fetch(url);
    const data = await resp.json();
    if (data.error) { preEl.textContent = 'Error: ' + data.error; return; }

    const cfg  = data.config  || {};
    const valid = data.valid;
    const warns = data.warnings || [];
    const badge = valid
      ? '<span class="badge valid">✓ valid</span>'
      : '<span class="badge invalid">✗ warnings</span>';
    const gt = cfg.graphType || '?';
    const hdr = (data.headers_used || []).join(', ') || '—';
    metaEl.innerHTML = `graphType: <b>${gt}</b>${badge} &nbsp; headers: ${hdr}` +
      (warns.length ? `<br><span class="badge warn">⚠ ${warns.join(' | ')}</span>` : '');
    if (activeTab === 'modify' && data.changes) {
      const c = data.changes;
      metaEl.innerHTML += `<br>added: ${(c.added||[]).join(', ')||'none'} &nbsp; ` +
        `removed: ${(c.removed||[]).join(', ')||'none'} &nbsp; ` +
        `changed: ${(c.changed||[]).join(', ')||'none'}`;
    }
    preEl.textContent = JSON.stringify(cfg, null, 2);
  } catch(e) {
    preEl.textContent = 'Request failed: ' + e;
  }
}

// Populate from URL params on load (so bookmarked URLs auto-fill the form)
(function restoreFromUrl() {
  const p = new URLSearchParams(window.location.search);
  const tab = p.get('_tab') || 'generate';
  if (tab === 'modify') {
    document.querySelector('[data-tab=modify]').click();
    if (p.get('config'))      document.getElementById('m-config').value = p.get('config');
    if (p.get('instruction')) document.getElementById('m-instr').value  = p.get('instruction');
    if (p.get('headers'))     document.getElementById('m-headers').value = p.get('headers');
    if (p.get('column_types'))document.getElementById('m-types').value  = p.get('column_types');
    if (p.get('data'))        document.getElementById('m-data').value   = p.get('data');
  } else {
    if (p.get('description')) document.getElementById('g-desc').value    = p.get('description');
    if (p.get('headers'))     document.getElementById('g-headers').value  = p.get('headers');
    if (p.get('column_types'))document.getElementById('g-types').value   = p.get('column_types');
    if (p.get('data'))        document.getElementById('g-data').value    = p.get('data');
    if (p.get('temperature')) document.getElementById('g-temp').value    = p.get('temperature');
  }
  buildUrl();
})();
</script>
</body>
</html>
"""


def _cx_response(result: dict, cx: dict, status: int = 200) -> Response:
    """
    Return either a JSONP or plain JSON response depending on whether the
    CanvasXpress 'callback' parameter was present in the request.

    JSONP format (used by CanvasXpress askLLM() script-tag injection):
        CanvasXpress.callbackLLM({...json...});
        Content-Type: application/javascript

    Plain JSON (used by fetch() / REST clients):
        {...json...}
        Content-Type: application/json

    Also enriches the result with the fields callbackLLM expects:
        success, prompt, datetime, target, client
    """
    import re
    from datetime import datetime, timezone

    # Enrich with CanvasXpress-expected fields
    result.setdefault("success", result.get("valid", True))
    if cx.get("prompt"):
        result.setdefault("prompt", cx["prompt"])
    if cx.get("target"):
        result["target"] = cx["target"]
    if cx.get("client"):
        result["client"] = cx["client"]
    result["datetime"] = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")

    callback = cx.get("callback", "").strip()
    if callback:
        # Sanitise callback name — allow only alphanumeric, dots, underscores
        safe_cb = re.sub(r"[^a-zA-Z0-9_.]", "", callback)
        body = f"{safe_cb}({json.dumps(result)});".encode("utf-8")
        return Response(
            content=body,
            status_code=status,
            media_type="application/javascript; charset=utf-8",
            headers={"Access-Control-Allow-Origin": "*"},
        )
    return JSONResponse(result, status_code=status)


@mcp.custom_route("/generate", methods=["GET", "POST"])
async def rest_generate(request: Request) -> Response:
    """
    REST / JSONP endpoint for generate_canvasxpress_config.

    JSON  (fetch):  GET /generate?description=Violin+plot&headers=Gene,Expr
    JSONP (script): GET /generate?callback=CanvasXpress.callbackLLM&target=myChart
                        &description=...&headers=...&column_types=Gene=string,Expr=numeric
                        &temperature=0&client_id=...

    Query / body parameters:
      description   (str, required) — plain English chart description. Alias: prompt, q.
      headers       (str)           — comma-separated column names, or JSON array.
      data          (str)           — JSON array of arrays (first row = header row).
      column_types  (str)           — "Col=type, …" or JSON object. Alias: types.
      temperature   (float)         — 0.0–1.0, default 0.
      callback      (str)           — JSONP callback (CanvasXpress.callbackLLM).
      target        (str)           — CanvasXpress chart target ID (passed through).
      client_id     (str)           — CanvasXpress client ID (passed through as 'client').
    """
    kwargs, status, err = await _kwargs_from_request(request, require_description=True)
    cx = kwargs.pop("_cx", {})
    if status != 200:
        return _cx_response({"error": err, "success": False}, cx, status)
    try:
        result = generate_canvasxpress_config(**kwargs)
    except Exception as exc:
        log.exception("REST /generate error")
        return _cx_response({"error": str(exc), "success": False}, cx, 500)
    return _cx_response(result, cx)


@mcp.custom_route("/modify", methods=["GET", "POST"])
async def rest_modify(request: Request) -> Response:
    """
    REST / JSONP endpoint for modify_canvasxpress_config.

    GET  /modify?config={"graphType":"Bar",...}&instruction=add+a+title
    POST /modify   (JSON body with same keys)

    Query / body parameters:
      config        (str|obj, required) — existing CanvasXpress JSON config.
      instruction   (str, required)     — plain English modification instruction.
      headers       (str)               — optional comma-separated column names.
      data          (str)               — optional JSON array of arrays.
      column_types  (str)               — optional "Col=type,…" or JSON object.
      temperature   (float)             — 0.0–1.0, default 0.
      callback      (str)               — JSONP callback name.
      target        (str)               — CanvasXpress chart target ID (passed through).
      client_id     (str)               — CanvasXpress client ID (passed through).
    """
    kwargs, status, err = await _kwargs_from_request(request, require_description=False)
    cx = kwargs.pop("_cx", {})
    if status != 200:
        return _cx_response({"error": err, "success": False}, cx, status)
    if "config" not in kwargs:
        return _cx_response({"error": "'config' is required", "success": False}, cx, 400)
    if "instruction" not in kwargs:
        return _cx_response({"error": "'instruction' is required", "success": False}, cx, 400)
    try:
        result = modify_canvasxpress_config(**kwargs)
    except Exception as exc:
        log.exception("REST /modify error")
        return _cx_response({"error": str(exc), "success": False}, cx, 500)
    return _cx_response(result, cx)


@mcp.custom_route("/ui", methods=["GET"])
async def rest_ui(request: Request) -> HTMLResponse:
    """Serve the browser-based form UI at /ui."""
    return HTMLResponse(_UI_HTML)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    info = provider_info()
    log.info(
        "Starting CanvasXpress MCP server on %s:%d  provider=%s  model=%s",
        HOST, PORT, info["provider"], info["model"],
    )
    log.info("MCP endpoint: http://%s:%d/mcp", HOST if HOST != "0.0.0.0" else "localhost", PORT)
    if DEBUG:
        log.info("Debug mode ON  — set CX_DEBUG=0 to disable")
        log.info("Retrieval : %s", "vector (sqlite-vec)" if _use_vector_index else "SequenceMatcher (fallback)")
        log.info("Examples  : %d loaded", len(EXAMPLES))
    mcp.run(transport="http", host=HOST, port=PORT, middleware=_cors_middleware)
