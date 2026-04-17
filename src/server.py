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
import re
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
import cx_selector
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

# Admin key — required for destructive endpoints (e.g. /feedback/purge).
# Set ADMIN_KEY in .env to a secret string.  If not set, a random key is
# generated at startup and printed once to the log at WARNING level.
_env_admin_key = os.environ.get("ADMIN_KEY", "").strip()
if _env_admin_key:
    ADMIN_KEY: str = _env_admin_key
else:
    import uuid as _uuid_init
    ADMIN_KEY = str(_uuid_init.uuid4())
    # Deferred log — logger not yet configured at import time; use print so
    # the key is always visible in the startup output.
    print(
        f"[cx-mcp] WARNING: ADMIN_KEY not set in .env — "
        f"using auto-generated key for this session: {ADMIN_KEY}",
        flush=True,
    )

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


SYSTEM_PROMPT = """You are an expert CanvasXpress data visualization assistant.
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
    prompt     = SYSTEM_PROMPT

    # Inject live parameter+valid-values snippet from cx_knowledge
    param_snippet = cx_knowledge.get_param_snippet(graph_type=graph_type)
    if param_snippet:
        prompt += "\n" + param_snippet

    return prompt, tier, graph_type


# Warm the cx_knowledge schema cache
cx_knowledge.warm_cache()

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
        return {}, []

    try:
        config = json.loads(raw)
    except json.JSONDecodeError as e:
        log.warning("LLM returned non-JSON response (%s). raw=%s", e, raw[:200])
        return {}, []

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
        return config, []

    try:
        modified = json.loads(raw)
    except json.JSONDecodeError as e:
        log.warning("LLM returned non-JSON in modify (%s). Returning original config.", e)
        return config, []

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
    "colorBy", "shapeBy", "sizeBy", "ellipseBy", "stackBy", "pivotBy",
    "ridgeBy", "splitSamplesBy", "splitVariablesBy",
    "hierarchy",
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

# ---------------------------------------------------------------------------
# Call logging + feedback (thumbs up/down)
# ---------------------------------------------------------------------------

CALL_LOG_DB = DATA_DIR / "call_log.db"

import sqlite3 as _sqlite3
import threading as _threading
import uuid as _uuid


class _CallLog:
    """
    Thread-safe SQLite logger for every tool call.

    Schema
    ──────
    tool_calls
      id          TEXT PRIMARY KEY   — UUID4
      tool        TEXT               — tool name (from _PATH_TO_TOOL)
      path        TEXT               — URL path
      request     TEXT               — JSON-serialised request body (query or POST)
      response    TEXT               — JSON-serialised response body (config + meta)
      status      INTEGER            — HTTP status code
      ts          TEXT               — ISO-8601 UTC timestamp
      rating      INTEGER NULL       — 1 = thumbs up, -1 = thumbs down
      comment     TEXT NULL          — optional free-text feedback
    """

    _lock = _threading.Lock()

    def __init__(self, db_path: Path):
        self._path = str(db_path)
        self._init_db()

    def _connect(self):
        con = _sqlite3.connect(self._path, check_same_thread=False)
        con.execute("PRAGMA journal_mode=WAL")
        return con

    def _init_db(self):
        with self._lock:
            con = self._connect()
            con.execute("""
                CREATE TABLE IF NOT EXISTS tool_calls (
                    id       TEXT PRIMARY KEY,
                    tool     TEXT,
                    path     TEXT,
                    request  TEXT,
                    response TEXT,
                    status   INTEGER,
                    ts       TEXT,
                    rating   INTEGER,
                    comment  TEXT
                )
            """)
            con.commit()
            con.close()

    def log(
        self,
        call_id: str,
        tool: str,
        path: str,
        request: dict | str,
        response: dict | str,
        status: int,
    ) -> None:
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).isoformat()
        req_str  = json.dumps(request)  if isinstance(request,  dict) else request
        resp_str = json.dumps(response) if isinstance(response, dict) else response
        with self._lock:
            con = self._connect()
            con.execute(
                "INSERT OR IGNORE INTO tool_calls "
                "(id, tool, path, request, response, status, ts) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (call_id, tool, path, req_str, resp_str, status, ts),
            )
            con.commit()
            con.close()

    def rate(self, call_id: str, rating: int, comment: str | None = None) -> bool:
        """Set rating (-1/1) and optional comment. Returns False if id not found."""
        with self._lock:
            con = self._connect()
            cur = con.execute(
                "UPDATE tool_calls SET rating=?, comment=? WHERE id=?",
                (rating, comment, call_id),
            )
            con.commit()
            found = cur.rowcount > 0
            con.close()
        return found

    def purge(
        self,
        tool: str | None = None,
        rated_only: bool = False,
    ) -> int:
        """
        Delete rows from tool_calls.
        - tool=None, rated_only=False  → delete ALL rows
        - tool='...'                   → delete only rows for that tool
        - rated_only=True              → delete only rows that have a rating
        Returns the number of rows deleted.
        """
        clauses = []
        params: list = []
        if tool:
            clauses.append("tool = ?")
            params.append(tool)
        if rated_only:
            clauses.append("rating IS NOT NULL")
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._lock:
            con = self._connect()
            cur = con.execute(f"DELETE FROM tool_calls {where}", params)
            con.commit()
            deleted = cur.rowcount
            con.close()
        return deleted

    def export(
        self,
        tool: str | None = None,
        rated_only: bool = False,
        limit: int = 500,
    ) -> list[dict]:
        clauses = []
        params: list = []
        if tool:
            clauses.append("tool = ?")
            params.append(tool)
        if rated_only:
            clauses.append("rating IS NOT NULL")
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        with self._lock:
            con = self._connect()
            rows = con.execute(
                f"SELECT id, tool, path, request, response, status, ts, rating, comment "
                f"FROM tool_calls {where} ORDER BY ts DESC LIMIT ?",
                params,
            ).fetchall()
            con.close()
        keys = ["id", "tool", "path", "request", "response", "status", "ts", "rating", "comment"]
        result = []
        for row in rows:
            d = dict(zip(keys, row))
            # Deserialise stored JSON strings back to objects
            for field in ("request", "response"):
                try:
                    d[field] = json.loads(d[field])
                except Exception:
                    pass
            result.append(d)
        return result


# Singleton — initialised once at module load
_call_log = _CallLog(CALL_LOG_DB)


def _require_admin_key(request: "Request") -> "JSONResponse | None":
    """
    Return a 403 JSONResponse if the request does not supply the correct
    admin key in the ``X-Admin-Key`` header; return None if the key is valid.
    Uses hmac.compare_digest to prevent timing-based attacks.
    """
    import hmac
    provided = request.headers.get("X-Admin-Key", "")
    if not provided or not hmac.compare_digest(provided, ADMIN_KEY):
        return JSONResponse(
            {"error": "Forbidden: valid X-Admin-Key header required"},
            status_code=403,
        )
    return None


# ---------------------------------------------------------------------------
# Middleware: inject "tool" field into every JSON response
# ---------------------------------------------------------------------------

# Maps URL path → tool name injected into the response body.
_PATH_TO_TOOL: dict[str, str] = {
    "/generate":       "generate_canvasxpress_config",
    "/modify":         "modify_canvasxpress_config",
    "/km":             "generate_km_config",
    "/params":         "get_chart_parameters",
    "/axes":           "suggest_axes",
    "/select":         "select_canvasxpress_chart",
    "/explain":        "explain_canvasxpress_property",
    "/explain-r":      "explain_canvasxpress_r",
    "/explain-ggplot": "explain_ggplot_to_canvasxpress",
    "/minimal-params": "get_minimal_params",
}


class _InjectToolMiddleware:
    """
    Injects ``"tool"``, ``"valid"``, ``"datetime"``, and ``"request_id"`` into
    every JSON API response, and logs every call to the call-log SQLite DB.
    """

    def __init__(self, app):
        self._app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        path = scope.get("path", "")
        tool_name = _PATH_TO_TOOL.get(path)
        if tool_name is None:
            await self._app(scope, receive, send)
            return

        # --- Buffer the request body so we can log it ----------------------
        req_body_chunks: list[bytes] = []

        async def _buffered_receive():
            msg = await receive()
            if msg["type"] == "http.request":
                req_body_chunks.append(msg.get("body", b""))
            return msg

        # --- Intercept the response ----------------------------------------
        status_code = 200
        original_headers: list = []
        body_chunks: list[bytes] = []

        async def _intercept_send(message):
            nonlocal status_code, original_headers
            if message["type"] == "http.response.start":
                status_code = message["status"]
                original_headers = list(message.get("headers", []))
            elif message["type"] == "http.response.body":
                body_chunks.append(message.get("body", b""))
                if not message.get("more_body", False):
                    raw = b"".join(body_chunks)
                    call_id = str(_uuid.uuid4())
                    try:
                        data = json.loads(raw)
                        if isinstance(data, dict) and "error" not in data:
                            data["tool"]       = tool_name
                            data["request_id"] = call_id
                            if "valid" not in data:
                                data["valid"] = True
                            if "datetime" not in data:
                                from datetime import datetime, timezone
                                data["datetime"] = datetime.now(timezone.utc).strftime(
                                    "%a, %d %b %Y %H:%M:%S GMT"
                                )
                        raw = json.dumps(data).encode()

                        # --- Log to SQLite (best-effort, never block response) ---
                        try:
                            req_raw = b"".join(req_body_chunks)
                            try:
                                req_obj = json.loads(req_raw) if req_raw else {}
                            except Exception:
                                req_obj = req_raw.decode(errors="replace")
                            # Store only the non-binary parts of the response
                            resp_obj = data if isinstance(data, dict) else {}
                            _call_log.log(
                                call_id=call_id,
                                tool=tool_name,
                                path=path,
                                request=req_obj,
                                response=resp_obj,
                                status=status_code,
                            )
                        except Exception as _log_exc:
                            log.debug("call-log write failed: %s", _log_exc)

                    except Exception:
                        pass  # not JSON — pass through unchanged

                    new_headers = [
                        (k, v) for k, v in original_headers
                        if k.lower() not in (b"content-length",)
                    ]
                    new_headers.append((b"content-length", str(len(raw)).encode()))
                    await send({
                        "type":    "http.response.start",
                        "status":  status_code,
                        "headers": new_headers,
                    })
                    await send({
                        "type":      "http.response.body",
                        "body":      raw,
                        "more_body": False,
                    })

        await self._app(scope, _buffered_receive, _intercept_send)


# Build the CORS ASGI middleware list — passed to mcp.run() below.
# CORS_ORIGINS is read from .env (CORS_ORIGINS env var).
_cors_middleware: list = [
    Middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-API-Key", "X-Admin-Key"],
    ),
    Middleware(_InjectToolMiddleware),
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
    result = generate_config(description, resolved_headers, column_types, temperature)
    if isinstance(result, tuple):
        config, removed_params = result
    else:
        config, removed_params = result, []

    if not config:
        log.warning("LLM did not return a usable config for description: %s", description)
        return {
            "config":         {},
            "valid":          False,
            "warnings":       [
                "The description did not produce a valid CanvasXpress configuration. "
                "Try rephrasing — for example: 'Bar chart of expression values by gene', "
                "'Heatmap with RdBu colors', or 'Violin plot of score grouped by treatment'."
            ],
            "invalid_refs":   {},
            "headers_used":   resolved_headers or [],
            "types_used":     column_types or {},
            "removed_params": [],
        }

    graph_type = config.get("graphType", "unknown")
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
    result = modify_config(config, instruction, resolved_headers, column_types, temperature)
    if isinstance(result, tuple):
        modified, removed_params = result
    else:
        modified, removed_params = result, []

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
        "prompt":         instruction,
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
        "Generate, validate, and detect columns for Kaplan-Meier survival plot configs. "
        "Accepts any combination of: a plain English description, column headers or a full data "
        "array, and/or an existing config to validate and fix. "
        "Automatically detects which columns are time, event, and grouping from the dataset. "
        "KM statistics (median survival, confidence intervals, log-rank p-value) are computed "
        "and rendered by CanvasXpress itself. "
        "Examples: description='OS curve by treatment arm' with headers=['PatientID','OS_Time','OS_Status','Treatment']; "
        "or config={...} to validate an existing KM config."
    )
)
def generate_km_config(
    description: str | None = None,
    headers:     list[str] | None = None,
    data:        list[list] | None = None,
    config:      dict | None = None,
    temperature: float = 0.0,
) -> dict:
    """
    Args:
        description: Plain English description of the KM plot you want.
                     e.g. "Overall survival by treatment arm with 95% CI"
        headers:     Column names from your dataset.
                     e.g. ["PatientID", "OS_Time", "OS_Status", "Treatment"]
        data:        Full data array — first row must be column headers.
                     e.g. [["ID","Time","Event","Arm"],["P1",24,1,"A"],...]
                     When provided, headers are extracted automatically.
        config:      An existing KM config to validate, fix, and/or enrich.
                     e.g. {"graphType":"KaplanMeier","xAxis":["OS_Time"],...}
        temperature: LLM creativity 0.0–1.0 (default 0.0 = deterministic).

    Returns:
        Dict with keys:
          config           (dict)      - the CanvasXpress KM JSON config
          valid            (bool)      - True if config passes all KM validation rules
          errors           (list)      - must-fix issues (e.g. missing xAxis)
          warnings         (list)      - should-fix issues and notes
          suggestions      (list)      - optional improvements
          column_detection (dict|None) - detected time/event/group columns + confidence
    """
    log.info(
        "KM skill: description=%s headers=%s data_rows=%s config_keys=%s",
        bool(description), bool(headers),
        len(data) - 1 if data else 0,
        list(config.keys()) if config else None,
    )

    if not any([description, headers, data, config]):
        return {
            "config":           {"graphType": "KaplanMeier"},
            "valid":            False,
            "errors":           ["At least one of description, headers, data, or config must be provided."],
            "warnings":         [],
            "suggestions":      ["Pass headers or data so columns can be detected automatically."],
            "column_detection": None,
        }

    return cx_survival.handle_generate_km(
        description     = description,
        headers         = headers,
        data            = data,
        config          = config,
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

@mcp.tool(
    description=(
        "Return axis assignment rules for a given CanvasXpress graph type. "
        "Explains which axis keys are valid (xAxis, yAxis, zAxis, xAxis2), "
        "which are forbidden, what data types belong on each axis, and which "
        "axis title parameter applies (smpTitle vs xAxisTitle/yAxisTitle). "
        "Use this before generating or modifying a config when you are unsure "
        "about axis structure for a chart type. "
        "Examples: graph_type='Bar', 'Scatter2D', 'Heatmap', 'KaplanMeier', 'BarLine'."
    )
)
def get_axes_info(graph_type: str) -> dict:
    """
    Args:
        graph_type: CanvasXpress graph type string, e.g. "Bar", "Scatter2D", "Heatmap".

    Returns:
        Dict with keys:
          graph_type        (str)  - the requested graph type
          category          (str)  - "single_dim" | "multi_dim" | "combined"
          valid_axes        (list) - axis keys that apply (e.g. ["xAxis"])
          invalid_axes      (list) - axis keys that must NOT be used
          axis_title_param  (str)  - correct axis title param for the category
          notes             (str)  - human-readable summary of the axis rules
          schema_snippet    (str)  - relevant axis params from the live cx_knowledge schema
    """
    COMBINED = {"AreaLine", "BarLine", "DotLine", "Pareto", "StackedLine", "StackedPercentLine"}
    MULTI_DIM = {
        "Scatter2D", "Scatter3D", "ScatterBubble2D", "Volcano", "Spaghetti",
        "Contour", "Streamgraph", "Bump", "KaplanMeier", "TimeSeries",
    }

    gt = graph_type.strip()

    if gt in MULTI_DIM:
        needs_z = gt in {"Scatter3D", "ScatterBubble2D"}
        valid_axes = ["xAxis", "yAxis"] + (["zAxis"] if needs_z else [])
        category = "multi_dim"
        invalid_axes = []
        axis_title_param = "xAxisTitle / yAxisTitle"
        notes = (
            f"{gt} requires both xAxis (numeric) and yAxis (numeric). "
            + ("Also requires zAxis. " if needs_z else "")
            + "Use xAxisTitle and yAxisTitle for axis labels. Never use smpTitle."
        )
    elif gt in COMBINED:
        valid_axes = ["xAxis", "xAxis2"]
        category = "combined"
        invalid_axes = ["yAxis"]
        axis_title_param = "smpTitle"
        notes = (
            f"{gt} uses xAxis for the primary numeric series and xAxis2 for the "
            "secondary numeric series. Never use yAxis. "
            "Use smpTitle to label the categorical sample axis, never yAxisTitle."
        )
    else:
        # Default to single_dim (covers known + unknown graph types)
        valid_axes = ["xAxis"]
        category = "single_dim"
        invalid_axes = ["yAxis"]
        axis_title_param = "smpTitle"
        heatmap_note = (
            " Exception: for Heatmap, the variable names (genes/features) go in "
            "xAxis because that is how CanvasXpress structures heatmap data."
            if gt == "Heatmap" else ""
        )
        notes = (
            f"{gt} is single-dimensional: xAxis holds the NUMERIC column(s). "
            "The categorical dimension (samples) is populated automatically from the data — "
            "do not put category names in xAxis. "
            "Use smpTitle to label the sample axis. "
            "Never use yAxis or yAxisTitle." + heatmap_note
        )

    snippet = cx_knowledge.get_param_snippet(graph_type=gt)
    return {
        "graph_type":       gt,
        "category":         category,
        "valid_axes":       valid_axes,
        "invalid_axes":     invalid_axes,
        "axis_title_param": axis_title_param,
        "notes":            notes,
        "schema_snippet":   snippet or "(schema not loaded)",
    }




@mcp.tool(
    description=(
        "Recommend the most appropriate CanvasXpress graphType given column names, "
        "column types, and an optional plain-English description of what you want to show. "
        "Use this BEFORE generate_canvasxpress_config when you are unsure which chart "
        "type best fits your data structure. Accepts structured column metadata "
        "(numeric/factor/string/date) and returns a ranked list of graphType candidates "
        "with rationale, clinical use notes, and a ready-made description hint to pass "
        "directly to generate_canvasxpress_config. "
        "No LLM call — deterministic and instant. "
        "intent is optional — omitting it scores purely on column structure and column name patterns. "
        "Examples: column_types={\'SOC\':\'factor\',\'Treatment\':\'factor\',\'AE_Count\':\'numeric\'}; "
        "intent=\'overall survival by arm\' with "
        "column_types={\'Time\':\'numeric\',\'Event\':\'numeric\',\'Arm\':\'factor\'}."
    )
)
def select_canvasxpress_chart(
    intent: str,
    column_types: dict[str, str],
    n_samples: Optional[int] = None,
) -> dict:
    """
    Args:
        intent:       Plain-English description of what you want to show.
                      e.g. "AE counts by SOC across 3 treatment arms"
        column_types: Dict mapping column name to type.
                      Valid types: "numeric", "factor", "string", "date", "integer", "binary".
                      e.g. {"SOC": "factor", "Treatment": "factor", "AE_Count": "numeric"}
        n_samples:    Optional total row count. Used to prefer Dotplot over Boxplot for
                      small cohorts (n < 30) and warn about overplotting for large datasets
                      (n > 5000).

    Returns:
        Dict with keys:
          intent             (str)  - echoed back
          column_summary     (dict) - {n_factor, n_numeric, n_time}
          top_recommendation (dict) - best graphType with description, clinical_use, next_step
          alternatives       (list) - up to 3 other candidate graphTypes
          generate_hint      (str)  - suggested description string to pass to
                                      generate_canvasxpress_config as the \'description\' argument
    """
    log.info(
        "select_canvasxpress_chart: intent=%r  columns=%s  n_samples=%s",
        intent, list(column_types.keys()), n_samples,
    )
    def _llm_text(system: str, user: str) -> str:
        text, usage = llm_complete(system, user, temperature=0.0, max_tokens=300)
        log.debug("tiebreak LLM raw=%r  stop=%s", text[:120] if text else "", usage.get("stop_reason"))
        return text

    result = cx_selector.select_chart(intent, column_types, n_samples,
                                       llm_complete=_llm_text)
    tb = result.get("tiebreak", {})
    log.info(
        "select_canvasxpress_chart → top=%s  alts=%s  tiebreak_used=%s",
        result["top_recommendation"]["graphType"],
        [a["graphType"] for a in result["alternatives"]],
        tb.get("used", False),
    )

    # Attach minimal config to top recommendation and each alternative
    # and validate that all required axes are populated
    warnings: list[str] = []

    def _validate_minimal_config(gt: str, cfg: dict) -> list[str]:
        """Return a list of warnings for any required axis that is empty/missing."""
        req = get_minimal_parameters(gt).get("required_parameters", [])
        issues = []
        for param in req:
            if param == "graphType":
                continue
            val = cfg.get(param)
            if not val or val == []:
                issues.append(f"{gt}: required parameter '{param}' is not populated")
        return issues

    top_cfg = _build_minimal_config(result["top_recommendation"]["graphType"], column_types)
    result["top_recommendation"]["minimal_config"] = top_cfg
    warnings.extend(_validate_minimal_config(result["top_recommendation"]["graphType"], top_cfg))

    for alt in result["alternatives"]:
        alt_cfg = _build_minimal_config(alt["graphType"], column_types)
        alt["minimal_config"] = alt_cfg

    result["valid"]    = len(warnings) == 0
    result["warnings"] = warnings

    return result

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
    VALID_COLOR_SCHEMES = (
        "YlGn, YlGnBu, GnBu, BuGn, PuBuGn, PuBu, BuPu, RdPu, PuRd, OrRd, YlOrRd, YlOrBr, "
        "Purples, Blues, Greens, Oranges, Reds, Greys, PuOr, BrBG, PRGn, PiYG, RdBu, RdGy, "
        "RdYlBu, Spectral, RdYlGn, Bootstrap, Economist, Excel, GGPlot, Solarized, PaulTol, "
        "ColorBlind, Tableau, WallStreetJournal, Stata, BlackAndWhite, CanvasXpress"
    )
    VALID_THEMES = (
        "bw, classic, cx, dark, economist, excel, ggblanket, ggplot, gray, grey, "
        "highcharts, igray, light, linedraw, minimal, none, ptol, solarized, stata, "
        "tableau, void0, wsj"
    )
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


@mcp.tool(
    description=(
        "Explain how to use CanvasXpress in R. "
        "Covers installation, basic usage, creating charts with canvasXpress(), "
        "passing data frames, setting config parameters, and running in Shiny or R Markdown. "
        "Optionally filter by topic: 'installation', 'basic', 'shiny', 'rmarkdown', 'data', 'config'. "
        "Use this when a user asks how to use CanvasXpress in R or wants R code examples."
    )
)
def explain_canvasxpress_r(topic: str | None = None) -> dict:
    """
    Args:
        topic: Optional topic filter. One of: 'installation', 'basic', 'shiny',
               'rmarkdown', 'data', 'config'. Returns all topics if omitted.

    Returns:
        Dict with topic sections explaining CanvasXpress usage in R.
    """
    sections = {
        "installation": {
            "title": "Installation",
            "content": (
                "Install from CRAN:\n"
                "  install.packages('canvasXpress')\n\n"
                "Or install the development version from GitHub:\n"
                "  # install.packages('devtools')\n"
                "  devtools::install_github('neuhausi/canvasXpress')\n\n"
                "Load the package:\n"
                "  library(canvasXpress)"
            ),
        },
        "basic": {
            "title": "Basic Usage",
            "content": (
                "The main function is canvasXpress(). It accepts a data matrix/data frame "
                "and a list of configuration options.\n\n"
                "Simple bar chart example:\n"
                "  library(canvasXpress)\n\n"
                "  # Data: rows = variables/genes, cols = samples\n"
                "  data <- t(mtcars[1:5, 1:4])\n\n"
                "  canvasXpress(\n"
                "    data            = data,\n"
                "    graphType       = 'Bar',\n"
                "    title           = 'My Bar Chart',\n"
                "    colorScheme     = 'Blues'\n"
                "  )\n\n"
                "Scatter plot example:\n"
                "  canvasXpress(\n"
                "    data            = mtcars,\n"
                "    asSampleData    = TRUE,\n"
                "    graphType       = 'Scatter2D',\n"
                "    xAxis           = list('wt'),\n"
                "    yAxis           = list('mpg'),\n"
                "    colorBy         = 'cyl',\n"
                "    title           = 'Weight vs MPG'\n"
                "  )"
            ),
        },
        "data": {
            "title": "Data Format",
            "content": (
                "CanvasXpress in R expects data in one of two orientations:\n\n"
                "1. Standard matrix orientation (default):\n"
                "   - Rows = variables (genes, features, metrics)\n"
                "   - Columns = samples\n"
                "   - Use for heatmaps, bar charts, boxplots, etc.\n\n"
                "   data <- matrix(rnorm(20), nrow=4,\n"
                "                  dimnames=list(c('GeneA','GeneB','GeneC','GeneD'),\n"
                "                               c('S1','S2','S3','S4','S5')))\n\n"
                "2. Sample-as-rows orientation (asSampleData = TRUE):\n"
                "   - Rows = observations/samples\n"
                "   - Columns = variables\n"
                "   - Use for scatter plots, PCA, regression, etc.\n\n"
                "   canvasXpress(data = iris, asSampleData = TRUE, ...)\n\n"
                "3. Annotation data (smpAnnot / varAnnot):\n"
                "   - smpAnnot: data frame of sample metadata (rows = samples)\n"
                "   - varAnnot: data frame of variable metadata (rows = variables)\n\n"
                "   smp_meta <- data.frame(Treatment = c('A','A','B','B','B'),\n"
                "                          row.names  = colnames(data))\n"
                "   canvasXpress(\n"
                "     data     = data,\n"
                "     smpAnnot = smp_meta,\n"
                "     graphType = 'Heatmap',\n"
                "     smpOverlays = list('Treatment')\n"
                "   )"
            ),
        },
        "config": {
            "title": "Configuration Parameters",
            "content": (
                "All CanvasXpress JSON config parameters map directly to R function arguments "
                "or can be passed via the 'config' list.\n\n"
                "Passing parameters directly:\n"
                "  canvasXpress(\n"
                "    data             = data,\n"
                "    graphType        = 'Heatmap',\n"
                "    colorScheme      = 'RdBu',\n"
                "    samplesClustered = TRUE,\n"
                "    variablesClustered = TRUE,\n"
                "    showLegend       = TRUE,\n"
                "    title            = 'Gene Expression Heatmap'\n"
                "  )\n\n"
                "Using the config list for bulk parameter passing:\n"
                "  cfg <- list(\n"
                "    graphType        = 'Heatmap',\n"
                "    colorScheme      = 'RdBu',\n"
                "    samplesClustered = TRUE\n"
                "  )\n"
                "  do.call(canvasXpress, c(list(data = data), cfg))\n\n"
                "Key parameters:\n"
                "  graphType        - chart type (Bar, Scatter2D, Heatmap, Violin, etc.)\n"
                "  xAxis            - list of column names for x-axis\n"
                "  yAxis            - list of column names for y-axis (multi-dim only)\n"
                "  groupingFactors  - list of columns to group/color by\n"
                "  colorScheme      - color palette name\n"
                "  colorBy          - column to color points by\n"
                "  title            - chart title\n"
                "  width / height   - widget dimensions in pixels"
            ),
        },
        "shiny": {
            "title": "Using CanvasXpress in Shiny",
            "content": (
                "CanvasXpress integrates with Shiny via canvasXpressOutput() and renderCanvasXpress().\n\n"
                "  library(shiny)\n"
                "  library(canvasXpress)\n\n"
                "  ui <- fluidPage(\n"
                "    canvasXpressOutput('myPlot', width = '800px', height = '500px')\n"
                "  )\n\n"
                "  server <- function(input, output) {\n"
                "    output$myPlot <- renderCanvasXpress({\n"
                "      canvasXpress(\n"
                "        data        = t(mtcars[1:5, 1:4]),\n"
                "        graphType   = 'Bar',\n"
                "        colorScheme = 'Tableau'\n"
                "      )\n"
                "    })\n"
                "  }\n\n"
                "  shinyApp(ui, server)\n\n"
                "Reactive updates: rebuild the canvasXpress() call inside renderCanvasXpress() "
                "using reactive inputs normally — the widget re-renders automatically."
            ),
        },
        "rmarkdown": {
            "title": "Using CanvasXpress in R Markdown / Quarto",
            "content": (
                "CanvasXpress renders as an interactive HTML widget in R Markdown and Quarto documents.\n\n"
                "In an R Markdown chunk:\n"
                "  ```{r}\n"
                "  library(canvasXpress)\n\n"
                "  canvasXpress(\n"
                "    data      = t(mtcars[1:8, 1:5]),\n"
                "    graphType = 'Boxplot',\n"
                "    title     = 'mtcars Boxplot'\n"
                "  )\n"
                "  ```\n\n"
                "For static PDF output, use the saveAsPng argument or htmltools::save_html() "
                "to export as a standalone HTML file first:\n"
                "  library(htmltools)\n"
                "  cx <- canvasXpress(data = t(mtcars), graphType = 'Bar')\n"
                "  save_html(cx, 'my_chart.html')"
            ),
        },
    }

    topic_lower = topic.strip().lower() if topic else None
    if topic_lower and topic_lower in sections:
        return {
            "topic": topic_lower,
            "section": sections[topic_lower],
            "available_topics": list(sections.keys()),
        }

    if topic_lower and topic_lower not in sections:
        return {
            "error": f"Unknown topic '{topic}'. Valid topics: {list(sections.keys())}",
            "available_topics": list(sections.keys()),
        }

    return {
        "overview": (
            "The canvasXpress R package wraps the CanvasXpress JavaScript library as an "
            "htmlwidget, enabling interactive charts in RStudio, Shiny apps, and R Markdown "
            "documents. All CanvasXpress chart types and config parameters are supported."
        ),
        "sections": sections,
        "available_topics": list(sections.keys()),
    }


@mcp.tool(
    description=(
        "Explain how to use CanvasXpress with ggplot2 in R. "
        "canvasXpress() accepts a ggplot object directly and converts it to an interactive widget — "
        "no separate bridge function needed. Covers supported geoms, a usage example, and how the "
        "translation works. Use this when a user asks about ggplot2 + CanvasXpress."
    )
)
def explain_canvasxpress_ggplot(topic: str | None = None) -> dict:
    """
    Args:
        topic: Optional topic filter. One of: 'overview', 'installation', 'geoms', 'example'.
               Returns all topics if omitted.

    Returns:
        Dict with topic sections explaining CanvasXpress ggplot2 integration.
    """
    sections = {
        "overview": {
            "title": "Overview",
            "content": (
                "canvasXpress() accepts a ggplot2 object directly and converts it into an interactive "
                "CanvasXpress HTML widget. It acts as a translator: it parses the ggplot object's "
                "layers, geoms, and aesthetic mappings, converts them to a CanvasXpress JSON "
                "configuration, and renders an interactive chart with tooltips, zooming, and "
                "interactive legends.\n\n"
                "You build your plot with standard ggplot2 syntax, then pass the ggplot object to "
                "canvasXpress() instead of printing it:\n\n"
                "  library(ggplot2)\n"
                "  library(canvasXpress)\n\n"
                "  p <- ggplot(iris, aes(x = Petal.Length, y = Petal.Width, color = Species)) +\n"
                "         geom_point(size = 3) +\n"
                "         labs(title = 'Iris Petal Length vs. Width') +\n"
                "         theme_minimal()\n\n"
                "  canvasXpress(p)  # renders as an interactive CanvasXpress widget"
            ),
        },
        "installation": {
            "title": "Installation",
            "content": (
                "Both packages are needed — the ggplot2 integration is built into canvasXpress.\n\n"
                "  install.packages('canvasXpress')\n"
                "  install.packages('ggplot2')\n\n"
                "  library(canvasXpress)\n"
                "  library(ggplot2)\n\n"
                "Or install the development version of canvasXpress:\n"
                "  devtools::install_github('neuhausi/canvasXpress')\n\n"
                "The result renders as an interactive htmlwidget in RStudio, Shiny, and R Markdown."
            ),
        },
        "geoms": {
            "title": "Supported Geoms",
            "content": (
                "canvasXpress() supports 22+ ggplot2 geom types:\n\n"
                "Statistical & Density:\n"
                "  geom_density(), geom_density_2d(), geom_smooth(), geom_contour()\n\n"
                "Categorical:\n"
                "  geom_bar(), geom_col(), geom_boxplot(), geom_violin()\n\n"
                "Positional:\n"
                "  geom_point(), geom_jitter(), geom_dotplot(), geom_rug()\n\n"
                "Connectors:\n"
                "  geom_path(), geom_line(), geom_step(), geom_ribbon(), geom_area()\n\n"
                "Annotations:\n"
                "  geom_text(), geom_label(), geom_abline(), geom_hline(), geom_vline()\n\n"
                "Binning:\n"
                "  geom_bin2d(), geom_hex()\n\n"
                "Other:\n"
                "  geom_qq(), geom_quantile(), geom_raster()"
            ),
        },
        "example": {
            "title": "Usage Example",
            "content": (
                "Build any ggplot2 plot as normal, then pass the object to canvasXpress():\n\n"
                "  library(ggplot2)\n"
                "  library(canvasXpress)\n\n"
                "  # Scatter plot\n"
                "  p <- ggplot(iris, aes(x = Petal.Length, y = Petal.Width, color = Species)) +\n"
                "         geom_point(size = 3) +\n"
                "         labs(title = 'Iris Petal Length vs. Width')\n"
                "  canvasXpress(p)\n\n"
                "  # Boxplot\n"
                "  p2 <- ggplot(iris, aes(x = Species, y = Sepal.Length, fill = Species)) +\n"
                "          geom_boxplot()\n"
                "  canvasXpress(p2)\n\n"
                "  # Density with smoothing\n"
                "  p3 <- ggplot(mtcars, aes(x = wt, y = mpg)) +\n"
                "          geom_point() +\n"
                "          geom_smooth(method = 'lm')\n"
                "  canvasXpress(p3)\n\n"
                "The resulting widget includes interactive tooltips, zooming, pan, and legend toggling."
            ),
        },
    }

    topic_lower = topic.strip().lower() if topic else None
    if topic_lower and topic_lower in sections:
        return {
            "topic": topic_lower,
            "section": sections[topic_lower],
            "available_topics": list(sections.keys()),
        }

    if topic_lower and topic_lower not in sections:
        return {
            "error": f"Unknown topic '{topic}'. Valid topics: {list(sections.keys())}",
            "available_topics": list(sections.keys()),
        }

    return {
        "overview": (
            "canvasXpress() accepts a ggplot2 object directly — build your plot with ggplot2 syntax "
            "and pass it to canvasXpress() to get an interactive widget. Supports 22+ geom types."
        ),
        "sections": sections,
        "available_topics": list(sections.keys()),
    }


# ---------------------------------------------------------------------------
# Minimal-config builder (used by select_canvasxpress_chart)
# ---------------------------------------------------------------------------

# Chart types whose numeric columns go on both xAxis and yAxis
_MULTI_DIM_GTS = {
    "Scatter2D", "Scatter3D", "ScatterBubble2D", "Contour", "Streamgraph",
    "TimeSeries", "Spaghetti", "KaplanMeier", "Volcano",
}
# Chart types that use a second numeric axis (xAxis2)
_COMBINED_GTS = {
    "AreaLine", "BarLine", "DotLine", "StackedLine", "StackedPercentLine", "Pareto",
}
# Chart types that use groupingFactors instead of (or in addition to) a factor in xAxis
_GROUPING_GTS = {"Boxplot", "Violin", "Dotplot", "Treemap", "Ridgeline"}


def _build_minimal_config(
    graph_type: str,
    column_types: dict[str, str],
) -> dict:
    """
    Build the minimal CanvasXpress config for *graph_type* given *column_types*.

    Assigns columns to graphType-appropriate axes based on their types.
    Returns a ready-to-use config dict (no LLM call).
    """
    import cx_selector as _cxs

    # Exclude subject/ID columns (e.g. "Id", "PatientId") from numeric candidates
    # so they don't end up assigned to chart axes.
    def _is_id_col(name: str) -> bool:
        return _cxs._col_matches(name, "subject") and not _cxs._col_matches(name, "group")

    num_cols  = [c for c, t in column_types.items()
                 if t.lower() in _cxs._NUMERIC_ALIASES and not _is_id_col(c)]
    fac_cols  = [c for c, t in column_types.items() if t.lower() in _cxs._FACTOR_ALIASES]
    time_cols = [c for c, t in column_types.items() if t.lower() in _cxs._TIME_ALIASES]
    bool_cols = [c for c, t in column_types.items() if t.lower() in _cxs._BOOL_ALIASES]

    cfg: dict = {"graphType": graph_type}

    if graph_type in _MULTI_DIM_GTS:
        # xAxis = first numeric (or time), yAxis = second numeric
        x_candidates = time_cols + num_cols
        if x_candidates:
            cfg["xAxis"] = [x_candidates[0]]
        remaining_num = [c for c in num_cols if c not in cfg.get("xAxis", [])]
        if remaining_num:
            cfg["yAxis"] = [remaining_num[0]]
        if graph_type == "Scatter3D" and len(remaining_num) >= 2:
            cfg["zAxis"] = [remaining_num[1]]
        if graph_type == "ScatterBubble2D" and len(remaining_num) >= 2:
            cfg["zAxis"] = [remaining_num[1]]
        if graph_type == "KaplanMeier":
            if bool_cols:
                cfg["yAxis"] = bool_cols[:1]
            elif len(num_cols) >= 2:
                cfg["yAxis"] = [num_cols[1]]
            if fac_cols:
                cfg["colorBy"] = fac_cols[0]

    elif graph_type in _COMBINED_GTS:
        if num_cols:
            cfg["xAxis"] = [num_cols[0]]
        if len(num_cols) >= 2:
            cfg["xAxis2"] = [num_cols[1]]
        if fac_cols:
            cfg["smpTitle"] = fac_cols[0]

    elif graph_type in _GROUPING_GTS:
        if num_cols:
            cfg["xAxis"] = [num_cols[0]]
        if fac_cols:
            cfg["groupingFactors"] = [fac_cols[0]]

    elif graph_type == "Heatmap":
        cfg["xAxis"] = num_cols or [c for c in column_types]

    elif graph_type in {"Sunburst", "Tree"}:
        cfg["hierarchy"] = fac_cols[:2] if fac_cols else list(column_types.keys())[:2]
        if num_cols:
            cfg["xAxis"] = [num_cols[0]]

    elif graph_type in {"Sankey", "Alluvial"}:
        cfg["sankeyAxes"] = fac_cols[:2] if len(fac_cols) >= 2 else list(column_types.keys())[:2]
        if num_cols:
            cfg["xAxis"] = [num_cols[0]]

    elif graph_type == "Ridgeline":
        if num_cols:
            cfg["xAxis"] = [num_cols[0]]
        if fac_cols:
            cfg["ridgeBy"] = fac_cols[0]

    elif graph_type in {"TagCloud", "WordCloud"}:
        text_cols = [c for c, t in column_types.items() if t.lower() in _cxs._TEXT_ALIASES] or fac_cols
        if text_cols:
            cfg["xAxis"] = [text_cols[0]]
        if fac_cols:
            cfg["colorBy"] = fac_cols[0]

    elif graph_type == "Spaghetti":
        if num_cols:
            cfg["xAxis"] = [num_cols[0]]
        if len(num_cols) >= 2:
            cfg["yAxis"] = [num_cols[1]]
        if fac_cols:
            cfg["colorBy"] = fac_cols[0]

    else:
        # Default single-dim: numeric → xAxis, first factor → groupingFactors if present
        if num_cols:
            cfg["xAxis"] = num_cols[:1]
        if fac_cols and graph_type not in {"Pie", "Donut", "Bar", "Stacked",
                                            "StackedPercent", "Waterfall", "Lollipop"}:
            cfg["groupingFactors"] = [fac_cols[0]]

    # Color/group by second factor when available and not already set
    if len(fac_cols) >= 2 and "colorBy" not in cfg and graph_type not in _GROUPING_GTS:
        cfg["colorBy"] = fac_cols[1]
    elif fac_cols and "colorBy" not in cfg and graph_type in _MULTI_DIM_GTS:
        cfg["colorBy"] = fac_cols[0]

    return cfg


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


def _infer_column_types(data: list[list]) -> tuple[dict[str, str], list[str], int]:
    """
    Infer column types from a data array (first row = headers).

    Returns:
        column_types  — {col_name: type_string} where type is one of
                        'numeric', 'date', 'boolean', or 'factor'
        headers       — list of header strings
        n_samples     — number of data rows (excluding header)
    """
    if not data or len(data) < 1:
        return {}, [], 0

    headers = [str(h) for h in data[0]]
    rows = data[1:]
    n_samples = len(rows)

    _DATE_RE = re.compile(
        r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}"  # YYYY-MM-DD or YYYY/MM/DD
        r"|^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}"  # MM/DD/YYYY
        r"|^\d{4}$"                            # bare year
    )
    _BOOL_VALS = {"true", "false", "yes", "no", "1", "0", "t", "f", "y", "n"}

    column_types: dict[str, str] = {}
    for col_idx, col_name in enumerate(headers):
        values = []
        for row in rows:
            if col_idx < len(row) and row[col_idx] is not None and str(row[col_idx]).strip() not in ("", "NA", "na", "nan", "NaN", "NULL", "null"):
                values.append(row[col_idx])

        if not values:
            column_types[col_name] = "factor"
            continue

        # Try numeric first
        numeric_count = 0
        for v in values:
            try:
                float(str(v).replace(",", ""))
                numeric_count += 1
            except (ValueError, TypeError):
                pass
        if numeric_count == len(values):
            # 0/1-only integer columns are boolean (e.g. Event/censoring indicators)
            unique_vals = {str(v).strip() for v in values}
            if unique_vals <= {"0", "1"}:
                column_types[col_name] = "boolean"
            else:
                column_types[col_name] = "numeric"
            continue

        # Try date
        str_values = [str(v).strip() for v in values]
        if all(_DATE_RE.match(sv) for sv in str_values):
            column_types[col_name] = "date"
            continue

        # Try boolean
        if all(sv.lower() in _BOOL_VALS for sv in str_values):
            column_types[col_name] = "boolean"
            continue

        # Default to factor
        column_types[col_name] = "factor"

    return column_types, headers, n_samples


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
        # Auto-derive description from data when no explicit description is given
        raw_data = p.get("data")
        if raw_data:
            try:
                data_arr = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
                col_types, _, n_samp = _infer_column_types(data_arr)
                sel = cx_selector.select_chart("", col_types, n_samples=n_samp)
                desc = sel.get("generate_hint") or sel["top_recommendation"]["graphType"] + " chart"
            except Exception:
                pass
        if not desc:
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
<link rel="icon" href="/favicon.ico" type="image/png">
<style>
  *{box-sizing:border-box}
  body{font-family:system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 20px;background:#f5f5f5;color:#222}
  h1{font-size:1.4rem;margin-bottom:4px}
  h1 span{color:#c0392b}
  .subtitle{color:#666;font-size:.85rem;margin-bottom:16px}
  label{display:block;font-weight:600;font-size:.85rem;margin-top:14px;margin-bottom:3px}
  label span{font-weight:400;color:#888}
  input[type=text],textarea,select{width:100%;padding:7px 10px;border:1px solid #ccc;border-radius:5px;font-size:.9rem;font-family:inherit;background:#fff}
  textarea{resize:vertical;min-height:80px}
  .row{display:grid;grid-template-columns:1fr 1fr;gap:12px}
  .actions{margin-top:18px;display:flex;align-items:center;gap:10px;flex-wrap:wrap}
  button{padding:8px 18px;border:none;border-radius:5px;cursor:pointer;font-size:.9rem;font-weight:600}
  .btn-primary{background:#c0392b;color:#fff}
  .btn-secondary{background:#555;color:#fff}
  #url-box{flex:1;min-width:200px;font-size:.75rem;color:#555;background:#fff;border:1px solid #ccc;border-radius:5px;padding:7px 10px;word-break:break-all;cursor:text;white-space:pre-wrap}
  #result{margin-top:24px;background:#fff;border:1px solid #ddd;border-radius:6px;padding:16px;display:none}
  #result h3{margin:0 0 10px;font-size:.95rem}
  pre{margin:0;font-size:.82rem;overflow:auto;max-height:500px;background:#f8f8f8;padding:10px;border-radius:4px}
  .badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:.75rem;font-weight:700;margin-left:6px}
  .valid{background:#d4edda;color:#155724}
  .invalid{background:#f8d7da;color:#721c24}
  .warn{background:#fff3cd;color:#856404}
  .meta{font-size:.8rem;color:#666;margin-bottom:8px}
  .tab-bar{display:flex;gap:2px;margin-bottom:0;flex-wrap:nowrap;overflow-x:auto;border-bottom:1px solid #ccc;-webkit-overflow-scrolling:touch}
  .tab{padding:7px 13px;border-radius:5px 5px 0 0;border:1px solid #ccc;border-bottom:none;cursor:pointer;font-weight:600;font-size:.8rem;background:#eee;margin-bottom:-1px;white-space:nowrap}
  .tab.active{background:#fff;border-bottom:1px solid #fff}
  .panel{display:none}
  .panel.active{display:block}
  .card{background:#fff;border:1px solid #ddd;border-top:none;border-radius:0 0 6px 6px;padding:16px}
  .hint{font-size:.78rem;color:#888;margin-top:3px}
  .section-label{font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.05em;color:#999;margin-top:18px;margin-bottom:4px;border-bottom:1px solid #eee;padding-bottom:3px}
</style>
</head>
<body>
<h1>CanvasXpress <span>MCP</span> — Web UI</h1>
<p class="subtitle">Test all MCP server endpoints. Parameters are encoded in the URL — bookmark or share.</p>

<div class="tab-bar">
  <div class="tab active" data-tab="generate">Generate</div>
  <div class="tab" data-tab="modify">Modify</div>
  <div class="tab" data-tab="km">Kaplan-Meier</div>
  <div class="tab" data-tab="params">Params</div>
  <div class="tab" data-tab="axes">Axes</div>
  <div class="tab" data-tab="select">Select Chart</div>
  <div class="tab" data-tab="explain">Explain</div>
  <div class="tab" data-tab="explain-r">Explain R</div>
  <div class="tab" data-tab="explain-ggplot">Explain ggplot</div>
  <div class="tab" data-tab="minimal-params">Minimal Params</div>
</div>

<!-- ── GENERATE ─────────────────────────────────────────────────────── -->
<div class="panel card active" id="panel-generate">
  <div class="section-label">Required</div>
  <label>Description / Prompt
    <input type="text" id="g-desc" placeholder="e.g. Clustered heatmap with RdBu colors and dendrograms on both axes">
  </label>
  <div class="section-label">Optional</div>
  <div class="row">
    <div><label>Headers <span>comma-separated</span><input type="text" id="g-headers" placeholder="Gene, Sample1, Sample2, Treatment"></label></div>
    <div><label>Column types <span>Col=type,…</span><input type="text" id="g-types" placeholder="Gene=string,Sample1=numeric,Treatment=factor"></label></div>
  </div>
  <label>Data <span>JSON array of arrays — first row is headers</span>
    <textarea id="g-data" placeholder='[["Gene","S1","Treatment"],["BRCA1",1.2,"Control"]]'></textarea>
  </label>
  <label style="display:inline-block;width:auto">Temperature <span>0=deterministic</span></label>
  <input type="text" id="g-temp" value="0" style="width:80px;margin-left:8px">
</div>

<!-- ── MODIFY ───────────────────────────────────────────────────────── -->
<div class="panel card" id="panel-modify">
  <div class="section-label">Required</div>
  <label>Existing config <span>JSON object</span>
    <textarea id="m-config" style="min-height:120px" placeholder='{"graphType":"Bar","xAxis":["Expression"]}'></textarea>
  </label>
  <label>Instruction<input type="text" id="m-instr" placeholder="add a title My Chart and change colorScheme to Tableau"></label>
  <div class="section-label">Optional</div>
  <div class="row">
    <div><label>Headers<input type="text" id="m-headers" placeholder="Gene, Sample1, Treatment"></label></div>
    <div><label>Column types<input type="text" id="m-types" placeholder="Gene=string,Sample1=numeric"></label></div>
  </div>
  <label>Data<textarea id="m-data" placeholder='[["Gene","S1"],["BRCA1",1.2]]'></textarea></label>
</div>

<!-- ── KAPLAN-MEIER ─────────────────────────────────────────────────── -->
<div class="panel card" id="panel-km">
  <p class="hint">At least one of description, headers, data, or config is required.</p>
  <div class="section-label">Optional — provide at least one</div>
  <label>Description<input type="text" id="km-desc" placeholder="e.g. Overall survival curve by treatment arm"></label>
  <div class="row">
    <div><label>Headers<input type="text" id="km-headers" placeholder="PatientID, OS_Time, OS_Status, Treatment"></label></div>
    <div><label>Existing config <span>JSON — to validate/fix</span><input type="text" id="km-config" placeholder='{"graphType":"KaplanMeier","xAxis":["OS_Time"]}'></label></div>
  </div>
  <label>Data <span>JSON array — enables column detection</span>
    <textarea id="km-data" placeholder='[["ID","Time","Event","Arm"],["P1",24,1,"Control"],["P2",18,0,"Drug A"]]'></textarea>
  </label>
  <label style="display:inline-block;width:auto">Temperature</label>
  <input type="text" id="km-temp" value="0" style="width:80px;margin-left:8px">
</div>

<!-- ── PARAMS ───────────────────────────────────────────────────────── -->
<div class="panel card" id="panel-params">
  <p class="hint">Pass graph_type, param_name, both, or neither for a full schema summary.</p>
  <div class="row">
    <div><label>Graph type <span>optional</span><input type="text" id="p-graph" placeholder="e.g. Heatmap, Scatter2D, Violin"></label></div>
    <div><label>Parameter name <span>optional</span><input type="text" id="p-param" placeholder="e.g. colorScheme, areaType, lineType"></label></div>
  </div>
  <label style="display:inline-flex;align-items:center;gap:8px;margin-top:14px">
    <input type="checkbox" id="p-refresh" style="width:auto"> <span>Force refresh from GitHub</span>
  </label>
</div>

<!-- ── AXES ─────────────────────────────────────────────────────────── -->
<div class="panel card" id="panel-axes">
  <p class="hint">Returns axis assignment rules: valid axes, forbidden axes, and axis title parameter.</p>
  <label>Graph type <span>required</span><input type="text" id="ax-graph" placeholder="e.g. Bar, Scatter2D, Heatmap, KaplanMeier, BarLine"></label>
</div>

<!-- ── SELECT CHART ─────────────────────────────────────────────────── -->
<div class="panel card" id="panel-select">
  <p class="hint">Deterministic chart type recommendation — no LLM call. Returns ranked candidates with rationale.<br>Provide <b>data</b>, <b>column_types</b>, or both. Intent is optional — omitting it scores on structure and column names only.</p>
  <div class="section-label">Required — provide at least one of data or column_types</div>
  <label>Data <span>JSON array of arrays — first row is headers; column types are inferred</span>
    <textarea id="sel-data" style="min-height:80px" placeholder='[["Id","padj","Sig","xvals","-log10(yvals)"],["GENE1",0.001,"FC_P",1.5,9.2]]'></textarea>
  </label>
  <label>Column types <span>Col=type,… or JSON — overrides inferred types</span>
    <input type="text" id="sel-types" placeholder="Expression=numeric,CellType=factor,Gene=string">
  </label>
  <div class="section-label">Optional</div>
  <label>Intent <span>plain English — activates clinical keyword boosts</span>
    <input type="text" id="sel-intent" placeholder="e.g. show expression distribution by cell type">
  </label>
  <label style="display:inline-block;width:auto">Number of rows <span style="font-weight:400;color:#888">(overrides row count inferred from data)</span></label>
  <input type="text" id="sel-nsamples" placeholder="e.g. 500" style="width:120px;margin-left:8px">
</div>

<!-- ── EXPLAIN ──────────────────────────────────────────────────────── -->
<div class="panel card" id="panel-explain">
  <p class="hint">Returns a plain English explanation of any CanvasXpress config property.</p>
  <label>Property name <span>required</span>
    <input type="text" id="ex-prop" placeholder="e.g. colorScheme, groupingFactors, decorations, filterData">
  </label>
</div>

<!-- ── EXPLAIN R ────────────────────────────────────────────────────── -->
<div class="panel card" id="panel-explain-r">
  <p class="hint">Usage guide for CanvasXpress in R. Leave topic blank for the full guide.</p>
  <label>Topic <span>optional</span>
    <select id="exr-topic">
      <option value="">— all topics —</option>
      <option value="installation">installation</option>
      <option value="basic">basic</option>
      <option value="data">data</option>
      <option value="config">config</option>
      <option value="shiny">shiny</option>
      <option value="rmarkdown">rmarkdown</option>
    </select>
  </label>
</div>

<!-- ── EXPLAIN GGPLOT ───────────────────────────────────────────────── -->
<div class="panel card" id="panel-explain-ggplot">
  <p class="hint">Usage guide for the CanvasXpress ggplot2 bridge. Leave topic blank for the full guide.</p>
  <label>Topic <span>optional</span>
    <select id="exg-topic">
      <option value="">— all topics —</option>
      <option value="installation">installation</option>
      <option value="geoms">geoms</option>
      <option value="example">example</option>
    </select>
  </label>
</div>

<!-- ── MINIMAL PARAMS ───────────────────────────────────────────────── -->
<div class="panel card" id="panel-minimal-params">
  <p class="hint">Returns the minimal set of required parameters for a specific chart type.</p>
  <label>Graph type <span>required</span><input type="text" id="mp-graph" placeholder="e.g. Scatter2D, Heatmap, KaplanMeier, BarLine"></label>
</div>

<!-- ── Actions + URL + Result ───────────────────────────────────────── -->
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

document.querySelectorAll('.tab').forEach(function(t) {
  t.addEventListener('click', function() {
    document.querySelectorAll('.tab, .panel').forEach(function(el){ el.classList.remove('active'); });
    t.classList.add('active');
    document.getElementById('panel-' + t.dataset.tab).classList.add('active');
    activeTab = t.dataset.tab;
    buildUrl();
  });
});

function v(id){ var el=document.getElementById(id); return el?el.value.trim():''; }
function chk(id){ var el=document.getElementById(id); return el?el.checked:false; }

function buildUrl() {
  var p = new URLSearchParams(), url = BASE + '/';
  if (activeTab==='generate') {
    url+='generate';
    if(v('g-desc'))  p.set('description', v('g-desc'));
    if(v('g-headers'))p.set('headers',     v('g-headers'));
    if(v('g-types'))  p.set('column_types',v('g-types'));
    if(v('g-data'))   p.set('data',        v('g-data'));
    if(v('g-temp')&&v('g-temp')!=='0') p.set('temperature',v('g-temp'));
  } else if (activeTab==='modify') {
    url+='modify';
    if(v('m-config'))  p.set('config',      v('m-config'));
    if(v('m-instr'))   p.set('instruction', v('m-instr'));
    if(v('m-headers')) p.set('headers',     v('m-headers'));
    if(v('m-types'))   p.set('column_types',v('m-types'));
    if(v('m-data'))    p.set('data',        v('m-data'));
  } else if (activeTab==='km') {
    url+='km';
    if(v('km-desc'))   p.set('description',v('km-desc'));
    if(v('km-headers'))p.set('headers',    v('km-headers'));
    if(v('km-data'))   p.set('data',       v('km-data'));
    if(v('km-config')) p.set('config',     v('km-config'));
    if(v('km-temp')&&v('km-temp')!=='0') p.set('temperature',v('km-temp'));
  } else if (activeTab==='params') {
    url+='params';
    if(v('p-graph')) p.set('graph_type',v('p-graph'));
    if(v('p-param')) p.set('param_name',v('p-param'));
    if(chk('p-refresh')) p.set('refresh','true');
  } else if (activeTab==='axes') {
    url+='axes';
    if(v('ax-graph')) p.set('graph_type',v('ax-graph'));
  } else if (activeTab==='select') {
    url+='select';
    if(v('sel-intent'))   p.set('intent',      v('sel-intent'));
    if(v('sel-types'))    p.set('column_types',v('sel-types'));
    if(v('sel-data'))     p.set('data',        v('sel-data'));
    if(v('sel-nsamples')) p.set('n_samples',   v('sel-nsamples'));
  } else if (activeTab==='explain') {
    url+='explain';
    if(v('ex-prop')) p.set('property',v('ex-prop'));
  } else if (activeTab==='explain-r') {
    url+='explain-r';
    if(v('exr-topic')) p.set('topic',v('exr-topic'));
  } else if (activeTab==='explain-ggplot') {
    url+='explain-ggplot';
    if(v('exg-topic')) p.set('topic',v('exg-topic'));
  } else if (activeTab==='minimal-params') {
    url+='minimal-params';
    if(v('mp-graph')) p.set('graph_type',v('mp-graph'));
  }
  var qs=p.toString(); url=qs?url+'?'+qs:url;
  document.getElementById('url-box').textContent=url;
  return url;
}

['g-desc','g-headers','g-types','g-data','g-temp',
 'm-config','m-instr','m-headers','m-types','m-data',
 'km-desc','km-headers','km-data','km-config','km-temp',
 'p-graph','p-param','ax-graph',
 'sel-intent','sel-types','sel-data','sel-nsamples',
 'ex-prop','mp-graph'].forEach(function(id){
  var el=document.getElementById(id);
  if(el) el.addEventListener('input',buildUrl);
});
['p-refresh'].forEach(function(id){
  var el=document.getElementById(id);
  if(el) el.addEventListener('change',buildUrl);
});
['exr-topic','exg-topic'].forEach(function(id){
  var el=document.getElementById(id);
  if(el) el.addEventListener('change',buildUrl);
});

function copyUrl(){
  var url=buildUrl();
  navigator.clipboard.writeText(url).catch(function(){});
  var btn=document.querySelector('.btn-secondary');
  btn.textContent='Copied!';
  setTimeout(function(){ btn.innerHTML='&#128279; Copy URL'; },1500);
}

var TITLE_MAP={
  'generate':'Generated Config','modify':'Modified Config','km':'KM Config',
  'params':'Parameter Schema','axes':'Axis Info','select':'Chart Recommendation',
  'explain':'Property Explanation','explain-r':'R Guide',
  'explain-ggplot':'ggplot2 Guide','minimal-params':'Minimal Parameters'
};

async function submit(){
  var url=buildUrl();
  var resultEl=document.getElementById('result');
  var preEl=document.getElementById('result-pre');
  var metaEl=document.getElementById('result-meta');
  var titleEl=document.getElementById('result-title');
  resultEl.style.display='block';
  preEl.textContent='Loading\u2026';
  metaEl.textContent='';
  titleEl.textContent=TITLE_MAP[activeTab]||'Result';
  try {
    var resp=await fetch(url);
    var data=await resp.json();
    if(data.error){
      metaEl.innerHTML='<span class="badge invalid">Error '+resp.status+'</span>';
      preEl.textContent=data.error; return;
    }
    if(['generate','modify','km'].indexOf(activeTab)!==-1){
      var cfg=data.config||{}, valid=data.valid, warns=data.warnings||[], errs=data.errors||[];
      var badge=(valid&&!errs.length)?'<span class="badge valid">\u2713 valid</span>':'<span class="badge invalid">\u2717 issues</span>';
      var gt=cfg.graphType||'?', hdr=(data.headers_used||[]).join(', ')||'\u2014';
      metaEl.innerHTML='graphType: <b>'+gt+'</b>'+badge+' &nbsp; headers: '+hdr;
      if(warns.length) metaEl.innerHTML+='<br><span class="badge warn">\u26a0 '+warns.join(' | ')+'</span>';
      if(errs.length)  metaEl.innerHTML+='<br><span class="badge invalid">\u2716 '+errs.join(' | ')+'</span>';
      if(activeTab==='modify'&&data.changes){
        var c=data.changes;
        metaEl.innerHTML+='<br>added: '+((c.added||[]).join(', ')||'none')+
          ' &nbsp; removed: '+((c.removed||[]).join(', ')||'none')+
          ' &nbsp; changed: '+((c.changed||[]).join(', ')||'none');
      }
      if(activeTab==='km'&&data.column_detection){
        var cd=data.column_detection;
        metaEl.innerHTML+='<br>time: <b>'+(cd.time_col||'?')+'</b>'+
          ' &nbsp; event: <b>'+(cd.event_col||'?')+'</b>'+
          ' &nbsp; groups: <b>'+(cd.group_cols||[]).join(', ')+'</b>'+
          ' &nbsp; confidence: <b>'+cd.confidence+'</b>';
      }
      if(data.removed_params&&data.removed_params.length)
        metaEl.innerHTML+='<br><span class="badge warn">stripped: '+data.removed_params.join(', ')+'</span>';
      preEl.textContent=JSON.stringify(activeTab==='km'?data:cfg,null,2);
    } else if(activeTab==='params'){
      var count=data.param_count||(data.params?Object.keys(data.params).length:'?');
      metaEl.innerHTML=count+' parameters &nbsp; source: <b>'+(data.schema_source||'?')+'</b>';
      preEl.textContent=JSON.stringify(data,null,2);
    } else if(activeTab==='select'){
      var top=data.recommendations&&data.recommendations[0];
      if(top) metaEl.innerHTML='Top recommendation: <b>'+top.graphType+'</b>';
      preEl.textContent=JSON.stringify(data,null,2);
    } else {
      preEl.textContent=JSON.stringify(data,null,2);
    }
  } catch(e){ preEl.textContent='Request failed: '+e; }
}

(function restoreFromUrl(){
  var p=new URLSearchParams(window.location.search);
  var tab=p.get('_tab')||'generate';
  var tabEl=document.querySelector('[data-tab='+tab+']');
  if(tabEl) tabEl.click();
  var map={'g-desc':'description','g-headers':'headers','g-types':'column_types',
    'g-data':'data','g-temp':'temperature','m-config':'config','m-instr':'instruction',
    'm-headers':'headers','m-types':'column_types','m-data':'data',
    'km-desc':'description','km-headers':'headers','km-data':'data',
    'km-config':'config','km-temp':'temperature','p-graph':'graph_type',
    'p-param':'param_name','ax-graph':'graph_type','sel-intent':'intent',
    'sel-types':'column_types','sel-nsamples':'n_samples','ex-prop':'property',
    'mp-graph':'graph_type'};
  Object.keys(map).forEach(function(id){
    var val=p.get(map[id]), el=document.getElementById(id);
    if(val&&el) el.value=val;
  });
  if(p.get('refresh')==='true'){var el=document.getElementById('p-refresh');if(el)el.checked=true;}
  if(p.get('topic')){['exr-topic','exg-topic'].forEach(function(id){var el=document.getElementById(id);if(el)el.value=p.get('topic');});}
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
        return _cx_response({
            "config": {}, "valid": False, "success": False,
            "warnings": ["Could not generate configuration: " + str(exc)],
            "invalid_refs": {}, "headers_used": [], "types_used": {}, "removed_params": [],
        }, cx, 200)
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
        return _cx_response({
            "config": {}, "valid": False, "success": False,
            "warnings": ["Could not modify configuration: " + str(exc)],
            "invalid_refs": {}, "headers_used": [], "types_used": {}, "removed_params": [],
        }, cx, 200)
    return _cx_response(result, cx)



@mcp.custom_route("/km", methods=["GET", "POST"])
async def rest_km(request: Request) -> Response:
    """REST endpoint for generate_km_config."""
    kwargs, status, err = await _kwargs_from_request(request, require_description=False)
    cx = kwargs.pop("_cx", {})
    if status != 200:
        return _cx_response({"error": err, "success": False}, cx, status)
    km_args = {}
    if "description" in kwargs: km_args["description"] = kwargs["description"]
    if "headers"     in kwargs: km_args["headers"]      = kwargs["headers"]
    if "data"        in kwargs: km_args["data"]         = kwargs["data"]
    if "config"      in kwargs: km_args["config"]       = kwargs["config"]
    if "temperature" in kwargs: km_args["temperature"]  = kwargs["temperature"]
    if not km_args:
        return _cx_response({"error": "At least one of description, headers, data, or config is required.", "success": False}, cx, 400)
    try:
        result = generate_km_config(**km_args)
    except Exception as exc:
        log.exception("REST /km error")
        return _cx_response({"config": {}, "valid": False, "success": False,
            "warnings": ["Could not generate KM configuration: " + str(exc)],
            "errors": [], "suggestions": [], "column_detection": None}, cx, 200)
    return _cx_response(result, cx)


@mcp.custom_route("/params", methods=["GET", "POST"])
async def rest_params(request: Request) -> Response:
    """REST endpoint for query_canvasxpress_params."""
    if request.method == "GET":
        p = dict(request.query_params)
    else:
        ct = request.headers.get("content-type", "")
        p = await request.json() if "application/json" in ct else dict(await request.form())
    try:
        result = query_canvasxpress_params(
            graph_type=p.get("graph_type") or None,
            param_name=p.get("param_name") or None,
            refresh=p.get("refresh", "").lower() == "true",
        )
    except Exception as exc:
        log.exception("REST /params error")
        return JSONResponse({"error": str(exc)}, status_code=500)
    return JSONResponse(result)


@mcp.custom_route("/axes", methods=["GET", "POST"])
async def rest_axes(request: Request) -> Response:
    """REST endpoint for get_axes_info."""
    if request.method == "GET":
        p = dict(request.query_params)
    else:
        ct = request.headers.get("content-type", "")
        p = await request.json() if "application/json" in ct else dict(await request.form())
    gt = (p.get("graph_type") or "").strip()
    if not gt:
        return JSONResponse({"error": "'graph_type' is required"}, status_code=400)
    try:
        result = get_axes_info(gt)
    except Exception as exc:
        log.exception("REST /axes error")
        return JSONResponse({"error": str(exc)}, status_code=500)
    return JSONResponse(result)


@mcp.custom_route("/select", methods=["GET", "POST"])
async def rest_select(request: Request) -> Response:
    """
    REST endpoint for select_canvasxpress_chart.

    Accepts either:
      - column_types  (explicit "Col=type,…" or JSON object)
      - data          (JSON array of arrays — first row is headers; types are inferred)
      or both (explicit column_types take precedence over inferred types).

    Query / body parameters:
      intent        (str, required) — plain English description of what you want to show.
      column_types  (str)           — "Col=type,…" or JSON object. Alias: types.
      data          (str|array)     — JSON array of arrays; first row = headers.
                                      Column types are inferred from the values.
      n_samples     (int)           — override row count (default: inferred from data).
    """
    if request.method == "GET":
        p = dict(request.query_params)
    else:
        ct = request.headers.get("content-type", "")
        p = await request.json() if "application/json" in ct else dict(await request.form())

    intent = (p.get("intent") or "").strip()
    # intent is optional — Layer 1 (structural) and Layer 2 (semantic column names)
    # work without it; intent only activates Layer 3 keyword boosts.

    # --- resolve column_types and n_samples ---
    column_types: dict[str, str] = {}
    n_samples: int | None = None
    inferred_headers: list[str] = []
    type_source = "explicit"

    # 1. Parse data array if provided → infer types
    raw_data = p.get("data")
    if raw_data:
        try:
            data_arr = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
            inferred_types, inferred_headers, n_rows = _infer_column_types(data_arr)
            column_types = inferred_types
            n_samples = n_rows
            type_source = "inferred"
        except Exception as exc:
            return JSONResponse({"error": f"Could not parse 'data': {exc}"}, status_code=400)

    # 2. Explicit column_types override / supplement inferred types
    for key in ("column_types", "types"):
        raw_types = (p.get(key) or "").strip()
        if raw_types:
            explicit = _parse_col_types(raw_types)
            column_types.update(explicit)  # explicit wins
            type_source = "explicit" if not inferred_headers else "merged"
            break

    if not column_types:
        return JSONResponse(
            {"error": "Provide 'column_types' (e.g. Gene=string,Expr=numeric) or 'data' (JSON array of arrays)."},
            status_code=400,
        )

    # 3. n_samples override
    try:
        if p.get("n_samples"):
            n_samples = int(p["n_samples"])
    except (ValueError, TypeError):
        pass

    try:
        result = select_canvasxpress_chart(
            intent=intent,
            column_types=column_types,
            n_samples=n_samples,
        )
        # Annotate with provenance for transparency
        result["type_source"] = type_source
        if inferred_headers:
            result["headers_detected"] = inferred_headers
    except Exception as exc:
        log.exception("REST /select error")
        return JSONResponse({"error": str(exc)}, status_code=500)
    return JSONResponse(result)


@mcp.custom_route("/explain", methods=["GET", "POST"])
async def rest_explain(request: Request) -> Response:
    """REST endpoint for explain_config_property."""
    if request.method == "GET":
        p = dict(request.query_params)
    else:
        ct = request.headers.get("content-type", "")
        p = await request.json() if "application/json" in ct else dict(await request.form())
    prop = (p.get("property") or "").strip()
    if not prop:
        return JSONResponse({"error": "'property' is required"}, status_code=400)
    try:
        result = explain_config_property(prop)
    except Exception as exc:
        log.exception("REST /explain error")
        return JSONResponse({"error": str(exc)}, status_code=500)
    return JSONResponse({"property": prop, "explanation": result})


@mcp.custom_route("/explain-r", methods=["GET", "POST"])
async def rest_explain_r(request: Request) -> Response:
    """REST endpoint for explain_canvasxpress_r."""
    if request.method == "GET":
        p = dict(request.query_params)
    else:
        ct = request.headers.get("content-type", "")
        p = await request.json() if "application/json" in ct else dict(await request.form())
    try:
        result = explain_canvasxpress_r(topic=p.get("topic") or None)
    except Exception as exc:
        log.exception("REST /explain-r error")
        return JSONResponse({"error": str(exc)}, status_code=500)
    return JSONResponse(result)


@mcp.custom_route("/explain-ggplot", methods=["GET", "POST"])
async def rest_explain_ggplot(request: Request) -> Response:
    """REST endpoint for explain_canvasxpress_ggplot."""
    if request.method == "GET":
        p = dict(request.query_params)
    else:
        ct = request.headers.get("content-type", "")
        p = await request.json() if "application/json" in ct else dict(await request.form())
    try:
        result = explain_canvasxpress_ggplot(topic=p.get("topic") or None)
    except Exception as exc:
        log.exception("REST /explain-ggplot error")
        return JSONResponse({"error": str(exc)}, status_code=500)
    return JSONResponse(result)


@mcp.custom_route("/minimal-params", methods=["GET", "POST"])
async def rest_minimal_params(request: Request) -> Response:
    """REST endpoint for get_minimal_parameters."""
    if request.method == "GET":
        p = dict(request.query_params)
    else:
        ct = request.headers.get("content-type", "")
        p = await request.json() if "application/json" in ct else dict(await request.form())
    gt = (p.get("graph_type") or "").strip()
    if not gt:
        return JSONResponse({"error": "'graph_type' is required"}, status_code=400)
    try:
        result = get_minimal_parameters(gt)
    except Exception as exc:
        log.exception("REST /minimal-params error")
        return JSONResponse({"error": str(exc)}, status_code=500)
    return JSONResponse(result)

@mcp.custom_route("/feedback", methods=["POST"])
async def rest_feedback(request: Request) -> Response:
    """
    Submit thumbs-up / thumbs-down feedback for a previous tool call.

    POST /feedback
    Body (JSON):
      request_id  (str, required)  — UUID returned in the tool response
      rating      (int, required)  — 1 = thumbs up, -1 = thumbs down
      comment     (str, optional)  — free-text explanation
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "JSON body required"}, status_code=400)

    call_id = (body.get("request_id") or "").strip()
    if not call_id:
        return JSONResponse({"error": "'request_id' is required"}, status_code=400)

    rating = body.get("rating")
    if rating not in (1, -1):
        return JSONResponse({"error": "'rating' must be 1 (up) or -1 (down)"}, status_code=400)

    comment = body.get("comment") or None
    found = _call_log.rate(call_id, rating, comment)
    if not found:
        return JSONResponse({"error": f"No call found with request_id '{call_id}'"}, status_code=404)

    log.info("Feedback received: request_id=%s rating=%s", call_id, rating)
    return JSONResponse({"success": True, "request_id": call_id, "rating": rating})


@mcp.custom_route("/feedback/purge", methods=["POST"])
async def rest_feedback_purge(request: Request) -> Response:
    """
    Purge rows from the call log.  Requires ``X-Admin-Key`` header.

    POST /feedback/purge
    Headers:
      X-Admin-Key  (str, required)  — must match ADMIN_KEY in .env
    Body (JSON, all optional):
      tool        (str)   — delete only rows for this tool name
      rated_only  (bool)  — if true, delete only rows that have a rating
    Omitting both deletes ALL rows.
    """
    deny = _require_admin_key(request)
    if deny:
        return deny

    try:
        body = await request.json()
    except Exception:
        body = {}

    tool_filter = (body.get("tool") or "").strip() or None
    rated_only  = bool(body.get("rated_only", False))

    deleted = _call_log.purge(tool=tool_filter, rated_only=rated_only)
    log.info("Purged %d call-log rows (tool=%s rated_only=%s)", deleted, tool_filter, rated_only)
    return JSONResponse({"success": True, "deleted": deleted})


@mcp.custom_route("/feedback/export", methods=["GET", "POST"])
async def rest_feedback_export(request: Request) -> Response:
    """
    Export logged tool calls with optional filters.

    GET /feedback/export?tool=generate_canvasxpress_config&rated_only=true&limit=100
    Parameters:
      tool        (str)   — filter by tool name (optional)
      rated_only  (bool)  — if "true", return only calls that have a rating
      limit       (int)   — max rows to return (default 500)
    """
    p = dict(request.query_params)
    if request.method == "POST":
        try:
            p.update(await request.json())
        except Exception:
            pass

    tool_filter  = (p.get("tool") or "").strip() or None
    rated_only   = str(p.get("rated_only", "")).lower() == "true"
    try:
        limit = int(p.get("limit", 500))
    except (ValueError, TypeError):
        limit = 500

    rows = _call_log.export(tool=tool_filter, rated_only=rated_only, limit=limit)
    return JSONResponse({"count": len(rows), "rows": rows})


@mcp.custom_route("/ui", methods=["GET"])
async def rest_ui(request: Request) -> HTMLResponse:
    """Serve the browser-based form UI at /ui."""
    return HTMLResponse(_UI_HTML)


@mcp.custom_route("/favicon.ico", methods=["GET"])
async def rest_favicon(request: Request) -> Response:
    """Serve the CanvasXpress favicon from the bundled file."""
    import pathlib
    favicon = pathlib.Path(__file__).parent / "favicon.png"
    data = favicon.read_bytes()
    return Response(
        content=data,
        status_code=200,
        media_type="image/png",
        headers={"Cache-Control": "no-cache"},
    )


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
