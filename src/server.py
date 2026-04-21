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
    # Map chart — must come AFTER heatmap/treemap entries (shorter, would shadow them)
    "choropleth": "Map", "geographic map": "Map", "world map": "Map",
    "country map": "Map", "state map": "Map", "continent map": "Map",
    "map of": "Map",
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


# ---------------------------------------------------------------------------
# Map ID inference
# ---------------------------------------------------------------------------

_MAP_ID_PATTERNS: list[tuple[str, str]] = [
    # Regions / multi-country (check before individual countries)
    ("world continent",           "WorldContinents"),
    ("worldcontinent",            "WorldContinents"),
    ("world medium",              "WorldMedium"),
    ("worldmedium",               "WorldMedium"),
    ("world high",                "WorldHigh"),
    ("worldhigh",                 "WorldHigh"),
    ("north america",             "NorthAmerica"),
    ("south america",             "SouthAmerica"),
    ("oceania",                   "Oceania"),
    ("africa",                    "Africa"),
    ("europe",                    "Europe"),
    ("asia",                      "Asia"),
    ("all countries",             "Countries"),
    ("countries",                 "Countries"),
    ("world",                     "World"),
    # US Albers projection (explicit request only — pie overlays work on any map)
    ("albers pie",              "albersStatesPie"),
    ("albersstatespie",         "albersStatesPie"),
    ("albers usa",              "albersStatesPie"),
    ("albers state",            "albersStatesPie"),
    # US sub-regions
    ("usa congressional district","USADistricts"),
    ("us congressional district", "USADistricts"),
    ("congressional district",    "USADistricts"),
    ("usa district",              "USADistricts"),
    ("us district",               "USADistricts"),
    ("usa county",                "USACounties"),
    ("us county",                 "USACounties"),
    ("usa state",                 "USAStates"),
    ("us state",                  "USAStates"),
    ("united states state",       "USAStates"),
    ("united states",             "USAStates"),
    # US states (full names)
    ("california",         "CA"),
    ("new york",           "NY"),
    ("texas",              "TX"),
    ("florida",            "FL"),
    ("washington",         "WA"),
    ("illinois",           "IL"),
    ("ohio",               "OH"),
    ("georgia",            "GA"),
    ("michigan",           "MI"),
    ("pennsylvania",       "PA"),
    ("north carolina",     "NC"),
    ("new jersey",         "NJ"),
    ("virginia",           "VA"),
    ("arizona",            "AZ"),
    ("colorado",           "CO"),
    ("oregon",             "OR"),
    # Countries (full names)
    ("canada",             "CAN"),
    ("australia",          "AUS"),
    ("united kingdom",     "GBR"),
    ("great britain",      "GBR"),
    ("brazil",             "BRA"),
    ("mexico",             "MEX"),
    ("france",             "FRA"),
    ("germany",            "DEU"),
    ("china",              "CHN"),
    ("japan",              "JPN"),
    ("india",              "IND"),
    ("spain",              "ESP"),
    ("italy",              "ITA"),
    ("russia",             "RUS"),
    ("argentina",          "ARG"),
    ("south africa",       "ZAF"),
    ("new zealand",        "NZL"),
    ("netherlands",        "NLD"),
    ("sweden",             "SWE"),
    ("norway",             "NOR"),
    ("denmark",            "DNK"),
    ("finland",            "FIN"),
    ("switzerland",        "CHE"),
    ("portugal",           "PRT"),
    ("poland",             "POL"),
    # Fallback — generic map requests become the world map
    ("usa",                "USA"),
    ("u.s.a",              "USA"),
]


def _infer_map_id(description: str) -> Optional[str]:
    """Try to extract a CanvasXpress mapId from a natural-language description."""
    desc_lower = description.lower()
    for pattern, map_id in _MAP_ID_PATTERNS:
        if pattern in desc_lower:
            return map_id
    return None


# Map mapId → human-readable description of what the first column's ID values should be
_MAP_ID_COLUMN_HINT: dict[str, str] = {
    "World":          "ISO 3-letter country codes (e.g. \"ALB\", \"ARG\", \"AUS\", \"BRA\", \"CAN\", \"CHN\", \"DEU\", \"ESP\", \"FRA\", \"GBR\", \"IND\", \"ITA\", \"JPN\", \"MEX\", \"RUS\", \"USA\", \"ZAF\")",
    "WorldMedium":    "ISO 3-letter country codes (same as World — medium resolution)",
    "WorldHigh":      "ISO 3-letter country codes (same as World — high resolution)",
    "Countries":      "ISO 3-letter country codes (same as World — e.g. \"ALB\", \"ARG\", \"AUS\", \"BRA\", \"CAN\", \"CHN\", \"DEU\", \"ESP\", \"FRA\", \"GBR\", \"IND\", \"ITA\", \"JPN\", \"MEX\", \"RUS\", \"USA\", \"ZAF\")",
    "WorldContinents":"continent names (\"Africa\", \"Asia\", \"Europe\", \"NorthAmerica\", \"Oceania\", \"SouthAmerica\")",
    "Africa":         "ISO 3-letter codes for African countries (e.g. \"DZA\", \"EGY\", \"ETH\", \"GHA\", \"KEN\", \"MAR\", \"MOZ\", \"NGA\", \"ZAF\", \"TZA\", \"UGA\", \"ZMB\")",
    "Asia":           "ISO 3-letter codes for Asian countries (e.g. \"BGD\", \"CHN\", \"IDN\", \"IND\", \"IRN\", \"IRQ\", \"JPN\", \"KAZ\", \"KOR\", \"MYS\", \"PAK\", \"PHL\", \"SAU\", \"THA\", \"TUR\", \"VNM\")",
    "Europe":         "ISO 3-letter codes for European countries (e.g. \"AUT\", \"BEL\", \"CHE\", \"CZE\", \"DEU\", \"DNK\", \"ESP\", \"FIN\", \"FRA\", \"GBR\", \"GRC\", \"HUN\", \"ITA\", \"NLD\", \"NOR\", \"POL\", \"PRT\", \"ROU\", \"RUS\", \"SWE\", \"UKR\")",
    "NorthAmerica":   "ISO 3-letter codes for North American countries (e.g. \"CAN\", \"CRI\", \"CUB\", \"DOM\", \"GTM\", \"HND\", \"HTI\", \"JAM\", \"MEX\", \"NIC\", \"PAN\", \"SLV\", \"TTO\", \"USA\")",
    "SouthAmerica":   "ISO 3-letter codes for South American countries (e.g. \"ARG\", \"BOL\", \"BRA\", \"CHL\", \"COL\", \"ECU\", \"GUY\", \"PER\", \"PRY\", \"SUR\", \"URY\", \"VEN\")",
    "Oceania":        "ISO 3-letter codes for Oceania countries (e.g. \"AUS\", \"FJI\", \"NZL\", \"PNG\", \"SLB\", \"VUT\")",
    "USAStates":      "2-letter US state/territory codes (e.g. \"AL\", \"AK\", \"AZ\", \"AR\", \"CA\", \"CO\", \"CT\", \"DE\", \"FL\", \"GA\", \"HI\", \"ID\", \"IL\", \"IN\", \"IA\", \"KS\", \"KY\", \"LA\", \"ME\", \"MD\", \"MA\", \"MI\", \"MN\", \"MS\", \"MO\", \"MT\", \"NE\", \"NV\", \"NH\", \"NJ\", \"NM\", \"NY\", \"NC\", \"ND\", \"OH\", \"OK\", \"OR\", \"PA\", \"RI\", \"SC\", \"SD\", \"TN\", \"TX\", \"UT\", \"VT\", \"VA\", \"WA\", \"WV\", \"WI\", \"WY\", \"DC\")",
    "albersStatesPie":"2-letter US state codes (same as USAStates — used for Albers pie maps)",
    "USACounties":    "5-digit FIPS county codes (e.g. \"01001\" for Autauga AL, \"06037\" for Los Angeles CA, \"48113\" for Dallas TX)",
    # Country sub-region values — use mapPropertyId in config to specify the matching feature property
    "CAN":            "Canadian province/territory names — set mapPropertyId: \"prov_name_en\" in config (e.g. \"Alberta\", \"British Columbia\", \"Manitoba\", \"New Brunswick\", \"Newfoundland and Labrador\", \"Northwest Territories\", \"Nova Scotia\", \"Nunavut\", \"Ontario\", \"Prince Edward Island\", \"Quebec\", \"Saskatchewan\", \"Yukon\")",
    "USA":            "2-letter US state codes (same as USAStates — e.g. \"CA\", \"TX\", \"NY\", \"FL\", \"WA\")",
    "GBR":            "UK district/county HASC codes — set mapPropertyId: \"HASC_2\" in config (e.g. \"GB.BA\", \"GB.BN\", \"GB.BD\", \"GB.LN\", \"GB.KE\", \"GB.EX\" — 192 districts); or district names with mapPropertyId: \"NAME_2\"",
    "AUS":            "Australian state/territory names — set mapPropertyId: \"STATE_NAME\" in config (e.g. \"New South Wales\", \"Victoria\", \"Queensland\", \"South Australia\", \"Western Australia\", \"Tasmania\", \"Northern Territory\", \"Australian Capital Territory\")",
}


def _map_id_column_hint(map_id: str) -> str:
    """Return a hint string describing the expected ID values for the first column of a map dataset."""
    hint = _MAP_ID_COLUMN_HINT.get(map_id)
    if hint:
        return hint
    # For US 2-letter state codes used as map_id (e.g. "CA", "TX") — county-level maps
    if len(map_id) == 2 and map_id.isupper():
        return f"county names or FIPS codes within the state of {map_id}"
    # For ISO 3-letter country codes not explicitly listed
    if len(map_id) == 3 and map_id.isupper():
        return (
            f"feature property values from https://canvasxpress.org/data/maps/{map_id}.json"
            f" — set mapPropertyId in config to specify which feature property to match against"
        )
    return "geographic ID codes matching the map's features"


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
  Map:       ALWAYS include mapId (REQUIRED — never omit it).
             Infer mapId from the description using these rules:
               World map / global         → "World"
               World continents           → "WorldContinents"
               Africa                     → "Africa"
               Asia                       → "Asia"
               Europe                     → "Europe"
               North America              → "NorthAmerica"
               South America              → "SouthAmerica"
               Oceania / Pacific          → "Oceania"
               US / United States (all)   → "USAStates"
               US counties                → "USACounties"
               Albers USA (explicit only)  → "albersStatesPie"
                 Use ONLY when "Albers" projection is explicitly requested.
                 Pie overlays work on ANY mapId — do NOT change mapId just because
                 pie charts are requested.

             PIE OVERLAYS — use decorations.pie for any map that shows pie charts
             per region (e.g. "show party breakdown as pies", "pie for each state",
             "pie chart per country"):
               decorations.pie.smps   = [list of numeric columns for pie slices]
               decorations.pie.colors = [one color per slice, optional]
               decorations.pie.size   = 2.5 (float multiplier for pie size, optional)
             sizeBy is a TOP-LEVEL config key (NOT inside decorations) — it is the column
             name whose numeric values scale the pie/symbol size per region.
             e.g. config = { "sizeBy": "Total", "decorations": { "pie": { "smps": [...] } } }
             Do NOT put slice columns in xAxis — they belong only in decorations.pie.smps.
             Pie overlays are independent of mapId and mapProjection.
               Country by name or code    → ISO 3-letter code (Canada→"CAN", UK→"GBR",
                                            Mexico→"MEX", France→"FRA", Germany→"DEU",
                                            Australia→"AUS", Brazil→"BRA", India→"IND",
                                            China→"CHN", Japan→"JPN", Spain→"ESP",
                                            Italy→"ITA", Russia→"RUS", Argentina→"ARG")
               US state by name or code   → 2-letter code (California→"CA", Texas→"TX",
                                            New York→"NY", Florida→"FL", Washington→"WA")

             MAP DATA FORMAT — when columns/headers are provided:
             The FIRST column in the dataset is always the geographic ID column.
             Its values identify map components (the "features") by their standard codes.
             CanvasXpress automatically matches these IDs to the map geometry.
             The remaining columns contain numeric values to display on the map.

             ID column values by map type:
               mapId="World" / continent regions
                 → ISO 3-letter country codes: "ALB", "ARG", "ARM", "AUS", "AUT",
                   "BEL", "BRA", "CAN", "CHE", "CHN", "DEU", "ESP", "FRA", "GBR",
                   "IND", "ITA", "JPN", "KOR", "MEX", "NLD", "NOR", "POL", "PRT",
                   "RUS", "SWE", "TUR", "UKR", "USA", "ZAF", etc.
               mapId="WorldContinents"
                 → Continent names: "Africa", "Asia", "Europe", "NorthAmerica",
                   "Oceania", "SouthAmerica"
               mapId="USAStates"
                 → 2-letter US state FIPS codes: "AL", "AK", "AZ", "AR", "CA",
                   "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA",
                   "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO",
                   "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH",
                   "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT",
                   "VA", "WA", "WV", "WI", "WY", "DC"
               mapId="USACounties"
                 → 5-digit FIPS county codes: "01001", "06037", "48113", etc.
               mapId=<country ISO3> (e.g. "CAN", "GBR", "AUS")
                 → feature property values from the map file at
                   https://canvasxpress.org/data/maps/<ISO3>.json
                   Set mapPropertyId in config to specify which feature property to match.
                   Canada (CAN) — use mapPropertyId: "prov_name_en"
                     e.g. "Alberta", "British Columbia", "Manitoba", "New Brunswick",
                          "Newfoundland and Labrador", "Northwest Territories", "Nova Scotia",
                          "Nunavut", "Ontario", "Prince Edward Island", "Quebec",
                          "Saskatchewan", "Yukon"
                   Australia (AUS) — use mapPropertyId: "STATE_NAME"
                     e.g. "New South Wales", "Victoria", "Queensland", "South Australia",
                          "Western Australia", "Tasmania", "Northern Territory",
                          "Australian Capital Territory"
                   UK (GBR) — use mapPropertyId: "HASC_2" for 192 districts
                     e.g. "GB.BA", "GB.BN", "GB.BD", "GB.LN", "GB.KE", "GB.EX"
                     or mapPropertyId: "NAME_2" for district names
               mapId=<US state code> (e.g. "CA", "TX", "NY")
                 → County names or FIPS codes for that state's counties

             IMPORTANT: Do NOT use the first column as xAxis, yAxis, or groupingFactors.
             The ID column is consumed automatically by CanvasXpress map rendering.
             Do NOT include xAxis, yAxis, or groupingFactors for Map charts.
             For pie overlays: do NOT put slice columns in xAxis —
             they belong only in decorations.pie.smps.

             MARKER PINS — when the description mentions specific places, landmarks,
             addresses, ZIP codes, or any named locations to mark on the map, output
             a "_markers_to_geocode" key at the top level of the config (NOT inside
             decorations). Each item in the array must use ONE of:
               {"location": "place name, city, or address", "label": "display label"}
               {"zip": "US ZIP code", "label": "display label"}
             Add optional "color" (default "red") and "shape"
             ("teardrop"|"circle"|"star"|"square", default "teardrop").
             Use "location" for any named place, city, landmark, or address worldwide.
             Use "zip" for US ZIP codes (5 digits).
             Do NOT guess or fabricate lat/lng coordinates — the server will geocode them.
             Example for "markers at the Eiffel Tower, Big Ben, and ZIP 10001":
               "_markers_to_geocode": [
                 {"location": "Eiffel Tower, Paris", "label": "Eiffel Tower", "color": "red"},
                 {"location": "Big Ben, London",      "label": "Big Ben",      "color": "blue"},
                 {"zip": "10001",                     "label": "10001",        "color": "green"}
               ]

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
EXCEPTION — "_markers_to_geocode" is a valid server-side processing key (NOT a CX parameter).
  Include it whenever a Map description mentions named places, landmarks, addresses, or ZIP codes.
  It will be removed from the config before sending to CanvasXpress.

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
        # Detect whether this is a Map chart so we can give the right column guidance
        _detected_gt = detect_graph_type(description)
        _is_map = _detected_gt == "Map"

        if _is_map:
            # For maps the first column is the geographic ID column; remaining columns are values.
            _inferred_mid = _infer_map_id(description)
            _id_hint = _map_id_column_hint(_inferred_mid) if _inferred_mid else "geographic ID codes"
            _id_col  = headers[0]
            _val_cols = headers[1:] if len(headers) > 1 else []
            header_hint = (
                f"\n\nThis is a Map chart. The dataset has these columns: {', '.join(headers)}."
                f"\n   '{_id_col}' is the geographic ID column — its values must be {_id_hint}."
            )
            if _val_cols:
                header_hint += (
                    f"\n   The remaining columns ({', '.join(_val_cols)}) contain numeric values"
                    f" to display on the map."
                )
            header_hint += (
                "\n   Do NOT assign any of these columns to xAxis, yAxis, or groupingFactors."
                "\n   The ID column is consumed automatically by CanvasXpress map rendering."
            )
        elif column_types:
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

    def rate(self, call_id: str, rating: int, comment: str | None = None) -> dict | None:
        """Set rating (-1/1) and optional comment. Returns the row dict or None if not found."""
        with self._lock:
            con = self._connect()
            cur = con.execute(
                "UPDATE tool_calls SET rating=?, comment=? WHERE id=?",
                (rating, comment, call_id),
            )
            con.commit()
            if cur.rowcount == 0:
                con.close()
                return None
            row = con.execute(
                "SELECT tool, request, ts FROM tool_calls WHERE id=?", (call_id,)
            ).fetchone()
            con.close()
        if not row:
            return None
        try:
            req = json.loads(row[1]) if row[1] else {}
        except Exception:
            req = {}
        return {
            "tool":      row[0],
            "target":    req.get("target") or req.get("renderTo") or "",
            "client_id": req.get("client_id") or req.get("client") or "",
            "ts":        row[2],
        }

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
    "/map":            "create_map_config",
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
                    # Detect JSONP: unwrap callback(json); before parsing
                    import re as _re
                    _jsonp_prefix: str | None = None
                    try:
                        _m = _re.match(
                            rb'^([a-zA-Z0-9_.]+)\(([\s\S]*)\);\s*$', raw.strip()
                        )
                        if _m:
                            _jsonp_prefix = _m.group(1).decode("utf-8")
                            raw = _m.group(2)
                    except Exception:
                        pass
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
                        if _jsonp_prefix:
                            raw = f"{_jsonp_prefix}({raw.decode()});".encode("utf-8")

                        # --- Log to SQLite (best-effort, never block response) ---
                        try:
                            req_raw = b"".join(req_body_chunks)
                            try:
                                req_obj = json.loads(req_raw) if req_raw else {}
                            except Exception:
                                req_obj = req_raw.decode(errors="replace")
                            # For GET/JSONP requests the body is empty — also capture query params
                            if not req_obj:
                                from urllib.parse import parse_qs
                                qs = scope.get("query_string", b"").decode(errors="replace")
                                req_obj = {k: v[0] if len(v) == 1 else v
                                           for k, v in parse_qs(qs).items()}
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
                        if _jsonp_prefix:
                            # Re-wrap stripped JSONP if inner JSON parse failed
                            raw = f"{_jsonp_prefix}({raw.decode()});".encode("utf-8")
                        # else pass through unchanged

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
        "Also handles Map (choropleth) charts — include headers where the first column contains "
        "geographic IDs (ISO-3 country codes for world maps, 2-letter state codes for US maps, etc.) "
        "and remaining columns hold the values to display. "
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
                     For Map charts, mention the geographic scope
                     (e.g. 'world map', 'US states', 'map of Canada').
        headers: Optional list of column names from your dataset.
                 e.g. ["Gene", "Sample1", "Sample2", "Treatment"]
                 For Map charts, the first column should be the geographic ID column
                 (ISO-3 country codes for world maps, 2-letter state codes for US maps, etc.)
                 and subsequent columns should contain numeric values to display.
                 e.g. ["Country", "GDP", "Population"] for a world map
                      ["State", "Sales", "Units"]     for a US states map
        data: Optional flat CSV-style array of arrays where the first row
              contains column headers and subsequent rows contain data values.
              e.g. [["Gene","Sample1","Treatment"],["BRCA1",1.2,"Control"]]
              When provided, headers are extracted from row 0 automatically.
              If both headers and data are provided, data takes precedence.
              For Map charts, first-column values must be geographic ID codes:
                World map:  ISO-3 codes like "ALB", "ARG", "AUS", "CAN", "FRA" ...
                US States:  2-letter codes like "CA", "TX", "NY", "FL", "WA" ...
                US Counties: 5-digit FIPS codes like "06037", "48113" ...
                Country sub-regions: feature property values from
                  https://canvasxpress.org/data/maps/<ISO3>.json —
                  set mapPropertyId in config to specify the matching property.
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

    # ── Map post-processing: ensure mapId is always set ─────────────────────
    if graph_type == "Map" and not config.get("mapId"):
        inferred = _infer_map_id(description)
        if inferred:
            config["mapId"] = inferred
            log.info("Inferred mapId '%s' from description", inferred)
        else:
            log.warning("Map config missing mapId and could not infer from description: %s", description)

    # ── Map post-processing: geocode _markers_to_geocode if LLM emitted them ─
    _geocode_warnings: list[str] = []
    raw_markers = config.pop("_markers_to_geocode", None)
    if graph_type == "Map" and raw_markers and isinstance(raw_markers, list):
        log.info("Geocoding %d marker(s) from _markers_to_geocode", len(raw_markers))
        map_result = create_map_config(
            map_id=config.get("mapId", "World"),
            markers=raw_markers,
        )
        geocoded_decos = map_result.get("config", {}).get("decorations")
        if geocoded_decos:
            # Merge with any existing decorations (e.g. pie) from the LLM config
            existing = config.get("decorations", {})
            if isinstance(existing, dict):
                existing.update(geocoded_decos)
            else:
                existing = geocoded_decos
            config["decorations"] = existing
            log.info("Added %d geocoded marker(s) to config", len(geocoded_decos.get("marker", [])))
        if map_result.get("warnings"):
            _geocode_warnings = map_result["warnings"]
            log.warning("Geocoding warnings: %s", _geocode_warnings)

    # ── KaplanMeier post-processing: fix colorBy type + fill missing axes ───
    if graph_type == "KaplanMeier":
        # colorBy must be a plain string, not a list
        color_by = config.get("colorBy")
        if isinstance(color_by, list):
            config["colorBy"] = color_by[0] if color_by else None
            if not color_by:
                config.pop("colorBy", None)
            log.info("KM: converted colorBy list to string: %s", config.get("colorBy"))

        # If headers available and axes are missing/wrong, use heuristic detection
        if resolved_headers and (not config.get("xAxis") or not config.get("yAxis")):
            detection = cx_survival.detect_km_columns(resolved_headers)
            if not config.get("xAxis") and detection.get("time_col"):
                config["xAxis"] = [detection["time_col"]]
                log.info("KM: auto-assigned xAxis=%s from column detection", config["xAxis"])
            if not config.get("yAxis") and detection.get("event_col"):
                config["yAxis"] = [detection["event_col"]]
                log.info("KM: auto-assigned yAxis=%s from column detection", config["yAxis"])
            if not config.get("colorBy") and detection.get("color_cols"):
                config["colorBy"] = detection["color_cols"][0]
                log.info("KM: auto-assigned colorBy=%s from column detection", config["colorBy"])

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
        "warnings":       validation["warnings"] + _geocode_warnings,
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
    category_cardinalities: Optional[dict[str, int]] = None,
    max_level_fractions: Optional[dict[str, float]] = None,
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
                                       category_cardinalities=category_cardinalities,
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

    _bmc_kwargs = dict(
        max_level_fractions=max_level_fractions,
        category_cardinalities=category_cardinalities,
        n_samples=n_samples,
    )
    top_cfg = _build_minimal_config(result["top_recommendation"]["graphType"], column_types, **_bmc_kwargs)
    result["top_recommendation"]["minimal_config"] = top_cfg
    warnings.extend(_validate_minimal_config(result["top_recommendation"]["graphType"], top_cfg))

    for alt in result["alternatives"]:
        alt_cfg = _build_minimal_config(alt["graphType"], column_types, **_bmc_kwargs)
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
        f"See the full API: https://canvasxpress.org/parameters.html"
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
_GROUPING_GTS = {"Boxplot", "Violin", "Dotplot", "Treemap"}

# Array-valued axes that must always be present (as [] if unset) per graph type.
_GT_REQUIRED_AXES: dict[str, tuple[str, ...]] = {
    # ── Multi-dimensional: x, y, z ──────────────────────────────────────────
    **{gt: ("xAxis", "yAxis", "zAxis") for gt in _MULTI_DIM_GTS},
    # ── Dual-axis / combined: x + x2 ────────────────────────────────────────
    **{gt: ("xAxis", "xAxis2") for gt in _COMBINED_GTS},
    # ── Grouping charts: x + groupingFactors ────────────────────────────────
    **{gt: ("xAxis", "groupingFactors") for gt in _GROUPING_GTS},
    # ── colorBy-grouped single-axis ─────────────────────────────────────────
    "Ridgeline":            ("xAxis",),
    "Histogram":            ("xAxis",),
    # ── Hierarchical ────────────────────────────────────────────────────────
    "Sunburst":             ("hierarchy", "xAxis"),
    "Tree":                 ("hierarchy", "xAxis"),
    "TreeBracket":          ("hierarchy", "xAxis"),
    # ── Flow / alluvial ─────────────────────────────────────────────────────
    "Sankey":               ("sankeyAxes", "xAxis"),
    "Alluvial":             ("sankeyAxes", "xAxis"),
    # ── Two-axis charts (x + y, no z) ───────────────────────────────────────
    "Hexplot":              ("xAxis", "yAxis"),
    "Binplot":              ("xAxis", "yAxis"),
    "Bubble":               ("xAxis", "yAxis", "zAxis"),
    "Bump":                 ("xAxis", "yAxis"),
    "Dumbbell":             ("xAxis", "yAxis"),
    "QQ":                   ("xAxis", "yAxis"),
    "Ribbon":               ("xAxis", "yAxis"),
    "Tornado":              ("xAxis", "yAxis"),
    # ── x + groupingFactors ─────────────────────────────────────────────────
    "Gantt":                ("xAxis", "groupingFactors"),
    # ── Single-axis (xAxis only) ─────────────────────────────────────────────
    "Heatmap":              ("xAxis",),
    "Density":              ("xAxis",),
    "TagCloud":             ("xAxis",),
    "WordCloud":            ("xAxis",),
    "Bar":                  ("xAxis",),
    "Stacked":              ("xAxis",),
    "StackedPercent":       ("xAxis",),
    "Waterfall":            ("xAxis",),
    "Lollipop":             ("xAxis",),
    "Line":                 ("xAxis",),
    "Area":                 ("xAxis",),
    "Correlation":          ("xAxis",),
    "SPLOM":                ("xAxis", "yAxis"),
    "Radar":                ("xAxis",),
    "ParallelCoordinates":  ("xAxis",),
    "Network":              ("xAxis",),
    "Venn":                 ("xAxis",),
    "Chord":                ("xAxis",),
    "CDF":                  ("xAxis",),
    "Cleveland":            ("xAxis",),
    "Upset":                ("xAxis",),
    "Bullet":               ("xAxis",),
}


def _build_minimal_config(
    graph_type: str,
    column_types: dict[str, str],
    max_level_fractions: Optional[dict[str, float]] = None,
    category_cardinalities: Optional[dict[str, int]] = None,
    n_samples: Optional[int] = None,
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

    def _is_bad_grouping_col(col: str) -> bool:
        """True if col should never be used as a grouping/factor axis."""
        # Rule 1: one level covers > 90% of rows → near-constant, useless for grouping
        if max_level_fractions and max_level_fractions.get(col, 0.0) > 0.9:
            return True
        # Rule 2: cardinality > 50% of n_samples → effectively a row label
        if category_cardinalities and n_samples and n_samples > 0:
            if category_cardinalities.get(col, 0) > n_samples * 0.5:
                return True
        return False

    num_cols  = [c for c, t in column_types.items()
                 if t.lower() in _cxs._NUMERIC_ALIASES and not _is_id_col(c)]
    fac_cols  = [c for c, t in column_types.items()
                 if t.lower() in _cxs._FACTOR_ALIASES and not _is_id_col(c) and not _is_bad_grouping_col(c)]
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

    elif graph_type == "SPLOM":
        # All numeric columns go to both xAxis and yAxis (pairwise matrix).
        # Factor columns colour the points — never use groupingFactors.
        splom_nums = num_cols  # ID cols already excluded above
        cfg["xAxis"] = splom_nums[:]
        cfg["yAxis"] = splom_nums[:]
        if fac_cols:
            cfg["colorBy"] = fac_cols[0]

    elif graph_type == "Density":
        if num_cols:
            cfg["xAxis"] = [num_cols[0]]
        if fac_cols:
            cfg["colorBy"] = fac_cols[0]

    elif graph_type == "Histogram":
        if num_cols:
            cfg["xAxis"] = [num_cols[0]]
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

    # Ensure all array-valued axes expected for this graph type are present;
    # any that weren't assigned by the branch logic above get an empty array.
    _required = _GT_REQUIRED_AXES.get(graph_type, ("xAxis", "yAxis", "zAxis"))
    for _axis in _required:
        if _axis not in cfg:
            cfg[_axis] = []

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
# ZIP code / location → lat/lng resolvers (free APIs, no key required)
# ---------------------------------------------------------------------------

def _zip_to_latlon(zipcode: str) -> tuple[float, float] | None:
    """
    Resolve a US ZIP code to (lat, lng) using the free zippopotam.us API.
    Returns None on failure (bad ZIP, network error, etc.).
    """
    import urllib.request as _urlreq
    import json as _json
    zipcode = zipcode.strip().zfill(5)
    url = f"https://api.zippopotam.us/us/{zipcode}"
    try:
        with _urlreq.urlopen(url, timeout=5) as resp:
            data = _json.loads(resp.read())
        place = data["places"][0]
        return float(place["latitude"]), float(place["longitude"])
    except Exception:
        return None


def _geocode_location(place: str) -> tuple[float, float] | None:
    """
    Geocode any city, address, or landmark worldwide using the free
    Nominatim (OpenStreetMap) API.  Returns (lat, lng) or None on failure.
    """
    import urllib.request as _urlreq
    import urllib.parse as _urlparse
    import json as _json
    url = ("https://nominatim.openstreetmap.org/search?"
           + _urlparse.urlencode({"q": place.strip(), "format": "json", "limit": 1}))
    req = _urlreq.Request(url, headers={"User-Agent": "canvasxpress-mcp/1.0"})
    try:
        with _urlreq.urlopen(req, timeout=5) as resp:
            results = _json.loads(resp.read())
        if results:
            return float(results[0]["lat"]), float(results[0]["lon"])
        return None
    except Exception:
        return None


@mcp.tool(
    description=(
        "Generate a CanvasXpress map visualization config. "
        "CanvasXpress supports world maps, continent maps, country maps, U.S. state maps, "
        "and custom maps (via topoJSON). "
        "Pass a map identifier to mapId: 'World', 'WorldContinents', 'Africa', 'Asia', "
        "'Europe', 'NorthAmerica', 'SouthAmerica', 'Oceania', 'USAStates', 'USACounties', "
        "a three-letter country code (e.g. 'USA', 'CAN', 'MEX', 'ARG', 'GBR'), "
        "a two-letter U.S. state code (e.g. 'CA', 'NY', 'TX'), "
        "or the full name of a country or U.S. state. "
        "Optionally supply data to enrich the map — CanvasXpress automatically maps "
        "the provided IDs to the corresponding geographic elements. "
        "Optionally supply markers (lat/lng pins with labels, colors, and shapes) to annotate "
        "specific locations on the map. "
        "Supports pie chart overlays on map regions by passing decorations={'pie': {...}}, "
        "and proportional sizing of regions/symbols via 'size_by'. "
        "Returns a CanvasXpress JSON config ready to pass to new CanvasXpress()."
    )
)
def create_map_config(
    map_id: str,
    data: list[list] | None = None,
    title: str | None = None,
    color_scheme: str | None = None,
    markers: list[dict] | None = None,
    color_by: str | None = None,
    size_by: str | None = None,
    decorations: dict | None = None,
    topo_json: str | None = None,
    legend_order: dict | None = None,
) -> dict:
    """
    Args:
        map_id:       Map identifier string. Examples:
                      'World', 'WorldContinents', 'Africa', 'Asia', 'Europe',
                      'NorthAmerica', 'SouthAmerica', 'Oceania',
                      'USAStates', 'USACounties',
                      three-letter country code ('USA', 'CAN', 'MEX', 'ARG', 'GBR'),
                      two-letter U.S. state code ('CA', 'NY', 'TX', 'FL'),
                      or the full name of a country or U.S. state.
        data:         Optional CSV-style array of arrays (first row = column headers)
                      to enrich the map with additional data.
                      The FIRST data column must contain geographic ID codes that
                      CanvasXpress uses to match rows to map features:
                        map_id='World'        → ISO 3-letter country codes
                                                 e.g. ["ALB","ARG","ARM","AUS","AUT",
                                                        "BEL","BRA","CAN","CHN","DEU",
                                                        "ESP","FRA","GBR","IND","ITA",
                                                        "JPN","MEX","RUS","USA","ZAF"]
                        map_id='WorldContinents' → continent names
                                                 e.g. ["Africa","Asia","Europe",
                                                        "NorthAmerica","Oceania","SouthAmerica"]
                        map_id='USAStates'    → 2-letter US state codes
                                                 e.g. ["AL","AK","AZ","AR","CA","CO",
                                                        "CT","DE","FL","GA","HI","ID",
                                                        "IL","IN","IA","KS","KY","LA",
                                                        "ME","MD","MA","MI","MN","MO",
                                                        "MS","MT","NE","NV","NH","NJ",
                                                        "NM","NY","NC","ND","OH","OK",
                                                        "OR","PA","RI","SC","SD","TN",
                                                        "TX","UT","VT","VA","WA","WV",
                                                        "WI","WY","DC"]
                        map_id='USACounties'  → 5-digit FIPS county codes
                                                 e.g. ["01001","06037","48113"]
                        map_id=<country ISO3> → feature property values from the map file.
                                                Use mapPropertyId in the config to specify
                                                which feature property to match against.
                          (e.g. 'CAN')  mapPropertyId: "prov_name_en"
                                         e.g. ["Alberta","British Columbia","Manitoba",
                                               "New Brunswick","Newfoundland and Labrador",
                                               "Northwest Territories","Nova Scotia","Nunavut",
                                               "Ontario","Prince Edward Island","Quebec",
                                               "Saskatchewan","Yukon"]
                          (e.g. 'AUS')  mapPropertyId: "STATE_NAME"
                                         e.g. ["New South Wales","Victoria","Queensland",
                                               "South Australia","Western Australia",
                                               "Tasmania","Northern Territory",
                                               "Australian Capital Territory"]
                          (e.g. 'GBR')  mapPropertyId: "HASC_2" (192 districts)
                                         e.g. ["GB.BA","GB.BN","GB.BD","GB.LN","GB.KE",...]
                                         or mapPropertyId: "NAME_2" for district names.
                                         See https://canvasxpress.org/data/maps/GBR.json
                          Other ISO3    → check https://canvasxpress.org/data/maps/<ISO3>.json
                        map_id=<US state code> → County FIPS codes or county names
                      Remaining columns contain numeric values to visualize.
                      e.g. [["Country","GDP","Population"],
                             ["USA",21000,331],
                             ["CAN",1800,38]]
        title:        Optional chart title string.
        color_scheme: Optional color palette name.
                      e.g. 'Blues', 'RdBu', 'YlOrRd', 'Greens', 'Tableau'.
        color_by:     Optional column name whose values color the map regions/symbols.
                      e.g. 'Winner' to color states by election winner.
        size_by:      Optional column name whose numeric values scale the size of each
                      map symbol (relevant for pie or point maps).
                      e.g. 'Total' to scale pie size by total votes.
        decorations:  Optional dict of map decoration overlays, mirroring the CanvasXpress
                      'decorations' config key.  Currently supports:
                        'pie' (dict) — overlay pie charts on each map region.
                          Keys:
                            'smps'   (list[str], required) — columns that become pie slices.
                            'colors' (list[str], optional) — one color per slice.
                            'size'   (float, optional)     — size multiplier (default 2.5).
                          When using pie overlays, set map_id to 'albersStatesPie' or another
                          pie-capable map ID, and supply topo_json if needed.
                      e.g. {'pie': {'smps': ['Democrat','Republican','Libertarian','Other'],
                                    'colors': ['blue','red','yellow','green'],
                                    'size': 2.5}}
        topo_json:    Optional URL to a custom topoJSON file.
                      e.g. 'https://www.canvasxpress.org/data/json/usa-albers-states.json'
        legend_order: Optional dict mapping column names to an ordered list of values
                      controlling legend display order.
                      e.g. {'Winner': ['Republican', 'Democrat']}
        markers:      Optional list of marker dicts to pin locations on the map.
                      Each marker requires 'lat' and 'lng' (decimal degrees).
                      Optional fields: 'label' (str), 'color' (str, default 'red'),
                      'shape' ('teardrop'|'circle'|'star'|'square', default 'teardrop'),
                      'size' (int 1-10, default 4).
                      CanvasXpress uses coords: [lat, lng] format internally.
                      Instead of 'lat'/'lng', you may pass:
                        'zip'      — US ZIP code (resolved via zippopotam.us)
                        'location' — any city, address, or landmark worldwide
                                     (resolved via Nominatim/OpenStreetMap, free)
                      e.g. [{'lat': 40.23, 'lng': -74.93, 'label': 'Newtown PA'},
                             {'zip': '10001', 'label': 'New York, NY', 'color': 'blue'},
                             {'location': 'Paris, France', 'label': 'Paris', 'color': 'green'},
                             {'location': 'Tokyo', 'label': 'Tokyo', 'shape': 'circle'}]

    Returns:
        Dict with keys:
          config       (dict) - CanvasXpress JSON config with graphType='Map' and mapId
          valid        (bool) - True if the config is valid
          warnings     (list) - any validation warnings
          map_id       (str)  - the map identifier used
          headers_used (list) - column headers from data (if provided)
    """
    map_id = map_id.strip() if map_id else ""
    if not map_id:
        return {
            "config":       {},
            "valid":        False,
            "warnings":     ["'map_id' is required and cannot be empty."],
            "map_id":       "",
            "headers_used": [],
        }

    config: dict = {
        "graphType": "Map",
        "mapId":     map_id,
    }

    warnings: list[str] = []

    if title and title.strip():
        config["title"] = title.strip()

    VALID_COLOR_SCHEMES = {
        "YlGn", "YlGnBu", "GnBu", "BuGn", "PuBuGn", "PuBu", "BuPu", "RdPu", "PuRd",
        "OrRd", "YlOrRd", "YlOrBr", "Purples", "Blues", "Greens", "Oranges", "Reds",
        "Greys", "PuOr", "BrBG", "PRGn", "PiYG", "RdBu", "RdGy", "RdYlBu", "Spectral",
        "RdYlGn", "Bootstrap", "Economist", "Excel", "GGPlot", "Solarized", "PaulTol",
        "ColorBlind", "Tableau", "WallStreetJournal", "Stata", "BlackAndWhite", "CanvasXpress",
    }
    if color_scheme and color_scheme.strip():
        cs = color_scheme.strip()
        if cs not in VALID_COLOR_SCHEMES:
            warnings.append(
                f"Unknown colorScheme '{cs}'. "
                f"Valid options: {', '.join(sorted(VALID_COLOR_SCHEMES))}"
            )
        config["colorScheme"] = cs

    if color_by and color_by.strip():
        config["colorBy"] = color_by.strip()

    if size_by and size_by.strip():
        config["sizeBy"] = size_by.strip()

    if topo_json and topo_json.strip():
        config["topoJSON"] = topo_json.strip()

    if legend_order and isinstance(legend_order, dict):
        config["legendOrder"] = legend_order

    # Extract headers from data if provided
    headers_used: list[str] = []
    if data is not None:
        try:
            headers_used = extract_headers_from_data(data)
        except ValueError as e:
            return {
                "config":       config,
                "valid":        False,
                "warnings":     [str(e)],
                "map_id":       map_id,
                "headers_used": [],
            }

    # Build decorations.marker array if markers were provided
    if markers:
        VALID_SHAPES = {"teardrop", "circle", "star", "square", "triangle", "diamond"}
        marker_list = []
        for i, m in enumerate(markers):
            lat = m.get("lat")
            lng = m.get("lng")
            # Resolve US ZIP code if lat/lng not explicitly provided
            if (lat is None or lng is None) and m.get("zip"):
                resolved = _zip_to_latlon(str(m["zip"]))
                if resolved is None:
                    warnings.append(
                        f"Marker {i}: could not resolve ZIP '{m['zip']}' to coordinates — skipped."
                    )
                    continue
                lat, lng = resolved
            # Resolve any city/place name worldwide via Nominatim
            if (lat is None or lng is None) and m.get("location"):
                resolved = _geocode_location(str(m["location"]))
                if resolved is None:
                    warnings.append(
                        f"Marker {i}: could not geocode location '{m['location']}' — skipped."
                    )
                    continue
                lat, lng = resolved
            if lat is None or lng is None:
                warnings.append(
                    f"Marker {i}: missing 'lat'/'lng', 'zip', or 'location' — skipped."
                )
                continue
            try:
                lat = float(lat)
                lng = float(lng)
            except (TypeError, ValueError):
                warnings.append(
                    f"Marker {i}: 'lat' and 'lng' must be numeric — skipped."
                )
                continue
            shape = m.get("shape", "teardrop")
            if shape not in VALID_SHAPES:
                warnings.append(
                    f"Marker {i}: unknown shape '{shape}'. "
                    f"Valid shapes: {', '.join(sorted(VALID_SHAPES))}. Defaulting to 'teardrop'."
                )
                shape = "teardrop"
            entry: dict = {
                "coords": [lat, lng],
                "color":  m.get("color", "red"),
                "shape":  shape,
                "size":   int(m.get("size", 4)),
            }
            if m.get("label"):
                entry["label"] = str(m["label"])
            marker_list.append(entry)
        if marker_list:
            config["decorations"] = {"marker": marker_list}

    # Build decorations.pie if supplied via the decorations parameter
    if decorations and isinstance(decorations, dict):
        pie = decorations.get("pie")
        if pie and isinstance(pie, dict):
            smps = pie.get("smps")
            if not smps or not isinstance(smps, list):
                warnings.append(
                    "decorations.pie: 'smps' (list of column names) is required — skipped."
                )
            else:
                pie_entry: dict = {"smps": smps}
                if pie.get("colors") and isinstance(pie["colors"], list):
                    pie_entry["colors"] = pie["colors"]
                pie_entry["size"] = float(pie.get("size", 2.5))
                # Merge into existing decorations dict (may already have 'marker')
                if "decorations" not in config:
                    config["decorations"] = {}
                config["decorations"]["pie"] = [pie_entry]

    log.info("Map config created: mapId=%s headers=%s markers=%d",
             map_id, headers_used, len(markers) if markers else 0)
    return {
        "config":       config,
        "valid":        len(warnings) == 0,
        "warnings":     warnings,
        "map_id":       map_id,
        "headers_used": headers_used,
    }


# ---------------------------------------------------------------------------
# HTML renderer — converts tool results to a display-ready HTML string
# ---------------------------------------------------------------------------

# Maps each graphType to its canvasxpress.org/examples/<slug>-1.html slug.
# Virtual types that have no dedicated example page fall back to the underlying type.
_GT_EXAMPLE_SLUG: dict[str, str] = {
    # native types — slug == gt.lower()
    "Area": "area", "AreaLine": "arealine", "Bar": "bar", "BarLine": "barline",
    "Boxplot": "boxplot", "Circular": "circular", "Correlation": "correlation",
    "DotLine": "dotline", "Dotplot": "dotplot", "Gantt": "gantt",
    "Heatmap": "heatmap", "Histogram": "histogram", "Line": "line",
    "Network": "network", "ParallelCoordinates": "parallelcoordinates",
    "Sankey": "sankey", "Scatter2D": "scatter2d", "Scatter3D": "scatter3d",
    "ScatterBubble2D": "scatterbubble2d", "SPLOM": "splom", "Stacked": "stacked",
    "StackedLine": "stackedline", "StackedPercent": "stackedpercent",
    "StackedPercentLine": "stackedpercentline", "Streamgraph": "streamgraph",
    "TagCloud": "tagcloud", "Tree": "tree", "Upset": "upset",
    # virtual types with dedicated example pages
    "Bubble": "bubble", "Bullet": "bullet", "Chord": "chord",
    "Contour": "contour", "Density": "density", "Dumbbell": "dumbbell",
    "Lollipop": "lollipop", "Radar": "radar", "Sunburst": "sunburst",
    "Violin": "violin", "Waterfall": "waterfall",
    # virtual types with hyphenated slug
    "KaplanMeier": "kaplan-meier",
    "Map": "map",
    # virtual types with no dedicated page — fall back to underlying native type
    "Alluvial": "sankey", "Bin": "scatter2d", "Binplot": "scatter2d",
    "Bump": "scatter2d", "CDF": "scatter2d", "Cleveland": "dotplot",
    "Distribution": "histogram", "Donut": "circular",
    "Hex": "scatter2d", "Hexplot": "scatter2d", "Pareto": "barline",
    "QQ": "scatter2d", "Quantile": "scatter2d", "Ribbon": "sankey",
    "Ridgeline": "scatter2d", "Spaghetti": "scatter2d",
    "TimeSeries": "scatter2d", "Time-Series": "scatter2d", "Time Series": "scatter2d",
    "Tornado": "stacked", "TreeBracket": "tree", "Volcano": "scatter2d",
    "WordCloud": "tagcloud",
}


def _result_to_html(result: dict, tool: str) -> str:
    """
    Convert a tool result dict into an HTML fragment suitable for chat display.

    Uses <span> for prose, <pre><code> for code/config blocks, <ul>/<li> for lists.
    Returns a <div class="cX-Chat-LLM-Response"> string.
    """
    import html as _html_mod
    import json as _json

    def esc(s) -> str:
        return _html_mod.escape(str(s))

    def _is_code(text: str) -> bool:
        """Heuristic: ≥40 % of lines look like code → treat as a code block."""
        lines = [l for l in text.splitlines() if l.strip()]
        if not lines:
            return False
        hits = sum(
            1 for l in lines
            if re.match(r"^\s{2,}", l)
            or l.strip().startswith(("#", "library(", "install.", "devtools::",
                                     "canvasXpress(", "do.call(", "shiny", "output$",
                                     "renderCanvas", "canvasXpressOutput", "```"))
            or l.strip().startswith(("{", "[", "}", "]"))
        )
        return hits / len(lines) >= 0.4

    def render_content(text: str) -> str:
        """Split on fenced code blocks first, then blank lines; prose → <p>, code → <pre><code>."""
        parts_out = []
        # Split on fenced code blocks (``` ... ```) preserving the delimiter
        segments = re.split(r"(```[^\n]*\n.*?```)", text.strip(), flags=re.DOTALL)
        for seg in segments:
            if seg.startswith("```"):
                # Strip the opening fence line (```{r}, ```python, ``` etc.) and closing ```
                inner = re.sub(r"^```[^\n]*\n", "", seg)
                inner = re.sub(r"\n?```$", "", inner)
                parts_out.append(f"<pre><code>{esc(inner)}</code></pre>")
            else:
                # Further split prose segment on blank lines
                for para in re.split(r"\n{2,}", seg.strip()):
                    if not para.strip():
                        continue
                    if _is_code(para):
                        parts_out.append(f"<pre><code>{esc(para.rstrip())}</code></pre>")
                    else:
                        lines_html = "<br>".join(f"<span>{esc(l)}</span>" for l in para.splitlines() if l.strip())
                        if lines_html:
                            parts_out.append(f"<p>{lines_html}</p>")
        return "\n".join(parts_out)

    parts = ['<div class="cX-Chat-LLM-Response">']

    def _link(url: str, label: str) -> str:
        return f'<a href="{url}" target="_blank" rel="noopener">{esc(label)}</a>'

    def _footer(*links) -> str:
        return '<p class="cX-Chat-LLM-Links">' + " &nbsp;|&nbsp; ".join(links) + "</p>"

    def _example_link(gt: str) -> str:
        slug = _GT_EXAMPLE_SLUG.get(gt, gt.lower())
        return _link(f"https://canvasxpress.org/examples/{slug}-1.html", f"{gt} examples")

    # ── params ──────────────────────────────────────────────────────────────
    if tool == "params":
        if "param" in result and "description" in result:
            # Single param lookup
            p = result.get("param", "")
            parts.append(f"<h3><code>{esc(p)}</code></h3>")
            if result.get("description"):
                parts.append(f"<p>{esc(result['description'])}</p>")
            if result.get("type"):
                parts.append(f"<p><strong>Type:</strong> <span>{esc(result['type'])}</span></p>")
            vv = result.get("valid_values")
            if vv:
                if isinstance(vv, list):
                    items = "".join(f"<li><code>{esc(v)}</code></li>" for v in vv)
                    parts.append(f"<p><strong>Valid values:</strong></p><ul>{items}</ul>")
                else:
                    parts.append(f"<p><strong>Valid values:</strong> <span>{esc(vv)}</span></p>")
            gts = result.get("graph_types")
            if gts and isinstance(gts, list):
                parts.append(f"<p><strong>Applies to:</strong> <span>{esc(', '.join(gts))}</span></p>")
            parts.append(_footer(
                _link(f"https://canvasxpress.org/assets/api/{p}.html", f"{p} API docs"),
                _link("https://canvasxpress.org/parameters.html", "Full API reference"),
            ))
        elif "graph_type" in result and "params" in result:
            # Params for a specific graph type
            gt = result.get("graph_type", "")
            cnt = result.get("param_count", "")
            parts.append(f"<h3>Parameters for <code>{esc(gt)}</code> <em>({esc(cnt)})</em></h3>")
            params_dict = result.get("params", {})
            if isinstance(params_dict, dict):
                for name, info in params_dict.items():
                    desc = info.get("description", "") if isinstance(info, dict) else str(info)
                    parts.append(f'<div><code>{esc(name)}</code> — <span>{esc(desc)}</span></div>')
            parts.append(_footer(
                _example_link(gt),
                _link("https://canvasxpress.org/parameters.html", "Full API reference"),
            ))
        else:
            # All params listing
            cnt = result.get("param_count", "")
            parts.append(f"<h3>All Parameters <em>({esc(cnt)})</em></h3>")
            params_dict = result.get("params", {})
            if isinstance(params_dict, dict):
                for name, info in params_dict.items():
                    desc = info.get("description", "") if isinstance(info, dict) else str(info)
                    parts.append(f'<div><code>{esc(name)}</code> — <span>{esc(desc)}</span></div>')
            if result.get("tip"):
                parts.append(f"<p><em>{esc(result['tip'])}</em></p>")
            parts.append(_footer(
                _link("https://canvasxpress.org/parameters.html", "Full API reference"),
            ))

    # ── axes ─────────────────────────────────────────────────────────────────
    elif tool == "axes":
        gt = result.get("graph_type", "")
        cat = result.get("category", "")
        parts.append(f"<h3>Axes: <code>{esc(gt)}</code> <em>({esc(cat)})</em></h3>")
        valid = result.get("valid_axes", [])
        if valid:
            items = "".join(f"<li><code>{esc(a)}</code></li>" for a in valid)
            parts.append(f"<p><strong>Valid axes:</strong></p><ul>{items}</ul>")
        invalid = result.get("invalid_axes", [])
        if invalid:
            items = "".join(f"<li><code>{esc(a)}</code></li>" for a in invalid)
            parts.append(f"<p><strong>Not allowed:</strong></p><ul>{items}</ul>")
        atp = result.get("axis_title_param")
        if atp:
            parts.append(f"<p><strong>Axis title param:</strong> <code>{esc(atp)}</code></p>")
        parts.append(_footer(
            _example_link(gt),
            _link("https://canvasxpress.org/parameters.html", "API reference"),
        ))

    # ── explain ───────────────────────────────────────────────────────────────
    elif tool == "explain":
        prop = result.get("property", "")
        explanation = result.get("explanation", "")
        # Strip markdown bold wrapper if present: **`prop`** —
        explanation = re.sub(r"\*\*`[^`]+`\*\*\s*[—-]\s*", "", explanation)
        parts.append(f"<h3><code>{esc(prop)}</code></h3>")
        parts.append(f"<p>{esc(explanation)}</p>")
        parts.append(_footer(
            _link(f"https://canvasxpress.org/assets/api/{prop}.html", f"{prop} API docs"),
            _link("https://canvasxpress.org/parameters.html", "Full API reference"),
        ))

    # ── explain-r / explain-ggplot ────────────────────────────────────────────
    elif tool in ("explain-r", "explain-ggplot"):
        if "error" in result:
            parts.append(f'<p class="cX-Chat-LLM-Error">{esc(result["error"])}</p>')
            avail = result.get("available_topics", [])
            if avail:
                items = "".join(f"<li><code>{esc(t)}</code></li>" for t in avail)
                parts.append(f"<p><strong>Available topics:</strong></p><ul>{items}</ul>")
        elif "section" in result:
            # Single-topic response
            sec = result["section"]
            parts.append(f"<h3>{esc(sec.get('title', result.get('topic', '')))}</h3>")
            parts.append(render_content(sec.get("content", "")))
        else:
            # All-topics response
            overview = result.get("overview")
            if overview:
                parts.append(f"<p>{esc(overview)}</p>")
            sections = result.get("sections", {})
            for _key, sec in sections.items():
                if not isinstance(sec, dict):
                    continue
                parts.append(f"<h3>{esc(sec.get('title', _key))}</h3>")
                parts.append(render_content(sec.get("content", "")))
        if tool == "explain-r":
            parts.append(_footer(
                _link("https://cran.r-project.org/package=canvasXpress", "CRAN package"),
                _link("https://github.com/neuhausi/canvasXpress", "GitHub"),
                _link("https://canvasxpress.org/r-interface.html", "R vignette"),
            ))
        else:
            parts.append(_footer(
                _link("https://canvasxpress.org/ggplot-interface.html", "ggplot vignette"),
            ))

    # ── minimal-params ────────────────────────────────────────────────────────
    elif tool == "minimal-params":
        if "error" in result:
            parts.append(f'<p class="cX-Chat-LLM-Error">{esc(result["error"])}</p>')
            if result.get("tip"):
                parts.append(f"<p>{esc(result['tip'])}</p>")
        else:
            gt = result.get("graphType", "")
            params_list = result.get("required_parameters", [])
            parts.append(f"<h3>Minimal parameters for <code>{esc(gt)}</code></h3>")
            items = "".join(f"<li><code>{esc(p)}</code></li>" for p in params_list)
            parts.append(f"<ul>{items}</ul>")
            # Build a minimal JSON config example
            config_ex: dict = {"graphType": gt}
            for p in params_list:
                if p == "graphType":
                    continue
                if p in ("xAxis", "yAxis", "zAxis", "xAxis2", "groupingFactors",
                         "sankeyAxes", "hierarchy"):
                    config_ex[p] = ["<column>"]
                else:
                    config_ex[p] = "<value>"
            parts.append(f"<pre><code>{esc(_json.dumps(config_ex, indent=2))}</code></pre>")
            parts.append(_footer(
                _example_link(gt),
                _link("https://canvasxpress.org/parameters.html", "API reference"),
            ))

    # ── select_canvasxpress_chart ─────────────────────────────────────────────
    elif tool == "select_canvasxpress_chart":
        target = result.get("target", "")

        def _apply_btn(cfg: dict, label: str) -> str:
            cfg_json = esc(_json.dumps(cfg))  # &quot; escapes " for HTML attr
            return (
                f'<button class="cX-Chat-LLM-Apply" '
                f'onclick="CanvasXpress.applySelectConfig(\'{esc(target)}\', JSON.parse(this.dataset.cfg))" '
                f'data-cfg="{cfg_json}">{esc(label)}</button>'
            )

        def _rec_card(rec: dict, is_top: bool) -> list:
            gt       = rec.get("graphType", "")
            score    = rec.get("score", "")
            desc     = rec.get("description", "")
            clinical = rec.get("clinical_use", "")
            next_s   = rec.get("next_step", "")
            factors  = rec.get("scoring_factors", [])
            min_cfg  = rec.get("minimal_config", {})
            cls = "cX-Chat-Select-Top" if is_top else "cX-Chat-Select-Alt"
            h = [f'<div class="{cls}">',
                 f'<div class="cX-Chat-Select-Header">'
                 f'<span class="cX-Chat-Select-GT">{esc(gt)}</span>'
                 f'<span class="cX-Chat-Select-Score">score: {esc(str(score))}</span>'
                 f'</div>']
            if desc:
                h.append(f'<p class="cX-Chat-Select-Desc">{esc(desc)}</p>')
            if clinical:
                h.append(f'<p class="cX-Chat-Select-Clinical"><strong>Clinical use:</strong> {esc(clinical)}</p>')
            if factors:
                items = "".join(f"<li>{esc(f)}</li>" for f in factors)
                h.append(f'<ul class="cX-Chat-Select-Factors">{items}</ul>')
            if next_s:
                h.append(f'<p class="cX-Chat-Select-Next"><em>{esc(next_s)}</em></p>')
            if min_cfg:
                h.append(_apply_btn(min_cfg, f"Apply {gt}"))
            h.append('</div>')
            return h

        top = result.get("top_recommendation", {})
        if top:
            parts.append('<h3>Recommended</h3>')
            parts.extend(_rec_card(top, True))

        tiebreak = result.get("tiebreak", {})
        if tiebreak and tiebreak.get("used"):
            parts.append(f'<p class="cX-Chat-Select-Next"><em>{esc(tiebreak.get("reason", ""))}</em></p>')

        alts = result.get("alternatives", [])
        if alts:
            parts.append('<h3>Alternatives</h3>')
            parts.append('<div class="cX-Chat-Select-Grid">')
            for alt in alts:
                parts.extend(_rec_card(alt, False))
            parts.append('</div>')

        for w in result.get("warnings", []):
            parts.append(f'<p class="cX-Chat-LLM-Error">{esc(w)}</p>')

        parts.append(_footer(
            _link("https://canvasxpress.org/examples.html", "Examples"),
            _link("https://canvasxpress.org/parameters.html", "API reference"),
        ))

    # ── map ──────────────────────────────────────────────────────────────────
    elif tool == "map":
        map_id = result.get("map_id", "")
        config = result.get("config", {})
        parts.append(f"<h3>Map Config: <code>{esc(map_id)}</code></h3>")
        for w in result.get("warnings", []):
            parts.append(f'<p class="cX-Chat-LLM-Error">{esc(w)}</p>')
        parts.append(f"<pre><code>{esc(_json.dumps(config, indent=2))}</code></pre>")
        headers_used = result.get("headers_used", [])
        if headers_used:
            parts.append(
                f"<p><strong>Data columns:</strong> "
                f"<span>{esc(', '.join(headers_used))}</span></p>"
            )
        parts.append(_footer(
            _link("https://canvasxpress.org/examples/map-1.html", "Map examples"),
            _link("https://canvasxpress.org/parameters.html", "API reference"),
        ))

    parts.append("</div>")
    return {"content": "\n".join(parts)}


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
        r"|^\d{4}[-/]\d{1,2}$"               # YYYY-MM bare year-month
        r"|^\d{4}$"                            # bare year
    )
    _YEAR_COL_RE = re.compile(r"year|yr|date|time|month|week|day|visit", re.IGNORECASE)
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
                continue
            # Integer values in [1900-2099] in a time-named column → treat as date
            # (handles Year=2019, Year=2020 etc. which parse as numeric but are dates)
            try:
                int_vals = [int(float(str(v))) for v in values]
                if (
                    all(float(str(v)) == int(float(str(v))) for v in values)
                    and all(1900 <= iv <= 2099 for iv in int_vals)
                    and _YEAR_COL_RE.search(col_name)
                ):
                    column_types[col_name] = "date"
                    continue
            except (ValueError, TypeError):
                pass
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
<link rel="stylesheet" href="https://canvasxpress.org/dist/canvasXpress.css" type="text/css"/>
<script type="text/javascript" src="https://canvasxpress.org/dist/canvasXpress.min.js"></script>
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
  #preview{margin-top:24px;margin-bottom:20px;height:auto;background:#fff;border:1px solid #ddd;border-radius:6px;padding:16px;display:none}
  #preview h3{margin:0 0 10px;font-size:.95rem;display:flex;align-items:center;gap:10px}
  #preview h3 span.preview-hint{font-size:.78rem;font-weight:400;color:#888}
  #cx-preview-wrap{width:100%;position:relative;min-height:60px}
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
  <div class="tab" data-tab="map">Map</div>
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

<!-- ── MAP ─────────────────────────────────────────────────────────── -->
<div class="panel card" id="panel-map">
  <p class="hint">Create a CanvasXpress map visualization config. Supply a map identifier — world region, country code, or U.S. state code/name.</p>
  <div class="section-label">Required</div>
  <label>Map ID <span>world region, country code, or state code/name</span>
    <input type="text" id="map-id" placeholder="e.g. World, USAStates, USA, CAN, CA, Texas, Europe">
  </label>
  <div class="section-label">Optional</div>
  <label>Title<input type="text" id="map-title" placeholder="e.g. COVID-19 Cases by State"></label>
  <label>Color scheme<input type="text" id="map-scheme" placeholder="e.g. Blues, YlOrRd, RdBu"></label>
  <label>Data <span>JSON array of arrays — first row is headers</span>
    <textarea id="map-data" placeholder='[["State","Value"],["CA",42],["NY",38]]'></textarea>
  </label>
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
<div id="preview">
  <h3>Live Preview <span class="preview-hint">powered by CanvasXpress</span></h3>
  <div id="cx-preview-wrap"><canvas id="cx-preview" width="1030" height="440"></canvas></div>
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
  } else if (activeTab==='map') {
    url+='map';
    if(v('map-id'))     p.set('map_id',      v('map-id'));
    if(v('map-title'))  p.set('title',        v('map-title'));
    if(v('map-scheme')) p.set('color_scheme', v('map-scheme'));
    if(v('map-data'))   p.set('data',         v('map-data'));
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
 'ex-prop','mp-graph',
 'map-id','map-title','map-scheme','map-data'].forEach(function(id){
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
  'explain-ggplot':'ggplot2 Guide','minimal-params':'Minimal Parameters',
  'map':'Map Config'
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
    if(['generate','modify','km','map'].indexOf(activeTab)!==-1){
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
      var top=data.top_recommendation;
      if(top) metaEl.innerHTML='Top recommendation: <b>'+top.graphType+'</b>';
      preEl.textContent=JSON.stringify(data,null,2);
      // Only render preview when the user supplied data
      var selRaw = v('sel-data');
      if(selRaw && top && top.minimal_config) {
        var selParsed = null;
        try { selParsed = JSON.parse(selRaw); } catch(e){}
        renderPreview(top.minimal_config, selParsed);
      } else {
        document.getElementById('preview').style.display = 'none';
      }
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
    'mp-graph':'graph_type',
    'map-id':'map_id','map-title':'title','map-scheme':'color_scheme','map-data':'data'};
  Object.keys(map).forEach(function(id){
    var val=p.get(map[id]), el=document.getElementById(id);
    if(val&&el) el.value=val;
  });
  if(p.get('refresh')==='true'){var el=document.getElementById('p-refresh');if(el)el.checked=true;}
  if(p.get('topic')){['exr-topic','exg-topic'].forEach(function(id){var el=document.getElementById(id);if(el)el.value=p.get('topic');});}
  buildUrl();
})();

// ── CanvasXpress live preview (Select Chart tab only) ────────────────────────
var _cxPreviewInstance = null;

/**
 * Convert [[headers],[row,...], ...] into CanvasXpress raw data format.
 * CanvasXpress accepts data as a flat array-of-arrays (first row = headers)
 * when passed as the `data` key — identical to the km.json example format.
 * We pass it straight through so CX handles type detection itself, mirroring
 * how callbackLLM in canvasXpress.init.js works.
 */
function buildCxData(rawData) {
  if (!rawData || !Array.isArray(rawData) || rawData.length < 2) return null;
  return rawData;  // CanvasXpress accepts [[headers],[row...], ...] directly
}

/**
 * Render (or re-render) a CanvasXpress chart in the preview area.
 * Follows the same pattern as CanvasXpress.callbackLLM in canvasXpress.init.js:
 *   1. Destroy any existing instance.
 *   2. Recreate the canvas element.
 *   3. Instantiate new CanvasXpress({renderTo, data, config}).
 */
function renderPreview(config, rawData) {
  var previewEl = document.getElementById('preview');
  var wrapEl    = document.getElementById('cx-preview-wrap');
  if (!config || !config.graphType) { previewEl.style.display = 'none'; return; }
  previewEl.style.display = 'block';
  // Destroy previous instance and reset canvas
  if (_cxPreviewInstance) {
    try { _cxPreviewInstance.destroy('cx-preview'); } catch(e){}
    _cxPreviewInstance = null;
  }
  wrapEl.innerHTML = '<canvas id="cx-preview" width="1030" height="440"></canvas>';
  var cxData = buildCxData(rawData);
  var cfg    = Object.assign({}, config);
  try {
    var initObj = { renderTo: 'cx-preview', config: cfg };
    if (cxData) { initObj.data = cxData; }
    _cxPreviewInstance = new CanvasXpress(initObj);
  } catch(e) {
    wrapEl.innerHTML = '<p style="color:#c0392b;font-size:.85rem;padding:8px">Preview error: ' + e.message + '</p>';
  }
}
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
    cx: dict = {}
    if p.get("target",    "").strip(): cx["target"]   = p["target"].strip()
    if p.get("client_id", "").strip(): cx["client"]   = p["client_id"].strip()
    if p.get("callback",  "").strip(): cx["callback"] = p["callback"].strip()
    try:
        result = query_canvasxpress_params(
            graph_type=p.get("graph_type") or None,
            param_name=p.get("param_name") or None,
            refresh=p.get("refresh", "").lower() == "true",
        )
    except Exception as exc:
        log.exception("REST /params error")
        return _cx_response({"error": str(exc), "success": False}, cx, 500)
    result["html"] = _result_to_html(result, "params")
    return _cx_response(result, cx)


@mcp.custom_route("/axes", methods=["GET", "POST"])
async def rest_axes(request: Request) -> Response:
    """REST endpoint for get_axes_info."""
    if request.method == "GET":
        p = dict(request.query_params)
    else:
        ct = request.headers.get("content-type", "")
        p = await request.json() if "application/json" in ct else dict(await request.form())
    cx: dict = {}
    if p.get("target",    "").strip(): cx["target"]   = p["target"].strip()
    if p.get("client_id", "").strip(): cx["client"]   = p["client_id"].strip()
    if p.get("callback",  "").strip(): cx["callback"] = p["callback"].strip()
    gt = (p.get("graph_type") or "").strip()
    if not gt:
        return _cx_response({"error": "'graph_type' is required", "success": False}, cx, 400)
    try:
        result = get_axes_info(gt)
    except Exception as exc:
        log.exception("REST /axes error")
        return _cx_response({"error": str(exc), "success": False}, cx, 500)
    result["html"] = _result_to_html(result, "axes")
    return _cx_response(result, cx)


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

    cx: dict = {}
    if p.get("target",    "").strip(): cx["target"]   = p["target"].strip()
    if p.get("client_id", "").strip(): cx["client"]   = p["client_id"].strip()
    if p.get("callback",  "").strip(): cx["callback"] = p["callback"].strip()

    # --- resolve column_types and n_samples ---
    column_types: dict[str, str] = {}
    n_samples: int | None = None
    inferred_headers: list[str] = []
    type_source = "explicit"

    # 1. Parse data array if provided → infer types
    raw_data = p.get("data")
    category_cardinalities: dict[str, int] | None = None
    max_level_fractions: dict[str, float] | None = None
    if raw_data:
        try:
            data_arr = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
            inferred_types, inferred_headers, n_rows = _infer_column_types(data_arr)
            column_types = inferred_types
            n_samples = n_rows
            type_source = "inferred"
            # Compute cardinalities and dominant-level fractions for factor columns
            if data_arr and len(data_arr) > 1:
                headers_row = [str(h) for h in data_arr[0]]
                import cx_selector as _cxs_sel
                category_cardinalities = {}
                max_level_fractions: dict[str, float] = {}
                for col_idx, col_name in enumerate(headers_row):
                    if inferred_types.get(col_name, "") in _cxs_sel._FACTOR_ALIASES:
                        val_counts: dict[str, int] = {}
                        for row in data_arr[1:]:
                            if col_idx < len(row) and row[col_idx] is not None:
                                v = str(row[col_idx]).strip()
                                if v not in ("", "NA", "na", "nan", "NaN", "NULL", "null"):
                                    val_counts[v] = val_counts.get(v, 0) + 1
                        category_cardinalities[col_name] = len(val_counts)
                        if val_counts and n_rows > 0:
                            max_level_fractions[col_name] = max(val_counts.values()) / n_rows
        except Exception as exc:
            return _cx_response({"error": f"Could not parse 'data': {exc}", "success": False}, cx, 400)

    # 2. Explicit column_types override / supplement inferred types
    for key in ("column_types", "types"):
        raw_types = (p.get(key) or "").strip()
        if raw_types:
            explicit = _parse_col_types(raw_types)
            column_types.update(explicit)  # explicit wins
            type_source = "explicit" if not inferred_headers else "merged"
            break

    if not column_types:
        return _cx_response(
            {"error": "Provide 'column_types' (e.g. Gene=string,Expr=numeric) or 'data' (JSON array of arrays).", "success": False},
            cx, 400,
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
            category_cardinalities=category_cardinalities,
            max_level_fractions=max_level_fractions,
        )
        # Annotate with provenance for transparency
        result["type_source"] = type_source
        if inferred_headers:
            result["headers_detected"] = inferred_headers
        result["target"] = cx.get("target", "")
        result["html"] = _result_to_html(result, "select_canvasxpress_chart")
    except Exception as exc:
        log.exception("REST /select error")
        return _cx_response({"error": str(exc), "success": False}, cx, 500)
    return _cx_response(result, cx)


@mcp.custom_route("/explain", methods=["GET", "POST"])
async def rest_explain(request: Request) -> Response:
    """REST endpoint for explain_config_property."""
    if request.method == "GET":
        p = dict(request.query_params)
    else:
        ct = request.headers.get("content-type", "")
        p = await request.json() if "application/json" in ct else dict(await request.form())
    cx: dict = {}
    if p.get("target",    "").strip(): cx["target"]   = p["target"].strip()
    if p.get("client_id", "").strip(): cx["client"]   = p["client_id"].strip()
    if p.get("callback",  "").strip(): cx["callback"] = p["callback"].strip()
    prop = (p.get("property") or "").strip()
    if not prop:
        return _cx_response({"error": "'property' is required", "success": False}, cx, 400)
    try:
        result = explain_config_property(prop)
    except Exception as exc:
        log.exception("REST /explain error")
        return _cx_response({"error": str(exc), "success": False}, cx, 500)
    resp = {"property": prop, "explanation": result}
    resp["html"] = _result_to_html(resp, "explain")
    return _cx_response(resp, cx)


@mcp.custom_route("/explain-r", methods=["GET", "POST"])
async def rest_explain_r(request: Request) -> Response:
    """REST endpoint for explain_canvasxpress_r."""
    if request.method == "GET":
        p = dict(request.query_params)
    else:
        ct = request.headers.get("content-type", "")
        p = await request.json() if "application/json" in ct else dict(await request.form())
    cx: dict = {}
    if p.get("target",    "").strip(): cx["target"]   = p["target"].strip()
    if p.get("client_id", "").strip(): cx["client"]   = p["client_id"].strip()
    if p.get("callback",  "").strip(): cx["callback"] = p["callback"].strip()
    try:
        result = explain_canvasxpress_r(topic=p.get("topic") or None)
    except Exception as exc:
        log.exception("REST /explain-r error")
        return _cx_response({"error": str(exc), "success": False}, cx, 500)
    result["html"] = _result_to_html(result, "explain-r")
    return _cx_response(result, cx)


@mcp.custom_route("/explain-ggplot", methods=["GET", "POST"])
async def rest_explain_ggplot(request: Request) -> Response:
    """REST endpoint for explain_canvasxpress_ggplot."""
    if request.method == "GET":
        p = dict(request.query_params)
    else:
        ct = request.headers.get("content-type", "")
        p = await request.json() if "application/json" in ct else dict(await request.form())
    cx: dict = {}
    if p.get("target",    "").strip(): cx["target"]   = p["target"].strip()
    if p.get("client_id", "").strip(): cx["client"]   = p["client_id"].strip()
    if p.get("callback",  "").strip(): cx["callback"] = p["callback"].strip()
    try:
        result = explain_canvasxpress_ggplot(topic=p.get("topic") or None)
    except Exception as exc:
        log.exception("REST /explain-ggplot error")
        return _cx_response({"error": str(exc), "success": False}, cx, 500)
    result["html"] = _result_to_html(result, "explain-ggplot")
    return _cx_response(result, cx)


@mcp.custom_route("/minimal-params", methods=["GET", "POST"])
async def rest_minimal_params(request: Request) -> Response:
    """REST endpoint for get_minimal_parameters."""
    if request.method == "GET":
        p = dict(request.query_params)
    else:
        ct = request.headers.get("content-type", "")
        p = await request.json() if "application/json" in ct else dict(await request.form())
    cx: dict = {}
    if p.get("target",    "").strip(): cx["target"]   = p["target"].strip()
    if p.get("client_id", "").strip(): cx["client"]   = p["client_id"].strip()
    if p.get("callback",  "").strip(): cx["callback"] = p["callback"].strip()
    gt = (p.get("graph_type") or "").strip()
    if not gt:
        return _cx_response({"error": "'graph_type' is required", "success": False}, cx, 400)
    try:
        result = get_minimal_parameters(gt)
    except Exception as exc:
        log.exception("REST /minimal-params error")
        return _cx_response({"error": str(exc), "success": False}, cx, 500)
    result["html"] = _result_to_html(result, "minimal-params")
    return _cx_response(result, cx)

@mcp.custom_route("/map", methods=["GET", "POST"])
async def rest_map(request: Request) -> Response:
    """
    REST / JSONP endpoint for create_map_config.

    GET  /map?map_id=World&title=World+Map&color_scheme=Blues
    POST /map   (JSON body with same keys)

    Query / body parameters:
      map_id        (str, required) — map identifier e.g. 'World', 'USAStates', 'USA', 'CA'.
      data          (str)           — optional JSON array of arrays (first row = headers).
      title         (str)           — optional chart title.
      color_scheme  (str)           — optional color palette name.
      callback      (str)           — JSONP callback name.
      target        (str)           — CanvasXpress chart target ID (passed through).
      client_id     (str)           — CanvasXpress client ID (passed through).
    """
    if request.method == "GET":
        p = dict(request.query_params)
    else:
        ct = request.headers.get("content-type", "")
        p = await request.json() if "application/json" in ct else dict(await request.form())

    cx: dict = {}
    if p.get("target",    "").strip(): cx["target"]   = p["target"].strip()
    if p.get("client_id", "").strip(): cx["client"]   = p["client_id"].strip()
    if p.get("callback",  "").strip(): cx["callback"] = p["callback"].strip()

    map_id = (p.get("map_id") or "").strip()
    if not map_id:
        return _cx_response({"error": "'map_id' is required", "success": False}, cx, 400)

    kwargs: dict = {"map_id": map_id}
    if p.get("title", "").strip():
        kwargs["title"] = p["title"].strip()
    if p.get("color_scheme", "").strip():
        kwargs["color_scheme"] = p["color_scheme"].strip()

    raw_data = p.get("data")
    if raw_data:
        try:
            kwargs["data"] = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
        except Exception as exc:
            return _cx_response(
                {"error": f"Could not parse 'data': {exc}", "success": False}, cx, 400
            )

    raw_markers = p.get("markers")
    if raw_markers:
        try:
            kwargs["markers"] = json.loads(raw_markers) if isinstance(raw_markers, str) else raw_markers
        except Exception as exc:
            return _cx_response(
                {"error": f"Could not parse 'markers': {exc}", "success": False}, cx, 400
            )

    try:
        result = create_map_config(**kwargs)
    except Exception as exc:
        log.exception("REST /map error")
        return _cx_response({
            "config": {}, "valid": False, "success": False,
            "warnings": ["Could not generate map configuration: " + str(exc)],
            "map_id": map_id, "headers_used": [],
        }, cx, 200)

    result["html"] = _result_to_html(result, "map")
    return _cx_response(result, cx)


@mcp.custom_route("/feedback", methods=["GET", "POST"])
async def rest_feedback(request: Request) -> Response:
    """
    Submit thumbs-up / thumbs-down feedback for a previous tool call.

    GET  /feedback?callback=fn&request_id=UUID&rating=1   (JSONP)
    POST /feedback  body: { request_id, rating, comment? } (JSON)

      request_id  (str, required)  — UUID returned in the tool response
      rating      (int, required)  — 1 = thumbs up, -1 = thumbs down
      comment     (str, optional)  — free-text explanation
    """
    if request.method == "GET":
        params     = dict(request.query_params)
        call_id    = (params.get("request_id") or "").strip()
        callback   = (params.get("callback") or "").strip()
        try:
            rating = int(params.get("rating", 0))
        except (ValueError, TypeError):
            rating = 0
        comment = params.get("comment") or None
    else:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "JSON body required"}, status_code=400)
        call_id  = (body.get("request_id") or "").strip()
        callback = ""
        rating   = body.get("rating")
        comment  = body.get("comment") or None
    if not call_id:
        return JSONResponse({"error": "'request_id' is required"}, status_code=400)
    if rating not in (1, -1):
        return JSONResponse({"error": "'rating' must be 1 (up) or -1 (down)"}, status_code=400)

    row = _call_log.rate(call_id, rating, comment)
    if row is None:
        return JSONResponse({"error": f"No call found with request_id '{call_id}'"}, status_code=404)

    log.info("Feedback received: request_id=%s rating=%s target=%s", call_id, rating, row.get("target", ""))
    data = {
        "success":    True,
        "request_id": call_id,
        "rating":     rating,
        "target":     row.get("target", ""),
        "client_id":  row.get("client_id", ""),
        "datetime":   row.get("ts", ""),
    }
    if callback:
        import json as _json
        body = f"{callback}({_json.dumps(data)});"
        return Response(content=body, media_type="application/javascript")
    return JSONResponse(data)


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
