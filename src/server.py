#!/usr/bin/env python3
"""
CanvasXpress MCP Server v4 — HTTP Transport
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
from fastmcp import FastMCP
from llm_providers import complete as llm_complete, provider_info, PROVIDER, MODEL
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


# Load knowledge DB at startup
_knowledge_db: Optional["KnowledgeDB"] = None
KNOWLEDGE_FILE = DATA_DIR / "knowledge.db"


def _load_knowledge_db() -> bool:
    global _knowledge_db
    if not KNOWLEDGE_FILE.exists():
        log.warning(
            "Knowledge DB not found at %s. Run build_knowledge_db.py to build it. "
            "Falling back to base prompt only.", KNOWLEDGE_FILE
        )
        return False
    try:
        _knowledge_db = KnowledgeDB(KNOWLEDGE_FILE)
        stats = _knowledge_db.get_stats()
        log.info(
            "Knowledge DB: %d sections (t1=%d t2=%d t3=%d)",
            stats["total_sections"],
            stats["by_tier"].get(1, 0),
            stats["by_tier"].get(2, 0),
            stats["by_tier"].get(3, 0),
        )
        return True
    except Exception as e:
        log.warning("Failed to load knowledge DB: %s. Using base prompt only.", e)
        return False


_SYSTEM_PROMPT_HEADER = """You are an expert CanvasXpress data visualization assistant.

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
YlGn, YlGnBu, GnBu, BuGn, PuBuGn, PuBu, BuPu, RdPu, PuRd, OrRd, YlOrRd, YlOrBr,
Purples, Blues, Greens, Oranges, Reds, Greys, PuOr, BrBG, PRGn, PiYG, RdBu, RdGy,
RdYlBu, Spectral, RdYlGn, Bootstrap, Economist, Excel, GGPlot, Solarized, PaulTol,
ColorBlind, Tableau, WallStreetJournal, Stata, BlackAndWhite, CanvasXpress

## VALID THEMES
bw, classic, cx, dark, economist, excel, ggblanket, ggplot, gray, grey,
highcharts, igray, light, linedraw, minimal, none, ptol, solarized, stata, tableau, void0, wsj

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

    if _knowledge_db is not None:
        sections = _knowledge_db.get_sections(graph_type=graph_type, tier=tier)
        for s in sections:
            label = s["section"].upper().replace("_", " ")
            prompt += f"\n## {label} (from {s['source']})\n{s['content']}\n"
        if DEBUG:
            log.debug(
                "KnowledgeDB: tier=%d graph=%s sections=%d chars=%d",
                tier, graph_type or "?", len(sections), len(prompt)
            )
    else:
        log.warning("Knowledge DB not loaded — run build_knowledge_db.py for better results")

    return prompt, tier, graph_type


# Load knowledge DB now
_load_knowledge_db()

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
) -> dict | str:
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
# Config modification
# ---------------------------------------------------------------------------

def modify_config(
    config: dict,
    instruction: str,
    headers: list[str] | None = None,
    column_types: dict[str, str] | None = None,
    temperature: float = 0.0,
) -> dict:
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

    return modified


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
    modified = modify_config(config, instruction, resolved_headers, column_types, temperature)

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
        "config":       modified,
        "valid":        validation["valid"],
        "warnings":     validation["warnings"],
        "invalid_refs": validation["invalid_refs"],
        "headers_used": resolved_headers or [],
        "types_used":   column_types or {},
        "changes":      changes,
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
    mcp.run(transport="http", host=HOST, port=PORT)
