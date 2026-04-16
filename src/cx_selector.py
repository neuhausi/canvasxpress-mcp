"""
cx_selector.py — structured, deterministic chart-type recommender.

Fills the gap BEFORE generate_canvasxpress_config: given a dict of
{column_name: type} and a plain-English intent, it returns a ranked
list of CanvasXpress graphType candidates with rationale.

No LLM call. No API key. Zero extra dependencies.
Designed to be called first, then the chosen graphType passed to
generate_canvasxpress_config via the generate_hint field.
"""

from __future__ import annotations
from typing import Optional

# ---------------------------------------------------------------------------
# Column-type normalisation
# ---------------------------------------------------------------------------

_NUMERIC_ALIASES = {"numeric", "continuous", "integer", "float", "count", "number"}
_FACTOR_ALIASES  = {"factor", "string", "categorical", "nominal", "ordinal", "binary", "character"}
_TIME_ALIASES    = {"time", "date", "datetime", "temporal"}


def _count_types(column_types: dict[str, str]) -> tuple[int, int, int]:
    """Return (n_factor, n_numeric, n_time)."""
    n_fac  = sum(1 for t in column_types.values() if t.lower() in _FACTOR_ALIASES)
    n_num  = sum(1 for t in column_types.values() if t.lower() in _NUMERIC_ALIASES)
    n_time = sum(1 for t in column_types.values() if t.lower() in _TIME_ALIASES)
    return n_fac, n_num, n_time


# ---------------------------------------------------------------------------
# Chart catalogue
# ---------------------------------------------------------------------------

CHART_CATALOGUE: dict[str, dict] = {
    "Bar": {
        "category": "single_dimensional",
        "description": "Vertical or horizontal bars comparing values across categories.",
        "best_for": ["compare magnitudes", "ae count", "count by", "incidence", "frequency"],
        "requires": {"n_fac": 1, "n_num": 1},
        "clinical_use": "AE counts by SOC/arm, subject counts by group",
        "next_step": "generate_canvasxpress_config with description='grouped bar chart of <metric> by <category>'",
    },
    "Stacked": {
        "category": "single_dimensional",
        "description": "Stacked bars showing part-of-whole within each group.",
        "best_for": ["proportion", "part of whole", "composition", "breakdown", "percent"],
        "requires": {"n_fac": 1, "n_num": 1},
        "clinical_use": "AE severity breakdown per arm, disposition categories",
        "next_step": "generate_canvasxpress_config with description='stacked bar of <metric> by <category>'",
    },
    "StackedPercent": {
        "category": "single_dimensional",
        "description": "100% stacked bars — each bar sums to 100%.",
        "best_for": ["100 percent", "relative proportion", "normalised"],
        "requires": {"n_fac": 1, "n_num": 1},
        "clinical_use": "Relative AE severity distribution across arms",
        "next_step": "generate_canvasxpress_config with description='100% stacked bar of <metric> by <category>'",
    },
    "Boxplot": {
        "category": "single_dimensional",
        "description": "Box-and-whisker showing median, IQR, and outliers per group.",
        "best_for": ["distribution", "spread", "outlier", "variability", "lab value"],
        "requires": {"n_fac": 1, "n_num": 1},
        "clinical_use": "Lab value distributions by arm/visit, PK endpoints",
        "next_step": "generate_canvasxpress_config with description='boxplot of <numeric> grouped by <factor>'",
    },
    "Violin": {
        "category": "single_dimensional",
        "description": "Kernel density mirrored around a centre line — richer shape than boxplot.",
        "best_for": ["distribution shape", "bimodality", "gene expression"],
        "requires": {"n_fac": 1, "n_num": 1},
        "clinical_use": "Biomarker distributions, gene expression by cell type",
        "next_step": "generate_canvasxpress_config with description='violin plot of <numeric> by <factor>'",
    },
    "Dotplot": {
        "category": "single_dimensional",
        "description": "Individual data points per group — best when n < ~50 per group.",
        "best_for": ["individual patient", "small n", "show every observation"],
        "requires": {"n_fac": 1, "n_num": 1},
        "clinical_use": "Individual lab values, small Phase I cohorts",
        "next_step": "generate_canvasxpress_config with description='dot plot of <numeric> by <factor>'",
    },
    "Heatmap": {
        "category": "single_dimensional",
        "description": "Color-coded matrix — rows × columns with optional clustering and dendrograms.",
        "best_for": ["pattern", "heatmap", "gene expression", "ae matrix", "two categorical"],
        "requires": {"n_fac": 2, "n_num": 1},
        "clinical_use": "AE incidence by SOC × arm, gene expression heatmap",
        "next_step": "generate_canvasxpress_config with description='heatmap of <numeric> with <factor1> rows and <factor2> columns'",
    },
    "Line": {
        "category": "single_dimensional",
        "description": "Connected points showing trends over a continuous or ordinal axis.",
        "best_for": ["trend", "longitudinal", "visit", "pk", "profile", "over time"],
        "requires": {"n_time": 1, "n_num": 1},
        "clinical_use": "Lab values over visits, PK concentration-time profiles",
        "next_step": "generate_canvasxpress_config with description='line chart of <numeric> over <time> colored by <factor>'",
    },
    "Scatter2D": {
        "category": "multi_dimensional",
        "description": "Points on x/y axes revealing correlation between two numeric variables.",
        "best_for": ["correlation", "scatter", "regression", "pca", "umap", "bivariate"],
        "requires": {"n_num": 2},
        "clinical_use": "Biomarker vs response, PK/PD, PCA plot by treatment",
        "next_step": "generate_canvasxpress_config with description='scatter plot of <num1> vs <num2> colored by <factor>'",
    },
    "Scatter3D": {
        "category": "multi_dimensional",
        "description": "3D scatter for three continuous dimensions.",
        "best_for": ["3d", "three numeric", "three continuous"],
        "requires": {"n_num": 3},
        "clinical_use": "Multivariate exploratory analysis",
        "next_step": "generate_canvasxpress_config with description='3D scatter of <num1> vs <num2> vs <num3>'",
    },
    "Volcano": {
        "category": "multi_dimensional",
        "description": "Scatter of -log10(p-value) vs log2(fold change) for differential analysis.",
        "best_for": ["volcano", "differential", "fold change", "deg", "gwas", "significance"],
        "requires": {"n_num": 2},
        "clinical_use": "DEG analysis, biomarker discovery",
        "next_step": "generate_canvasxpress_config with description='volcano plot with log2FoldChange on x-axis and -log10 pvalue on y-axis'",
    },
    "KaplanMeier": {
        "category": "multi_dimensional",
        "description": "Kaplan-Meier survival curve with optional CI and risk table.",
        "best_for": ["survival", "kaplan", "km", "time-to-event", "overall survival", "progression free"],
        "requires": {"n_num": 1, "n_fac": 1},  # time col often typed numeric; n_time not enforced
        "clinical_use": "OS, PFS, EFS by treatment arm",
        "next_step": "generate_km_config — use the dedicated KM skill for best results",
    },
    "Correlation": {
        "category": "single_dimensional",
        "description": "Symmetric heatmap of pairwise correlation coefficients.",
        "best_for": ["correlation matrix", "pairwise", "variable relationships"],
        "requires": {"n_num": 2},
        "clinical_use": "Lab correlation matrix, biomarker panel correlations",
        "next_step": "generate_canvasxpress_config with description='correlation matrix heatmap'",
    },
    "Sankey": {
        "category": "single_dimensional",
        "description": "Flow diagram showing quantities moving between states or groups.",
        "best_for": ["flow", "journey", "disposition", "sankey", "alluvial", "path"],
        "requires": {"n_fac": 2, "n_num": 1},
        "clinical_use": "Patient disposition, treatment pathway flow",
        "next_step": "generate_canvasxpress_config with description='Sankey diagram of flow from <factor1> to <factor2>'",
    },
    "Waterfall": {
        "category": "single_dimensional",
        "description": "Sorted bars showing individual subject response (e.g. % tumour change).",
        "best_for": ["waterfall", "individual response", "tumour", "recist", "best change"],
        "requires": {"n_fac": 1, "n_num": 1},
        "clinical_use": "RECIST best % change from baseline, individual patient response",
        "next_step": "generate_canvasxpress_config with description='waterfall plot of best percentage change colored by <factor>'",
    },
    "Histogram": {
        "category": "single_dimensional",
        "description": "Binned frequency distribution of a single continuous variable.",
        "best_for": ["histogram", "frequency", "single variable distribution", "normality"],
        "requires": {"n_num": 1},
        "clinical_use": "Baseline characteristic distribution",
        "next_step": "generate_canvasxpress_config with description='histogram of <numeric>'",
    },
    "Density": {
        "category": "single_dimensional",
        "description": "Smooth kernel density estimate — overlay multiple groups.",
        "best_for": ["density", "smooth distribution", "kde", "overlay groups"],
        "requires": {"n_num": 1},
        "clinical_use": "Overlaid PK distributions across arms",
        "next_step": "generate_canvasxpress_config with description='density plot of <numeric> grouped by <factor>'",
    },
    "Treemap": {
        "category": "single_dimensional",
        "description": "Nested rectangles sized by value — hierarchical proportion.",
        "best_for": ["treemap", "hierarchical", "proportional area", "nested categories"],
        "requires": {"n_fac": 1, "n_num": 1},
        "clinical_use": "Hierarchical AE counts (SOC → PT)",
        "next_step": "generate_canvasxpress_config with description='treemap of <numeric> by <factor> hierarchy'",
    },
    "Network": {
        "category": "network",
        "description": "Nodes and edges for relational or pathway data.",
        "best_for": ["network", "pathway", "protein interaction", "graph", "edges", "nodes"],
        "requires": {},
        "clinical_use": "Pathway enrichment, protein-protein interaction",
        "next_step": "generate_canvasxpress_config with description='network graph of pathways'",
    },
    "Venn": {
        "category": "single_dimensional",
        "description": "Overlapping circles showing set intersections.",
        "best_for": ["venn", "overlap", "set intersection", "gene lists"],
        "requires": {},
        "clinical_use": "Overlapping AE populations, gene set comparison",
        "next_step": "generate_canvasxpress_config with description='Venn diagram of overlapping sets'",
    },
}

# ---------------------------------------------------------------------------
# Decision rules — ordered, first keyword match wins
# ---------------------------------------------------------------------------

_RULES: list[dict] = [
    {"kws": ["kaplan", "km ", "survival", "overall survival", "progression free",
             "time-to-event", "efs", " os ", " pfs "],
     "types": ["KaplanMeier"]},
    {"kws": ["volcano", "fold change", "deg", "differential expression", "gwas"],
     "types": ["Volcano"]},
    {"kws": ["waterfall", "recist", "tumour shrinkage", "best change", "best percentage"],
     "types": ["Waterfall"]},
    {"kws": ["sankey", "alluvial", "flow", "disposition", "patient journey"],
     "types": ["Sankey"]},
    {"kws": ["network", "pathway", "protein interaction"],
     "types": ["Network"]},
    {"kws": ["venn", "set overlap", "intersection"],
     "types": ["Venn"]},
    {"kws": ["treemap", "tree map", "hierarchy"],
     "types": ["Treemap"]},
    {"kws": ["correlation matrix", "pairwise correlation"],
     "types": ["Correlation", "Heatmap"]},
    {"kws": ["heatmap", "heat map", "ae matrix", "gene expression matrix"],
     "types": ["Heatmap", "Correlation"]},
    {"kws": ["pca", "umap", "tsne", "t-sne", "embedding", "dimensionality"],
     "types": ["Scatter2D", "Scatter3D"]},
    {"kws": ["scatter", "regression", "bivariate"],
     "types": ["Scatter2D", "Correlation"]},
    {"kws": ["3d", "three dimensional", "three numeric"],
     "types": ["Scatter3D"]},
    {"kws": ["violin"],
     "types": ["Violin", "Boxplot"]},
    {"kws": ["distribution", "spread", "variability", "iqr", "outlier"],
     "types": ["Boxplot", "Violin", "Dotplot"]},
    {"kws": ["histogram", "frequency distribution"],
     "types": ["Histogram", "Density"]},
    {"kws": ["density", "kde"],
     "types": ["Density", "Histogram"]},
    {"kws": ["trend", "longitudinal", "over time", "over visit", "pk profile", "concentration"],
     "types": ["Line", "Area"]},
    {"kws": ["100%", "100 percent", "percent stack", "relative proportion"],
     "types": ["StackedPercent", "Stacked"]},
    {"kws": ["proportion", "part of whole", "composition", "breakdown", "percentage"],
     "types": ["Stacked", "StackedPercent"]},
    {"kws": ["pattern", "across groups", "across arms", "ae incidence by"],
     "types": ["Heatmap", "Bar"]},
    {"kws": ["individual patient", "individual subject", "per subject", "small n"],
     "types": ["Dotplot", "Scatter2D"]},
    # default magnitude comparison — catches "count by", "by soc", "by arm", etc.
    {"kws": ["count", "frequency", "incidence", "ae count", "compare", "by soc",
             "by arm", "magnitude", "bar"],
     "types": ["Bar", "Stacked", "Heatmap"]},
]


def _requirements_met(graph_type: str, n_fac: int, n_num: int, n_time: int) -> bool:
    req = CHART_CATALOGUE.get(graph_type, {}).get("requires", {})
    if req.get("n_fac", 0) > n_fac:
        return False
    if req.get("n_num", 0) > n_num:
        return False
    if req.get("n_time", 0) > n_time:
        return False
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def select_chart(
    intent: str,
    column_types: dict[str, str],
    n_samples: Optional[int] = None,
) -> dict:
    """
    Recommend CanvasXpress graphType(s) from column types + analytical intent.

    Args:
        intent:       Plain-English description of what you want to show.
                      e.g. "AE counts by SOC across 3 treatment arms"
        column_types: {column_name: type} where type is one of:
                      numeric | factor | string | date | integer | binary
        n_samples:    Optional row count — nudges Dotplot over Boxplot for small n (<30)
                      and warns about overplotting for large n (>5000).

    Returns:
        {
          intent             (str)  - echoed back
          column_summary     (dict) - {n_factor, n_numeric, n_time}
          top_recommendation (dict) - best graphType with rationale
          alternatives       (list) - up to 3 other candidates
          generate_hint      (str)  - suggested description to pass to
                                      generate_canvasxpress_config
        }
    """
    intent_lower = intent.lower()
    n_fac, n_num, n_time = _count_types(column_types)

    candidates: list[str] = []
    seen: set[str] = set()

    for rule in _RULES:
        if not any(kw in intent_lower for kw in rule["kws"]):
            continue
        for gt in rule["types"]:
            if gt not in seen and _requirements_met(gt, n_fac, n_num, n_time):
                seen.add(gt)
                candidates.append(gt)

    # Fallback when no rule matched
    if not candidates:
        fallback = "Bar" if n_fac >= 1 and n_num >= 1 else "Scatter2D"
        candidates = [fallback]

    # Small-n nudge: make Dotplot the top recommendation for tiny cohorts
    if n_samples is not None and n_samples < 30:
        if "Dotplot" not in candidates:
            candidates.insert(0, "Dotplot")
        elif candidates[0] != "Dotplot":
            candidates.remove("Dotplot")
            candidates.insert(0, "Dotplot")

    def _enrich(gt: str) -> dict:
        info = CHART_CATALOGUE.get(gt, {})
        out: dict = {
            "graphType":    gt,
            "category":     info.get("category", ""),
            "description":  info.get("description", ""),
            "clinical_use": info.get("clinical_use", ""),
            "next_step":    info.get(
                "next_step",
                f"generate_canvasxpress_config with description='{gt} chart'",
            ),
        }
        if n_samples is not None and n_samples < 30 and gt == "Dotplot":
            out["note"] = (
                f"n_samples={n_samples} is small — Dotplot shows every observation clearly."
            )
        if n_samples is not None and n_samples > 5000 and gt in {"Scatter2D", "Dotplot"}:
            out["note"] = (
                f"n_samples={n_samples} is large — consider BinPlot or Hexplot "
                "to avoid overplotting."
            )
        return out

    top  = _enrich(candidates[0])
    alts = [_enrich(gt) for gt in candidates[1:4]]

    # Build a ready-made description hint for generate_canvasxpress_config
    col_names    = list(column_types.keys())
    factor_cols  = [k for k, v in column_types.items() if v.lower() in _FACTOR_ALIASES]
    numeric_cols = [k for k, v in column_types.items() if v.lower() in _NUMERIC_ALIASES]
    time_cols    = [k for k, v in column_types.items() if v.lower() in _TIME_ALIASES]

    if factor_cols and numeric_cols:
        axis_part = (
            f"over {time_cols[0]}" if time_cols else f"grouped by {factor_cols[0]}"
        )
        color_part = (
            f" colored by {factor_cols[1]}" if len(factor_cols) > 1 else ""
        )
        generate_hint = (
            f"{candidates[0]} chart of {numeric_cols[0]} {axis_part}"
            f"{color_part} — columns: {', '.join(col_names)}"
        )
    else:
        generate_hint = f"{candidates[0]} chart — columns: {', '.join(col_names)}"

    return {
        "intent":             intent,
        "column_summary":     {
            "n_factor":  n_fac,
            "n_numeric": n_num,
            "n_time":    n_time,
        },
        "top_recommendation": top,
        "alternatives":       alts,
        "generate_hint":      generate_hint,
    }
