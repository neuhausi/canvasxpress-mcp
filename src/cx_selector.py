"""
cx_selector.py — structured, deterministic chart-type recommender.

Two-layer scoring system ported from recommendCharts.es5.js:
  Layer 1: Structural scores  (column type counts + row count + cardinality)
  Layer 2: Semantic adjustments (column name regex pattern matching)
  Layer 3: Intent boosts       (clinical/domain keyword overrides from intent)

No LLM call. No API key. Zero extra dependencies.
Designed to be called first, then the chosen graphType passed to
generate_canvasxpress_config via the generate_hint field.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Column-type normalisation
# ---------------------------------------------------------------------------

_NUMERIC_ALIASES = {"numeric", "continuous", "integer", "float", "count", "number"}
_FACTOR_ALIASES  = {"factor", "string", "categorical", "nominal", "ordinal", "binary", "character"}
_TIME_ALIASES    = {"time", "date", "datetime", "temporal"}
_BOOL_ALIASES    = {"boolean", "bool", "logical", "flag"}
_TEXT_ALIASES    = {"text", "freetext", "free_text", "long_text", "varchar"}


def _count_types(column_types: dict[str, str]) -> tuple[int, int, int, int, int]:
    """Return (n_factor, n_numeric, n_time, n_bool, n_text)."""
    n_fac  = sum(1 for t in column_types.values() if t.lower() in _FACTOR_ALIASES)
    n_num  = sum(1 for t in column_types.values() if t.lower() in _NUMERIC_ALIASES)
    n_time = sum(1 for t in column_types.values() if t.lower() in _TIME_ALIASES)
    n_bool = sum(1 for t in column_types.values() if t.lower() in _BOOL_ALIASES)
    n_text = sum(1 for t in column_types.values() if t.lower() in _TEXT_ALIASES)
    return n_fac, n_num, n_time, n_bool, n_text



# ---------------------------------------------------------------------------
# Chart catalogue  (53 graphTypes)
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
    "Area": {
        "category": "single_dimensional",
        "description": "Line chart with filled area beneath — emphasises volume over time.",
        "best_for": ["area", "cumulative", "volume over time"],
        "requires": {"n_time": 1, "n_num": 1},
        "clinical_use": "Cumulative event counts, drug exposure over time",
        "next_step": "generate_canvasxpress_config with description='area chart of <numeric> over <time>'",
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
        "requires": {"n_num": 1, "n_fac": 1},
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
    # ----- additional chart types -----
    "BarLine": {
        "category": "multi_dimensional",
        "description": "Combined bar + line on dual axes — compare count (bars) with rate (line).",
        "best_for": ["bar line", "dual axis", "count and rate"],
        "requires": {"n_fac": 1, "n_num": 2},
        "clinical_use": "AE count bars with cumulative incidence line",
        "next_step": "generate_canvasxpress_config with description='bar-line chart of <num1> bars and <num2> line by <factor>'",
    },
    "Ridgeline": {
        "category": "single_dimensional",
        "description": "Stacked density plots per group — compare distributions without overlap.",
        "best_for": ["ridgeline", "ridge plot", "joy plot", "group distributions"],
        "requires": {"n_fac": 1, "n_num": 1},
        "clinical_use": "Lab distributions across multiple visits or arms",
        "next_step": "generate_canvasxpress_config with description='ridgeline plot of <numeric> by <factor>'",
    },
    "Hexplot": {
        "category": "multi_dimensional",
        "description": "Hexagonal binning for large scatter datasets to avoid overplotting.",
        "best_for": ["hexplot", "hex bin", "large scatter", "overplotting", "density scatter"],
        "requires": {"n_num": 2},
        "clinical_use": "Large-scale biomarker vs response, high-n scatter",
        "next_step": "generate_canvasxpress_config with description='hexplot of <num1> vs <num2>'",
    },
    "Contour": {
        "category": "multi_dimensional",
        "description": "2D density contour lines over a scatter.",
        "best_for": ["contour", "2d density", "density contour"],
        "requires": {"n_num": 2},
        "clinical_use": "Population density of two continuous biomarkers",
        "next_step": "generate_canvasxpress_config with description='contour plot of <num1> vs <num2>'",
    },
    "Sunburst": {
        "category": "single_dimensional",
        "description": "Radial hierarchical treemap — outer rings are children.",
        "best_for": ["sunburst", "radial hierarchy", "multilevel pie"],
        "requires": {"n_fac": 2, "n_num": 1},
        "clinical_use": "Hierarchical AE breakdown (SOC → PT → severity)",
        "next_step": "generate_canvasxpress_config with description='sunburst of <numeric> by nested <factor> hierarchy'",
    },
    "Chord": {
        "category": "network",
        "description": "Circular chord diagram — pairwise flow between categories.",
        "best_for": ["chord", "circular flow", "pairwise flow", "migration"],
        "requires": {"n_fac": 2, "n_num": 1},
        "clinical_use": "Cross-arm patient movement, co-medication flows",
        "next_step": "generate_canvasxpress_config with description='chord diagram of flow between <factor1> and <factor2>'",
    },
    "ParallelCoordinates": {
        "category": "multi_dimensional",
        "description": "Parallel axes — each line is an observation, axes are variables.",
        "best_for": ["parallel coordinates", "multivariate", "multiple variables", "high dimensional"],
        "requires": {"n_num": 3},
        "clinical_use": "Multi-endpoint profile per subject, multivariate QC",
        "next_step": "generate_canvasxpress_config with description='parallel coordinates of <numeric variables> colored by <factor>'",
    },
    "SPLOM": {
        "category": "multi_dimensional",
        "description": "Scatter plot matrix — all pairwise scatter plots in a grid.",
        "best_for": ["splom", "scatter matrix", "pairs plot", "pairwise scatter"],
        "requires": {"n_num": 3},
        "clinical_use": "Pairwise biomarker exploration, PK parameter matrix",
        "next_step": "generate_canvasxpress_config with description='scatter plot matrix of <numeric variables>'",
    },
    "Radar": {
        "category": "multi_dimensional",
        "description": "Spider/radar chart — compare profiles across multiple axes.",
        "best_for": ["radar", "spider", "radial", "profile comparison", "multivariate comparison"],
        "requires": {"n_fac": 1, "n_num": 3},
        "clinical_use": "Safety profile comparison across arms",
        "next_step": "generate_canvasxpress_config with description='radar chart of <numeric axes> by <factor>'",
    },
    "Streamgraph": {
        "category": "single_dimensional",
        "description": "Stacked area with smooth flow — shows relative composition over time.",
        "best_for": ["streamgraph", "stream", "flow over time", "stacked area"],
        "requires": {"n_time": 1, "n_fac": 1, "n_num": 1},
        "clinical_use": "Concomitant medication use over study duration",
        "next_step": "generate_canvasxpress_config with description='streamgraph of <numeric> by <factor> over <time>'",
    },
    "Gantt": {
        "category": "single_dimensional",
        "description": "Horizontal bars with start/end times — subject-level timelines.",
        "best_for": ["gantt", "timeline", "exposure", "treatment duration", "interval"],
        "requires": {"n_fac": 1, "n_time": 2},
        "clinical_use": "Subject treatment timelines, concomitant medication exposure",
        "next_step": "generate_canvasxpress_config with description='Gantt chart of <factor> from <start_time> to <end_time>'",
    },
    "Lollipop": {
        "category": "single_dimensional",
        "description": "Point-on-stem variant of bar chart — cleaner for many categories.",
        "best_for": ["lollipop", "dot bar", "many categories"],
        "requires": {"n_fac": 1, "n_num": 1},
        "clinical_use": "Gene/protein expression ranked list",
        "next_step": "generate_canvasxpress_config with description='lollipop chart of <numeric> by <factor>'",
    },
    "TagCloud": {
        "category": "single_dimensional",
        "description": "Word/tag cloud sized by frequency.",
        "best_for": ["word cloud", "tag cloud", "text frequency", "term frequency"],
        "requires": {"n_fac": 1},
        "clinical_use": "Frequently reported terms, SAE narratives word frequency",
        "next_step": "generate_canvasxpress_config with description='word cloud of <factor> sized by frequency'",
    },
    "DotLine": {
        "category": "single_dimensional",
        "description": "Forest-plot style: horizontal CI lines with central estimate dot.",
        "best_for": ["forest plot", "confidence interval", "meta analysis", "hazard ratio"],
        "requires": {"n_fac": 1, "n_num": 1},
        "clinical_use": "Hazard ratios, odds ratios, risk differences",
        "next_step": "generate_canvasxpress_config with description='forest plot of <estimate> with CI by <factor>'",
    },
    "ScatterBubble2D": {
        "category": "multi_dimensional",
        "description": "Bubble chart — 2D scatter where bubble size encodes a third variable.",
        "best_for": ["bubble", "three dimensions", "size encoded", "bubble chart"],
        "requires": {"n_num": 3},
        "clinical_use": "Efficacy vs safety vs sample size, three-dimensional biomarker",
        "next_step": "generate_canvasxpress_config with description='bubble chart of <num1> vs <num2> sized by <num3>'",
    },
    # ----- alias chart types -----
    "Alluvial": {
        "category": "single_dimensional",
        "description": "Alluvial / Sankey-style flow diagram.",
        "best_for": ["alluvial", "flow diagram", "category flow"],
        "requires": {"n_fac": 2, "n_num": 1},
        "clinical_use": "Patient pathway flow across study visits",
        "next_step": "generate_canvasxpress_config with description='alluvial diagram from <factor1> to <factor2>'",
    },
    "Binplot": {
        "category": "multi_dimensional",
        "description": "Binned scatter — alternative to Hexplot for large datasets.",
        "best_for": ["binplot", "bin plot", "binned scatter"],
        "requires": {"n_num": 2},
        "clinical_use": "High-density scatter of continuous biomarkers",
        "next_step": "generate_canvasxpress_config with description='binplot of <num1> vs <num2>'",
    },
    "Bubble": {
        "category": "multi_dimensional",
        "description": "Bubble chart alias — scatter with size-encoded third variable.",
        "best_for": ["bubble chart", "bubble plot"],
        "requires": {"n_num": 3},
        "clinical_use": "Three-dimensional biomarker exploration",
        "next_step": "generate_canvasxpress_config with description='bubble chart of <num1> vs <num2> sized by <num3>'",
    },
    "Bullet": {
        "category": "single_dimensional",
        "description": "Bullet chart — performance vs target reference line.",
        "best_for": ["bullet chart", "target", "kpi", "gauge"],
        "requires": {"n_fac": 1, "n_num": 2},
        "clinical_use": "Endpoint vs pre-specified threshold",
        "next_step": "generate_canvasxpress_config with description='bullet chart of <metric> vs target by <factor>'",
    },
    "Bump": {
        "category": "single_dimensional",
        "description": "Rank-over-time (bump) chart — tracks rank changes longitudinally.",
        "best_for": ["bump chart", "rank over time", "ranking change"],
        "requires": {"n_fac": 1, "n_time": 1, "n_num": 1},
        "clinical_use": "Site or patient ranking across visits",
        "next_step": "generate_canvasxpress_config with description='bump chart of <factor> rank over <time>'",
    },
    "CDF": {
        "category": "single_dimensional",
        "description": "Cumulative distribution function plot.",
        "best_for": ["cdf", "cumulative distribution", "ecdf", "percentile"],
        "requires": {"n_num": 1},
        "clinical_use": "CDF of time-to-event, PK exposure CDF",
        "next_step": "generate_canvasxpress_config with description='CDF of <numeric>'",
    },
    "Cleveland": {
        "category": "single_dimensional",
        "description": "Cleveland dot plot — cleaner alternative to bar chart for many categories.",
        "best_for": ["cleveland", "dot chart", "cleveland plot"],
        "requires": {"n_fac": 1, "n_num": 1},
        "clinical_use": "Incidence rates by SOC/PT sorted list",
        "next_step": "generate_canvasxpress_config with description='Cleveland dot plot of <numeric> by <factor>'",
    },
    "Dumbbell": {
        "category": "single_dimensional",
        "description": "Dumbbell / connected dot plot — before/after comparison per category.",
        "best_for": ["dumbbell", "before after", "paired comparison", "change from baseline"],
        "requires": {"n_fac": 1, "n_num": 2},
        "clinical_use": "Baseline vs post-treatment comparison by patient group",
        "next_step": "generate_canvasxpress_config with description='dumbbell plot of <baseline> to <followup> by <factor>'",
    },
    "Pareto": {
        "category": "single_dimensional",
        "description": "Bar chart sorted descending with cumulative % line (80/20 rule).",
        "best_for": ["pareto", "80 20", "cumulative percent", "ranked bar"],
        "requires": {"n_fac": 1, "n_num": 1},
        "clinical_use": "Top AEs by frequency with cumulative %",
        "next_step": "generate_canvasxpress_config with description='Pareto chart of <numeric> by <factor>'",
    },
    "QQ": {
        "category": "multi_dimensional",
        "description": "Quantile-quantile plot for normality or distribution comparison.",
        "best_for": ["qq plot", "quantile quantile", "normality test", "qqplot"],
        "requires": {"n_num": 1},
        "clinical_use": "Normality assessment of lab endpoints",
        "next_step": "generate_canvasxpress_config with description='QQ plot of <numeric>'",
    },
    "Ribbon": {
        "category": "single_dimensional",
        "description": "Ribbon chart — area between two lines showing range over time.",
        "best_for": ["ribbon", "band chart", "range band", "confidence band"],
        "requires": {"n_time": 1, "n_num": 2},
        "clinical_use": "Mean ± SD lab value band over visits",
        "next_step": "generate_canvasxpress_config with description='ribbon chart of <lower> to <upper> over <time>'",
    },
    "Spaghetti": {
        "category": "single_dimensional",
        "description": "Individual trajectory lines per subject over time.",
        "best_for": ["spaghetti", "individual trajectories", "per subject", "subject level"],
        "requires": {"n_time": 1, "n_num": 1, "n_fac": 1},
        "clinical_use": "Individual patient lab trajectories, PK per-subject profiles",
        "next_step": "generate_canvasxpress_config with description='spaghetti plot of <numeric> over <time> per <subject_id>'",
    },
    "TimeSeries": {
        "category": "single_dimensional",
        "description": "Generic time series line chart.",
        "best_for": ["time series", "timeseries", "temporal trend"],
        "requires": {"n_time": 1, "n_num": 1},
        "clinical_use": "Lab values, vital signs over study time",
        "next_step": "generate_canvasxpress_config with description='time series of <numeric> over <time>'",
    },
    "Tornado": {
        "category": "single_dimensional",
        "description": "Tornado (butterfly) chart — diverging bars from centre for sensitivity analysis.",
        "best_for": ["tornado", "butterfly chart", "sensitivity", "diverging bar", "forest sensitivity"],
        "requires": {"n_fac": 1, "n_num": 2},
        "clinical_use": "Sensitivity analysis of model parameters, risk factor comparison",
        "next_step": "generate_canvasxpress_config with description='tornado chart of <low> vs <high> by <factor>'",
    },
    "TreeBracket": {
        "category": "single_dimensional",
        "description": "Tree/dendrogram with bracket annotations.",
        "best_for": ["tree bracket", "dendrogram", "hierarchical clustering", "phylogenetic"],
        "requires": {"n_fac": 1},
        "clinical_use": "Hierarchical clustering of samples or genes",
        "next_step": "generate_canvasxpress_config with description='tree bracket dendrogram of <factor>'",
    },
    "Upset": {
        "category": "single_dimensional",
        "description": "UpSet plot — scalable alternative to Venn for many sets.",
        "best_for": ["upset", "upset plot", "multiple set intersection", "many sets"],
        "requires": {"n_fac": 2},
        "clinical_use": "Multi-way AE overlap, multi-drug co-occurrence",
        "next_step": "generate_canvasxpress_config with description='UpSet plot of set intersections across <factors>'",
    },
    "Map": {
        "category": "geographic",
        "description": "Choropleth, bubble, pie or marker map — values encoded onto geographic regions or coordinates.",
        "best_for": ["map", "choropleth", "geographic", "geospatial", "country", "state", "region",
                     "world map", "usa map", "us map", "pie map", "marker map", "latitude", "longitude"],
        "requires": {"n_fac": 1, "n_num": 1},
        "clinical_use": "Site enrollment by country/region, prevalence by geography, election results by state",
        "next_step": "create_map_config with map_id='<USAStates|World|USACounties|…>' and color_by='<column>'",
    },
}


# ---------------------------------------------------------------------------
# Regex patterns  (ported from PATTERNS in recommendCharts.es5.js)
# ---------------------------------------------------------------------------

_PATTERNS_RAW: dict[str, list[str]] = {
    "pvalue":         [r"p[\s_]?val", r"pval", r"p\.value", r"significance", r"sig$", r"adj[\s_]?p"],
    "foldchange":     [r"fold[\s_]?change", r"fc$", r"log2fc", r"log2[\s_]?fold", r"lfc"],
    "expression":     [r"expr", r"fpkm", r"rpkm", r"tpm", r"cpm", r"count$", r"reads", r"umi"],
    "survival":       [r"surv", r"os$", r"pfs$", r"efs$", r"dfs$", r"ttp$", r"time[\s_]?to"],
    "event":          [r"event$", r"status$", r"censor", r"death", r"progress"],
    "time":           [r"time$", r"day$", r"week$", r"month$", r"year$", r"date", r"visit"],
    "group":          [r"arm$", r"group$", r"cohort$", r"treatment$", r"trt$", r"therapy$"],
    "subject":        [r"subject", r"patient", r"participant", r"id$", r"subj", r"ptid", r"^sample\d*$"],
    "genomic":        [r"gene$", r"gene_", r"symbol$", r"hugo", r"entrez", r"ensembl", r"locus"],
    "chromosome":     [r"chr$", r"chrom", r"chromosome", r"position", r"pos$", r"bp$"],
    "effect":         [r"effect", r"beta$", r"coef", r"estimate", r"or$", r"hr$", r"rr$"],
    "confidence":     [r"ci_", r"ci$", r"lower", r"upper", r"lb$", r"ub$", r"conf"],
    "frequency":      [r"freq", r"count$", r"n_", r"num_", r"incidence", r"rate$"],
    "category":       [r"cat$", r"category", r"class$", r"type$", r"grade$", r"soc$", r"pt$"],
    "severity":       [r"sever", r"grade$", r"ctcae", r"toxicity", r"ae_grade"],
    "response":       [r"response", r"recist", r"bcr", r"best", r"change", r"pct_change", r"delta"],
    "biomarker":      [r"marker", r"biomarker", r"level$", r"conc", r"concentration", r"titer"],
    "pk":             [r"auc$", r"cmax$", r"tmax$", r"t1_2", r"half[\s_]?life", r"cl$", r"vd$"],
    "correlation":    [r"corr", r"pearson", r"spearman", r"rho$", r"r2$", r"r_squared"],
    "proportion":     [r"prop$", r"pct$", r"percent", r"ratio$", r"fraction"],
    "weight":         [r"weight$", r"wt$", r"mass$", r"size$", r"area$"],
    "source":         [r"source$", r"from$", r"origin"],
    "target":         [r"target$", r"to$", r"dest", r"destination"],
    "node":           [r"node$", r"vertex", r"vertices", r"gene_a", r"gene_b"],
    "edge":           [r"edge$", r"link$", r"interact", r"pathway$"],
    "set":            [r"set$", r"^set[a-z0-9]", r"list$", r"group_", r"genes_"],
    "hierarchy":      [r"parent$", r"child$", r"level$", r"depth$", r"hier"],
    "start":          [r"start$", r"begin$", r"onset$", r"from_date"],
    "end":            [r"end$", r"stop$", r"finish$", r"to_date"],
    "x":              [r"^x$", r"x_", r"_x$", r"x\d$", r"xval"],
    "y":              [r"^y$", r"y_", r"_y$", r"y\d$", r"yval"],
    "z":              [r"^z$", r"z_", r"_z$", r"z\d$", r"zval"],
    "component":      [r"pc\d", r"pca\d", r"component\d", r"dim\d", r"umap\d", r"tsne\d"],
    "cluster":        [r"cluster", r"clust$", r"group$", r"celltype", r"cell_type"],
    "label":          [r"label$", r"name$", r"word$", r"term$", r"tag$", r"text$"],
    "frequency2":     [r"freq$", r"weight$", r"count$", r"n$", r"tf$"],
    "rank":           [r"rank$", r"position$", r"order$", r"place$"],
    "lower":          [r"lower", r"low$", r"min$", r"lb$", r"q1$", r"p25$"],
    "upper":          [r"upper", r"high$", r"max$", r"ub$", r"q3$", r"p75$"],
    "sd_sem":         [r"^sd$", r"^se$", r"^sem$", r"std", r"stderr", r"error$", r"_sd$", r"_se$"],
    "geographic":     [r"^country$", r"^nation$", r"^state$", r"^province$", r"^fips$", r"^iso$", r"^iso2$", r"^iso3$", r"^lat$", r"^lng$", r"^lon$", r"latitude", r"longitude"],
}

_PATTERNS: dict[str, list[re.Pattern]] = {
    k: [re.compile(p, re.IGNORECASE) for p in pats]
    for k, pats in _PATTERNS_RAW.items()
}


def _col_matches(col: str, group: str) -> bool:
    return any(p.search(col) for p in _PATTERNS.get(group, []))


def _any_col_matches(cols: list[str], group: str) -> bool:
    return any(_col_matches(c, group) for c in cols)


def _count_col_matches(cols: list[str], group: str) -> int:
    return sum(1 for c in cols if _col_matches(c, group))


# ---------------------------------------------------------------------------
# Context dataclass
# ---------------------------------------------------------------------------

@dataclass
class _Ctx:
    numeric:         int   # n_numeric columns
    category:        int   # n_factor columns
    datetime:        int   # n_time columns
    boolean:         int   # n_bool columns
    text:            int   # n_text columns
    row_count:       int   # n_samples
    min_cat_unique:  int   # min unique values among categorical cols
    max_cat_unique:  int   # max unique values among categorical cols
    cat0:            int   # unique values in first categorical col (0 if none)
    has_high_variance: bool


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


# ---------------------------------------------------------------------------
# Layer 1: Structural scorer functions  (one per chart type)
# ---------------------------------------------------------------------------

def _score_bar(ctx: _Ctx) -> dict:
    score = 0.0
    factors = []
    if ctx.category >= 1 and ctx.numeric >= 1:
        score += 0.6
        factors.append("1+ category + 1+ numeric")
    if ctx.category == 1 and ctx.numeric == 1:
        score += 0.2
        factors.append("exactly 1 cat, 1 num — classic bar")
    if ctx.row_count < 200:
        score += 0.1
        factors.append("small-medium dataset")
    if ctx.cat0 and ctx.cat0 <= 20:
        score += 0.1
        factors.append("manageable category count")
    return {"score": _clamp(score), "factors": factors}


def _score_stacked(ctx: _Ctx) -> dict:
    score = 0.0
    factors = []
    if ctx.category >= 2 and ctx.numeric >= 1:
        score += 0.7
        factors.append("2+ categories ideal for stacking")
    elif ctx.category >= 1 and ctx.numeric >= 1:
        score += 0.4
        factors.append("1 cat + numeric — possible stacked")
    if ctx.cat0 and 2 <= ctx.cat0 <= 8:
        score += 0.2
        factors.append("few stack levels")
    return {"score": _clamp(score), "factors": factors}


def _score_stacked_percent(ctx: _Ctx) -> dict:
    score = 0.0
    factors = []
    if ctx.category >= 2 and ctx.numeric >= 1:
        score += 0.65
        factors.append("2+ cats → proportion comparison")
    elif ctx.category >= 1 and ctx.numeric >= 2:
        score += 0.55
        factors.append("wide format: 1 cat + multiple numeric levels → 100% stacked")
    if ctx.cat0 and 2 <= ctx.cat0 <= 6:
        score += 0.2
        factors.append("few stack segments")
    return {"score": _clamp(score), "factors": factors}


def _score_boxplot(ctx: _Ctx) -> dict:
    if ctx.category < 1:
        return {"score": 0.0, "factors": ["no grouping factor — boxplot requires categories"]}
    if ctx.cat0 and ctx.row_count and ctx.cat0 >= ctx.row_count * 0.9:
        return {"score": 0.0, "factors": ["each category appears once — data is already aggregated, no distribution to plot"]}
    score = 0.0
    factors = []
    if ctx.numeric >= 1:
        score += 0.6
        factors.append("cat + numeric → distribution by group")
    if ctx.row_count >= 20:
        score += 0.2
        factors.append("sufficient n for box stats")
    if ctx.numeric >= 2:
        score += 0.1
        factors.append("multiple numeric vars")
    return {"score": _clamp(score), "factors": factors}


def _score_violin(ctx: _Ctx) -> dict:
    if ctx.category < 1:
        return {"score": 0.0, "factors": ["no grouping factor — violin requires categories"]}
    if ctx.cat0 and ctx.row_count and ctx.cat0 >= ctx.row_count * 0.9:
        return {"score": 0.0, "factors": ["each category appears once — data is already aggregated, no distribution to plot"]}
    score = 0.0
    factors = []
    if ctx.numeric >= 1:
        score += 0.55
        factors.append("cat + numeric → distribution shape")
    if ctx.row_count >= 50:
        score += 0.25
        factors.append("enough data for density estimate")
    if ctx.has_high_variance:
        score += 0.1
        factors.append("high variance → shape informative")
    if ctx.cat0 and ctx.cat0 > 4:
        score -= 0.15
        factors.append("5+ groups — ridgeline avoids overlapping violins")
    return {"score": _clamp(score), "factors": factors}


def _score_dotplot(ctx: _Ctx) -> dict:
    if ctx.category < 1:
        return {"score": 0.0, "factors": ["no grouping factor — dotplot requires categories"]}
    score = 0.0
    factors = []
    if ctx.numeric >= 1:
        score += 0.5
        factors.append("cat + numeric")
    if ctx.row_count < 50:
        score += 0.35
        factors.append("small n — show every point")
    elif ctx.row_count < 200:
        score += 0.1
        factors.append("medium n — dotplot still viable")
    return {"score": _clamp(score), "factors": factors}


def _score_heatmap(ctx: _Ctx) -> dict:
    score = 0.0
    factors = []
    if ctx.category >= 2 and ctx.numeric >= 1:
        score += 0.7
        factors.append("2 cats as row/col + numeric value")
    if ctx.category >= 1 and ctx.numeric >= 4:
        score += 0.5
        factors.append("4+ numeric vars — wide-format matrix view preferred over parallel coordinates")
    if ctx.numeric >= 5:
        score += 0.1
        factors.append("many numeric vars → matrix view")
    if ctx.row_count > 10:
        score += 0.05
        factors.append("enough rows for matrix")
    if ctx.numeric == 1 and ctx.category == 2:
        score -= 0.15
        factors.append("single numeric col — treemap may be cleaner than heatmap")
    return {"score": _clamp(score), "factors": factors}


def _score_line(ctx: _Ctx) -> dict:
    score = 0.0
    factors = []
    if ctx.datetime >= 1 and ctx.numeric >= 1:
        score += 0.75
        factors.append("time + numeric → trend line")
    if ctx.category >= 1:
        score += 0.1
        factors.append("colour by category")
    if ctx.row_count > 5:
        score += 0.1
        factors.append("enough time points")
    return {"score": _clamp(score), "factors": factors}


def _score_area(ctx: _Ctx) -> dict:
    score = 0.0
    factors = []
    if ctx.datetime >= 1 and ctx.numeric >= 1:
        score += 0.65
        factors.append("time + numeric → area trend")
    if ctx.category >= 1:
        score += 0.1
        factors.append("stacked area by category")
    return {"score": _clamp(score), "factors": factors}


def _score_scatter2d(ctx: _Ctx) -> dict:
    score = 0.0
    factors = []
    if ctx.numeric >= 2:
        score += 0.7
        factors.append("2+ numeric → x/y scatter")
    if ctx.category >= 1:
        score += 0.1
        factors.append("colour by category")
    if ctx.row_count > 10:
        score += 0.1
        factors.append("sufficient points")
    return {"score": _clamp(score), "factors": factors}


def _score_scatter3d(ctx: _Ctx) -> dict:
    score = 0.0
    factors = []
    if ctx.numeric >= 3:
        score += 0.75
        factors.append("3 numeric → x/y/z scatter")
    if ctx.category >= 1:
        score += 0.1
        factors.append("colour by category")
    return {"score": _clamp(score), "factors": factors}


def _score_volcano(ctx: _Ctx) -> dict:
    score = 0.0
    factors = []
    if ctx.numeric >= 2:
        score += 0.4
        factors.append("2 numeric — possible volcano")
    # semantic boost applied in Layer 2
    return {"score": _clamp(score), "factors": factors}


def _score_kaplan_meier(ctx: _Ctx) -> dict:
    score = 0.0
    factors = []
    if ctx.numeric >= 1 and ctx.category >= 1:
        score += 0.4
        factors.append("numeric time + category group")
    # Boolean event/censoring indicator is a strong structural signal
    # but only when a grouping factor AND multiple numerics are present
    # (avoids false positives on binary flag columns like vs/am in mtcars)
    if ctx.boolean >= 1 and ctx.numeric >= 1 and ctx.category >= 1 and ctx.numeric <= 3:
        score += 0.4
        factors.append("binary event indicator + numeric time")
    # semantic boost applied in Layer 2
    return {"score": _clamp(score), "factors": factors}


def _score_correlation(ctx: _Ctx) -> dict:
    score = 0.0
    factors = []
    if ctx.numeric >= 2:
        score += 0.5
        factors.append("2+ numeric → pairwise corr")
    if ctx.numeric >= 4:
        score += 0.25
        factors.append("many numeric vars → matrix useful")
    return {"score": _clamp(score), "factors": factors}


def _score_sankey(ctx: _Ctx) -> dict:
    score = 0.0
    factors = []
    if ctx.category >= 2 and ctx.numeric >= 1:
        score += 0.6
        factors.append("2 cats as source/target + flow value")
    # semantic boost in Layer 2
    return {"score": _clamp(score), "factors": factors}


def _score_waterfall(ctx: _Ctx) -> dict:
    score = 0.0
    factors = []
    if ctx.numeric >= 1 and ctx.category >= 1:
        score += 0.45
        factors.append("numeric response + subject identifier")
    # semantic boost in Layer 2
    return {"score": _clamp(score), "factors": factors}


def _score_histogram(ctx: _Ctx) -> dict:
    score = 0.0
    factors = []
    if ctx.numeric >= 1:
        score += 0.55
        factors.append("1+ numeric → frequency distribution")
    if ctx.category == 0:
        score += 0.2
        factors.append("no category → single variable focus")
    if ctx.row_count >= 30:
        score += 0.15
        factors.append("enough data for bins")
    return {"score": _clamp(score), "factors": factors}


def _score_density(ctx: _Ctx) -> dict:
    score = 0.0
    factors = []
    if ctx.numeric >= 1:
        score += 0.5
        factors.append("1+ numeric → density estimate")
    if ctx.category >= 1:
        score += 0.15
        factors.append("overlay groups by category")
    if ctx.row_count >= 50:
        score += 0.2
        factors.append("enough n for smooth density")
    if ctx.cat0 and ctx.cat0 > 3:
        score -= 0.25
        factors.append("4+ color groups → ridgeline preferred over overlapping density")
    return {"score": _clamp(score), "factors": factors}


def _score_treemap(ctx: _Ctx) -> dict:
    score = 0.0
    factors = []
    if ctx.category >= 1 and ctx.numeric >= 1:
        score += 0.5
        factors.append("cat + numeric → hierarchical size")
    if ctx.category >= 2:
        score += 0.2
        factors.append("nested hierarchy")
    if ctx.cat0 and ctx.cat0 > 10:
        score += 0.15
        factors.append("many categories → treemap handles well")
    return {"score": _clamp(score), "factors": factors}


def _score_network(ctx: _Ctx) -> dict:
    score = 0.0
    factors = []
    if ctx.category >= 2:
        score += 0.3
        factors.append("2 cats — possible node/edge pairs")
    # semantic boost in Layer 2
    return {"score": _clamp(score), "factors": factors}


def _score_venn(ctx: _Ctx) -> dict:
    score = 0.0
    factors = []
    if ctx.category >= 2:
        score += 0.3
        factors.append("2+ cats — possible set membership")
    # semantic boost in Layer 2
    return {"score": _clamp(score), "factors": factors}


def _score_bar_line(ctx: _Ctx) -> dict:
    score = 0.0
    factors = []
    if ctx.datetime >= 1 and ctx.numeric >= 2:
        score += 0.75
        factors.append("time + 2 numeric → dual-axis bar-line")
        if ctx.category >= 1:
            score += 0.05
            factors.append("grouped series")
    elif ctx.category >= 1 and ctx.numeric >= 2:
        score += 0.6
        factors.append("1 cat + 2 numeric → dual axis bar-line")
    return {"score": _clamp(score), "factors": factors}


def _score_ridgeline(ctx: _Ctx) -> dict:
    score = 0.0
    factors = []
    if ctx.category >= 1 and ctx.numeric >= 1:
        score += 0.45
        factors.append("cat + numeric → stacked densities")
    if ctx.cat0 and 3 <= ctx.cat0 <= 15:
        score += 0.2
        factors.append("several groups to compare")
    if ctx.cat0 and ctx.cat0 > 3:
        score += 0.2
        factors.append("4+ groups → ridgeline avoids overlapping color curves")
    if ctx.row_count >= 100:
        score += 0.2
        factors.append("enough n per group for density")
    return {"score": _clamp(score), "factors": factors}


def _score_hexplot(ctx: _Ctx) -> dict:
    score = 0.0
    factors = []
    if ctx.numeric >= 2:
        score += 0.4
        factors.append("2 numeric → hex binning")
    if ctx.row_count >= 500:
        score += 0.4
        factors.append("large n — hexplot prevents overplotting")
    return {"score": _clamp(score), "factors": factors}


def _score_contour(ctx: _Ctx) -> dict:
    score = 0.0
    factors = []
    if ctx.numeric >= 2:
        score += 0.45
        factors.append("2 numeric → 2D density contour")
    if ctx.row_count > 200:
        score += 0.2
        factors.append("large n — contour informative")
    if ctx.category == 0 and ctx.numeric == 2:
        score += 0.15
        factors.append("pure bivariate — contour preferred over violin")
    return {"score": _clamp(score), "factors": factors}


def _score_sunburst(ctx: _Ctx) -> dict:
    score = 0.0
    factors = []
    if ctx.category >= 2 and ctx.numeric >= 1:
        score += 0.6
        factors.append("2+ cats + numeric → radial hierarchy")
    return {"score": _clamp(score), "factors": factors}


def _score_chord(ctx: _Ctx) -> dict:
    score = 0.0
    factors = []
    if ctx.category >= 2 and ctx.numeric >= 1:
        score += 0.5
        factors.append("2 cats + numeric → circular flow")
    # semantic boost in Layer 2
    return {"score": _clamp(score), "factors": factors}


def _score_parallel_coordinates(ctx: _Ctx) -> dict:
    score = 0.0
    factors = []
    if ctx.numeric >= 2:
        score += 0.5
        factors.append("2+ numeric → parallel axes")
    if ctx.category >= 1:
        score += 0.15
        factors.append("colour lines by category")
    if 2 <= ctx.numeric <= 3:
        score += 0.25
        factors.append("< 4 numeric cols — parallel coordinates preferred over heatmap")
    elif ctx.numeric >= 4:
        score -= 0.35
        factors.append("4+ numeric cols — heatmap preferred for dense multivariate")
    return {"score": _clamp(score), "factors": factors}


def _score_splom(ctx: _Ctx) -> dict:
    score = 0.0
    factors = []
    if 2 <= ctx.numeric <= 4:
        score += 0.7
        factors.append("2–4 numeric → compact pairwise matrix")
    elif ctx.numeric >= 5:
        score += 0.6
        factors.append("5+ numeric → pairwise scatter matrix")
    if ctx.numeric > 8:
        score -= 0.2
        factors.append("too many variables — matrix too busy")
    return {"score": _clamp(score), "factors": factors}


def _score_radar(ctx: _Ctx) -> dict:
    score = 0.0
    factors = []
    if ctx.numeric >= 3 and ctx.category >= 1:
        score += 0.55
        factors.append("3+ numeric axes + category groups")
    if ctx.category >= 1 and ctx.cat0 and ctx.cat0 <= 5:
        score += 0.15
        factors.append("few groups → readable radar")
    if ctx.numeric >= 5:
        score += 0.1
        factors.append("5+ axes — spider chart ideal")
    return {"score": _clamp(score), "factors": factors}


def _score_streamgraph(ctx: _Ctx) -> dict:
    score = 0.0
    factors = []
    if ctx.datetime >= 1 and ctx.category >= 1 and ctx.numeric >= 1:
        score += 0.7
        factors.append("time + cat + numeric → stacked stream")
    return {"score": _clamp(score), "factors": factors}


def _score_gantt(ctx: _Ctx) -> dict:
    score = 0.0
    factors = []
    if ctx.datetime >= 2 and ctx.category >= 1:
        score += 0.75
        factors.append("start/end dates + category")
    elif ctx.numeric >= 2 and ctx.category >= 1:
        score += 0.4
        factors.append("numeric start/end + category — possible Gantt")
    # semantic boost in Layer 2
    return {"score": _clamp(score), "factors": factors}


def _score_lollipop(ctx: _Ctx) -> dict:
    score = 0.0
    factors = []
    if ctx.category >= 1 and ctx.numeric >= 1:
        score += 0.45
        factors.append("cat + numeric → lollipop")
    if ctx.cat0 and ctx.cat0 > 15:
        score += 0.2
        factors.append("many categories — lollipop cleaner than bar")
    return {"score": _clamp(score), "factors": factors}


def _score_tag_cloud(ctx: _Ctx) -> dict:
    score = 0.0
    factors = []
    if ctx.category >= 1:
        score += 0.3
        factors.append("category labels → word cloud")
    if ctx.text >= 1:
        score += 0.4
        factors.append("text column → tag cloud")
    # semantic boost in Layer 2
    return {"score": _clamp(score), "factors": factors}


def _score_dot_line(ctx: _Ctx) -> dict:
    score = 0.0
    factors = []
    if ctx.category >= 1 and ctx.numeric >= 1:
        score += 0.4
        factors.append("cat + numeric — possible forest plot")
    if ctx.numeric >= 3:
        score += 0.3
        factors.append("estimate + CI bounds pattern")
    # semantic boost in Layer 2
    return {"score": _clamp(score), "factors": factors}


def _score_scatter_bubble_2d(ctx: _Ctx) -> dict:
    score = 0.0
    factors = []
    if ctx.numeric >= 3:
        score += 0.65
        factors.append("3 numeric → x, y, size")
    if ctx.category >= 1:
        score += 0.1
        factors.append("colour bubbles by category")
    return {"score": _clamp(score), "factors": factors}


def _score_alluvial(ctx: _Ctx) -> dict:
    score = 0.0
    factors = []
    if ctx.category >= 2 and ctx.numeric >= 1:
        score += 0.55
        factors.append("2+ cats + numeric → alluvial flow")
    if ctx.category >= 3 and ctx.numeric == 0:
        score += 0.6
        factors.append("3+ factor cols with no numeric → sequential state transitions")
    return {"score": _clamp(score), "factors": factors}


def _score_binplot(ctx: _Ctx) -> dict:
    score = 0.0
    factors = []
    if ctx.numeric >= 2:
        score += 0.4
        factors.append("2 numeric → binned scatter")
    if ctx.row_count > 200:
        score += 0.35
        factors.append("large n → binning reduces overplot")
    return {"score": _clamp(score), "factors": factors}


def _score_bubble(ctx: _Ctx) -> dict:
    return _score_scatter_bubble_2d(ctx)


def _score_bullet(ctx: _Ctx) -> dict:
    score = 0.0
    factors = []
    if ctx.category >= 1 and ctx.numeric >= 2:
        score += 0.55
        factors.append("cat + 2 numeric (value + target)")
    if ctx.numeric >= 3:
        score += 0.25
        factors.append("3+ numeric — actual/target/range bullet pattern")
    return {"score": _clamp(score), "factors": factors}


def _score_bump(ctx: _Ctx) -> dict:
    score = 0.0
    factors = []
    if ctx.datetime >= 1 and ctx.category >= 1 and ctx.numeric >= 1:
        score += 0.6
        factors.append("time + cat + numeric rank")
    return {"score": _clamp(score), "factors": factors}


def _score_cdf(ctx: _Ctx) -> dict:
    score = 0.0
    factors = []
    if ctx.numeric >= 1:
        score += 0.45
        factors.append("1+ numeric → CDF")
    if ctx.row_count >= 30:
        score += 0.2
        factors.append("enough data for smooth CDF")
    if ctx.category >= 1:
        score += 0.2
        factors.append("group overlay → empirical CDF per group")
    return {"score": _clamp(score), "factors": factors}


def _score_cleveland(ctx: _Ctx) -> dict:
    score = 0.0
    factors = []
    if ctx.category >= 1 and ctx.numeric >= 1:
        score += 0.45
        factors.append("cat + numeric → Cleveland dot")
    if ctx.numeric == 2 and ctx.category >= 1:
        score += 0.25
        factors.append("exactly 2 numerics per category → paired before/after dots")
    if ctx.cat0 and ctx.cat0 > 10:
        score += 0.2
        factors.append("many categories — Cleveland cleaner than bar")
    return {"score": _clamp(score), "factors": factors}


def _score_dumbbell(ctx: _Ctx) -> dict:
    score = 0.0
    factors = []
    if ctx.category >= 1 and ctx.numeric >= 2:
        score += 0.6
        factors.append("cat + 2 numeric → before/after")
    return {"score": _clamp(score), "factors": factors}


def _score_pareto(ctx: _Ctx) -> dict:
    score = 0.0
    factors = []
    if ctx.category >= 1 and ctx.numeric >= 1:
        score += 0.5
        factors.append("cat + numeric → ranked bar + cumulative %")
    return {"score": _clamp(score), "factors": factors}


def _score_qq(ctx: _Ctx) -> dict:
    score = 0.0
    factors = []
    if ctx.numeric >= 1:
        score += 0.4
        factors.append("1+ numeric → quantile comparison")
    if ctx.row_count >= 20:
        score += 0.2
        factors.append("sufficient n for quantiles")
    return {"score": _clamp(score), "factors": factors}


def _score_ribbon(ctx: _Ctx) -> dict:
    score = 0.0
    factors = []
    if ctx.datetime >= 1 and ctx.numeric >= 2:
        score += 0.65
        factors.append("time + 2 numeric → band range")
    if ctx.datetime == 0 and ctx.numeric >= 2 and ctx.category >= 1:
        score += 0.4
        factors.append("numeric start/end ranges per group")
    return {"score": _clamp(score), "factors": factors}


def _score_spaghetti(ctx: _Ctx) -> dict:
    score = 0.0
    factors = []
    if ctx.datetime >= 1 and ctx.numeric >= 1 and ctx.category >= 1:
        score += 0.65
        factors.append("time + numeric + subject id")
    if ctx.row_count < 500:
        score += 0.15
        factors.append("moderate n — trajectories readable")
    return {"score": _clamp(score), "factors": factors}


def _score_time_series(ctx: _Ctx) -> dict:
    score = 0.0
    factors = []
    if ctx.datetime >= 1 and ctx.numeric >= 1:
        score += 0.7
        factors.append("time + numeric → time series")
    if ctx.category >= 1:
        score += 0.1
        factors.append("multiple series by category")
    return {"score": _clamp(score), "factors": factors}


def _score_tornado(ctx: _Ctx) -> dict:
    score = 0.0
    factors = []
    if ctx.category >= 1 and ctx.numeric >= 2:
        score += 0.55
        factors.append("cat + 2 numeric → diverging bar")
    return {"score": _clamp(score), "factors": factors}


def _score_tree_bracket(ctx: _Ctx) -> dict:
    score = 0.0
    factors = []
    if ctx.category >= 1:
        score += 0.35
        factors.append("category labels → dendrogram leaves")
    # semantic boost in Layer 2
    return {"score": _clamp(score), "factors": factors}


def _score_upset(ctx: _Ctx) -> dict:
    score = 0.0
    factors = []
    if ctx.category >= 2:
        score += 0.4
        factors.append("2+ cats → set membership")
    if ctx.category >= 4:
        score += 0.25
        factors.append("4+ sets → better than Venn")
    if ctx.boolean >= 3 and ctx.category >= 1:
        score += 0.6
        factors.append("3+ binary/boolean columns + category → set membership matrix")
    return {"score": _clamp(score), "factors": factors}


def _score_map(ctx: _Ctx) -> dict:
    # Map has no distinctive structural signature — a category ID column + numeric
    # is equally valid for Bar, Line, Heatmap, etc.  Score is kept near-zero so
    # Map only surfaces when the user explicitly requests it via intent keywords
    # (Layer 3) or explicit geographic column names (Layer 2).
    return {"score": 0.0, "factors": []}


def _score_oncoprint(ctx: _Ctx) -> dict:
    # An oncoprint (gene x sample alteration matrix) has no distinctive structural
    # signature from column types alone -- it looks like many categorical/sample
    # columns, indistinguishable from a Heatmap or a wide factor table. Score is
    # kept near-zero so Oncoprint only surfaces when the user explicitly requests
    # it via intent keywords (Layer 3), mirroring Map.
    return {"score": 0.0, "factors": []}


# ---------------------------------------------------------------------------
# Scorers registry
# ---------------------------------------------------------------------------

_SCORERS: list[dict] = [
    {"name": "Bar",                 "graphType": "Bar",                 "fn": _score_bar},
    {"name": "Stacked",             "graphType": "Stacked",             "fn": _score_stacked},
    {"name": "StackedPercent",      "graphType": "StackedPercent",      "fn": _score_stacked_percent},
    {"name": "Boxplot",             "graphType": "Boxplot",             "fn": _score_boxplot},
    {"name": "Violin",              "graphType": "Violin",              "fn": _score_violin},
    {"name": "Dotplot",             "graphType": "Dotplot",             "fn": _score_dotplot},
    {"name": "Heatmap",             "graphType": "Heatmap",             "fn": _score_heatmap},
    {"name": "Line",                "graphType": "Line",                "fn": _score_line},
    {"name": "Area",                "graphType": "Area",                "fn": _score_area},
    {"name": "Scatter2D",           "graphType": "Scatter2D",           "fn": _score_scatter2d},
    {"name": "Scatter3D",           "graphType": "Scatter3D",           "fn": _score_scatter3d},
    {"name": "Volcano",             "graphType": "Volcano",             "fn": _score_volcano},
    {"name": "KaplanMeier",         "graphType": "KaplanMeier",         "fn": _score_kaplan_meier},
    {"name": "Correlation",         "graphType": "Correlation",         "fn": _score_correlation},
    {"name": "Sankey",              "graphType": "Sankey",              "fn": _score_sankey},
    {"name": "Waterfall",           "graphType": "Waterfall",           "fn": _score_waterfall},
    {"name": "Histogram",           "graphType": "Histogram",           "fn": _score_histogram},
    {"name": "Density",             "graphType": "Density",             "fn": _score_density},
    {"name": "Treemap",             "graphType": "Treemap",             "fn": _score_treemap},
    {"name": "Network",             "graphType": "Network",             "fn": _score_network},
    {"name": "Venn",                "graphType": "Venn",                "fn": _score_venn},
    {"name": "BarLine",             "graphType": "BarLine",             "fn": _score_bar_line},
    {"name": "Ridgeline",           "graphType": "Ridgeline",           "fn": _score_ridgeline},
    {"name": "Hexplot",             "graphType": "Hexplot",             "fn": _score_hexplot},
    {"name": "Contour",             "graphType": "Contour",             "fn": _score_contour},
    {"name": "Sunburst",            "graphType": "Sunburst",            "fn": _score_sunburst},
    {"name": "Chord",               "graphType": "Chord",               "fn": _score_chord},
    {"name": "ParallelCoordinates", "graphType": "ParallelCoordinates", "fn": _score_parallel_coordinates},
    {"name": "SPLOM",               "graphType": "SPLOM",               "fn": _score_splom},
    {"name": "Radar",               "graphType": "Radar",               "fn": _score_radar},
    {"name": "Streamgraph",         "graphType": "Streamgraph",         "fn": _score_streamgraph},
    {"name": "Gantt",               "graphType": "Gantt",               "fn": _score_gantt},
    {"name": "Lollipop",            "graphType": "Lollipop",            "fn": _score_lollipop},
    {"name": "TagCloud",            "graphType": "TagCloud",            "fn": _score_tag_cloud},
    {"name": "DotLine",             "graphType": "DotLine",             "fn": _score_dot_line},
    {"name": "ScatterBubble2D",     "graphType": "ScatterBubble2D",     "fn": _score_scatter_bubble_2d},
    {"name": "Alluvial",            "graphType": "Alluvial",            "fn": _score_alluvial},
    {"name": "Binplot",             "graphType": "Binplot",             "fn": _score_binplot},
    {"name": "Bubble",              "graphType": "Bubble",              "fn": _score_bubble},
    {"name": "Bullet",              "graphType": "Bullet",              "fn": _score_bullet},
    {"name": "Bump",                "graphType": "Bump",                "fn": _score_bump},
    {"name": "CDF",                 "graphType": "CDF",                 "fn": _score_cdf},
    {"name": "Cleveland",           "graphType": "Cleveland",           "fn": _score_cleveland},
    {"name": "Dumbbell",            "graphType": "Dumbbell",            "fn": _score_dumbbell},
    {"name": "Pareto",              "graphType": "Pareto",              "fn": _score_pareto},
    {"name": "QQ",                  "graphType": "QQ",                  "fn": _score_qq},
    {"name": "Ribbon",              "graphType": "Ribbon",              "fn": _score_ribbon},
    {"name": "Spaghetti",           "graphType": "Spaghetti",           "fn": _score_spaghetti},
    {"name": "TimeSeries",          "graphType": "TimeSeries",          "fn": _score_time_series},
    {"name": "Tornado",             "graphType": "Tornado",             "fn": _score_tornado},
    {"name": "TreeBracket",         "graphType": "TreeBracket",         "fn": _score_tree_bracket},
    {"name": "Upset",               "graphType": "Upset",               "fn": _score_upset},
    {"name": "Map",                 "graphType": "Map",                 "fn": _score_map},
    {"name": "Oncoprint",           "graphType": "Oncoprint",           "fn": _score_oncoprint},
]


# ---------------------------------------------------------------------------
# Layer 2: Semantic adjustments  (column name regex → score deltas)
# ---------------------------------------------------------------------------

def _col_stem_similarity(names: list[str]) -> float:
    """
    Return fraction of name pairs that share a common prefix or suffix of >= 3 chars.
    E.g. ['Sepal.Length','Sepal.Width','Petal.Length','Petal.Width'] → high similarity.
    """
    if len(names) < 2:
        return 0.0
    pairs = 0
    similar = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs += 1
            a, b = names[i].lower(), names[j].lower()
            # common prefix of length >= 3
            pfx = 0
            for ca, cb in zip(a, b):
                if ca == cb:
                    pfx += 1
                else:
                    break
            # common suffix of length >= 3
            sfx = 0
            for ca, cb in zip(reversed(a), reversed(b)):
                if ca == cb:
                    sfx += 1
                else:
                    break
            if pfx >= 3 or sfx >= 3:
                similar += 1
    return similar / pairs if pairs else 0.0


def _compute_semantic_adjustments(col_names: list[str], column_types: dict[str, str] | None = None) -> dict[str, float]:
    """
    Returns a dict of {graphType: delta} where delta is added to the
    Layer 1 structural score.  Ported directly from computeSemanticAdjustments
    in recommendCharts.es5.js.
    """
    adj: dict[str, float] = {}

    def add(gt: str, delta: float) -> None:
        adj[gt] = adj.get(gt, 0.0) + delta

    # Volcano: p-value + fold-change columns
    if _any_col_matches(col_names, "pvalue") and _any_col_matches(col_names, "foldchange"):
        add("Volcano", 0.9)

    # Kaplan-Meier: survival + event columns
    if _any_col_matches(col_names, "survival") and _any_col_matches(col_names, "event"):
        add("KaplanMeier", 0.9)
    if _any_col_matches(col_names, "survival") and _any_col_matches(col_names, "time"):
        add("KaplanMeier", 0.5)
    # Most common survival format: explicit time + event/censoring columns
    if _any_col_matches(col_names, "time") and _any_col_matches(col_names, "event"):
        add("KaplanMeier", 0.9)
    # time + group (e.g. Time/Group without explicit Event column)
    if _any_col_matches(col_names, "time") and _any_col_matches(col_names, "group"):
        add("KaplanMeier", 0.4)

    # Heatmap / Correlation: expression matrix or many numeric vars
    if _any_col_matches(col_names, "expression"):
        add("Heatmap", 0.5)
        add("Violin", 0.2)
        add("Boxplot", 0.2)
    if _count_col_matches(col_names, "correlation") >= 2:
        add("Correlation", 0.6)

    # Waterfall: response/RECIST columns
    if _any_col_matches(col_names, "response"):
        add("Waterfall", 0.5)
        add("Bar", 0.1)

    # Network: node + edge columns
    if _any_col_matches(col_names, "node") and _any_col_matches(col_names, "edge"):
        add("Network", 0.8)
    if _any_col_matches(col_names, "node"):
        add("Network", 0.3)

    # Sankey / Alluvial: source + target
    if _any_col_matches(col_names, "source") and _any_col_matches(col_names, "target"):
        add("Sankey", 0.7)
        add("Alluvial", 0.6)
        add("Chord", 0.4)

    # Venn / Upset: set membership columns
    if _count_col_matches(col_names, "set") >= 2:
        add("Venn", 0.6)
        add("Upset", 0.7)

    # Treemap / Sunburst: hierarchy columns
    if _any_col_matches(col_names, "hierarchy"):
        add("Treemap", 0.5)
        add("Sunburst", 0.5)

    # GWAS: chromosome + p-value → Manhattan (use Scatter2D + semantic label)
    if _any_col_matches(col_names, "chromosome") and _any_col_matches(col_names, "pvalue"):
        add("Scatter2D", 0.5)
        add("Volcano", 0.3)

    # PK: AUC/Cmax/Tmax → Line
    if _any_col_matches(col_names, "pk"):
        add("Line", 0.4)
        add("Scatter2D", 0.2)

    # Forest / DotLine: effect + CI columns
    if _any_col_matches(col_names, "effect") and _any_col_matches(col_names, "confidence"):
        add("DotLine", 0.75)

    # Spaghetti: subject + time
    if _any_col_matches(col_names, "subject") and _any_col_matches(col_names, "time"):
        add("Spaghetti", 0.6)
        add("Line", 0.2)

    # Gantt: start + end time columns
    if _any_col_matches(col_names, "start") and _any_col_matches(col_names, "end"):
        add("Gantt", 0.75)

    # Scatter: x + y coords
    if _any_col_matches(col_names, "x") and _any_col_matches(col_names, "y"):
        add("Scatter2D", 0.35)
    if (_any_col_matches(col_names, "x") and _any_col_matches(col_names, "y")
            and _any_col_matches(col_names, "z")):
        add("Scatter3D", 0.45)

    # PCA / UMAP / tSNE components
    if _count_col_matches(col_names, "component") >= 2:
        add("Scatter2D", 0.5)
        add("Scatter3D", 0.2)

    # Bubble: x + y + weight/size
    if (_any_col_matches(col_names, "x") and _any_col_matches(col_names, "y")
            and _any_col_matches(col_names, "weight")):
        add("ScatterBubble2D", 0.6)
        add("Bubble", 0.6)

    # TagCloud: label + frequency
    if _any_col_matches(col_names, "label") and _any_col_matches(col_names, "frequency2"):
        add("TagCloud", 0.75)

    # SD/SE/SEM column → error-bar estimation plot
    if _any_col_matches(col_names, "sd_sem"):
        add("DotLine", 0.5)
        add("Ribbon", 0.2)

    # Time-named factor column (when not yet detected as datetime) → boost time charts
    if column_types is not None:
        has_time_factor = any(
            v.lower() in _FACTOR_ALIASES and _col_matches(k, "time")
            for k, v in column_types.items()
        )
        has_numeric_col = any(v.lower() in _NUMERIC_ALIASES for v in column_types.values())
        if has_time_factor and has_numeric_col:
            add("Area",        0.25)
            add("Streamgraph", 0.20)
            add("TimeSeries",  0.20)
            add("BarLine",     0.20)
            add("Bump",        0.15)

    # Bump: rank + time
    if _any_col_matches(col_names, "rank") and _any_col_matches(col_names, "time"):
        add("Bump", 0.7)

    # Ribbon: lower + upper bounds over time
    if _any_col_matches(col_names, "lower") and _any_col_matches(col_names, "upper"):
        add("Ribbon", 0.65)
        add("DotLine", 0.3)
        add("Bullet", 0.4)

    # Dumbbell: two numeric measures per category
    if (_count_col_matches(col_names, "lower") >= 1
            and _count_col_matches(col_names, "upper") >= 1
            and _any_col_matches(col_names, "category")):
        add("Dumbbell", 0.5)

    # TreeBracket: hierarchy/cluster
    if _any_col_matches(col_names, "cluster"):
        add("TreeBracket", 0.5)
        add("Heatmap", 0.2)

    # Geographic columns → Map (only exact geographic ID column names, not generic ones)
    if _any_col_matches(col_names, "geographic"):
        add("Map", 0.3)  # small nudge — intent must confirm

    # Biomarker → Violin / Boxplot
    if _any_col_matches(col_names, "biomarker"):
        add("Violin", 0.3)
        add("Boxplot", 0.2)

    # Group / arm → Bar / Line (Boxplot already has strong structural score without a boost)
    if _any_col_matches(col_names, "group"):
        add("Bar", 0.15)
        add("Line", 0.1)

    # SPLOM: similar numeric column names (e.g. Sepal.Length/Sepal.Width/Petal.Length/Petal.Width)
    # indicates a family of related measurements — ideal for pairwise scatter matrix.
    # Exclude row-label columns (Id/subject) from the similarity check.
    if column_types is not None:
        num_cols = [
            c for c, t in column_types.items()
            if t.lower() in _NUMERIC_ALIASES
            and not (_col_matches(c, "subject") and not _col_matches(c, "group"))
        ]
        if 2 <= len(num_cols) <= 8:
            sim = _col_stem_similarity(num_cols)
            if sim >= 0.5:
                add("SPLOM", 0.6)
            elif sim >= 0.25:
                add("SPLOM", 0.3)

    return adj


# ---------------------------------------------------------------------------
# Layer 3: Intent keyword boosts
# ---------------------------------------------------------------------------

_INTENT_BOOSTS: list[dict] = [
    # clinical survival
    {"kws": ["survival", "kaplan", "km ", " os ", " pfs ", "time-to-event", "efs", "overall survival"],
     "boosts": {"KaplanMeier": 0.9}},
    # volcano / DEG
    {"kws": ["volcano", "fold change", "deg", "differential expression", "gwas"],
     "boosts": {"Volcano": 0.9}},
    # RECIST / waterfall
    {"kws": ["waterfall", "recist", "tumour shrinkage", "best change", "best percentage"],
     "boosts": {"Waterfall": 0.8}},
    # sankey / flow
    {"kws": ["sankey", "alluvial", "flow", "disposition", "patient journey"],
     "boosts": {"Sankey": 0.7, "Alluvial": 0.5}},
    # network
    {"kws": ["network", "pathway", "protein interaction"],
     "boosts": {"Network": 1.25}},
    # venn / overlap
    {"kws": ["venn", "overlap", "set intersection"],
     "boosts": {"Venn": 0.7, "Upset": 0.4}},
    # heatmap
    {"kws": ["heatmap", "heat map", "ae matrix", "gene expression matrix", "expression matrix"],
     "boosts": {"Heatmap": 0.75}},
    # PCA / UMAP / embedding
    {"kws": ["pca", "umap", "tsne", "t-sne", "embedding", "dimensionality reduction"],
     "boosts": {"Scatter2D": 0.6}},
    # correlation
    {"kws": ["correlation matrix", "pairwise correlation", "correlat"],
     "boosts": {"Correlation": 0.7}},
    # scatter
    {"kws": ["scatter", "regression", "bivariate", "x vs y"],
     "boosts": {"Scatter2D": 0.5}},
    # 3d
    {"kws": ["3d", "three dimensional"],
     "boosts": {"Scatter3D": 1.05}},
    # violin
    {"kws": ["violin"],
     "boosts": {"Violin": 0.7}},
    # gene expression → violin preferred (use specific phrases to avoid matching 'gene expression heatmap')
    {"kws": ["expression by cell", "expression distribution", "mrna", "rna seq", "rna-seq"],
     "boosts": {"Violin": 0.7, "Boxplot": 0.2}},
    # distribution
    {"kws": ["distribution", "spread", "variability", "outlier"],
     "boosts": {"Boxplot": 0.4, "Violin": 0.3, "Histogram": 0.2}},
    # histogram
    {"kws": ["histogram", "frequency distribution"],
     "boosts": {"Histogram": 0.65}},
    # density
    {"kws": ["density", "kde"],
     "boosts": {"Density": 0.65}},
    # smooth density / kernel density (more specific than plain 'density')
    {"kws": ["smooth density", "density curve", "kernel density", "smooth distribution"],
     "boosts": {"Density": 0.25}},
    # trend / longitudinal
    {"kws": ["trend", "longitudinal", "over time", "over visit", "pk profile", "concentration"],
     "boosts": {"Line": 0.65, "BarLine": 0.20, "Spaghetti": 0.2}},
    # 100% stacked
    {"kws": ["100%", "100 percent", "percent stack", "relative proportion"],
     "boosts": {"StackedPercent": 0.9}},
    # stacked
    {"kws": ["proportion", "part of whole", "composition", "breakdown", "percentage"],
     "boosts": {"Stacked": 0.5}},
    # explicit stacked bar / stacked bars
    {"kws": ["stacked bar", "stacked bars"],
     "boosts": {"Stacked": 0.9, "StackedPercent": 0.4}},
    # treemap
    {"kws": ["treemap", "tree map", "hierarchy", "hierarchical", "hierarchically"],
     "boosts": {"Treemap": 0.80, "Sunburst": 0.3}},
    # individual patient / small n
    {"kws": ["individual patient", "individual subject", "per subject", "small n", "spaghetti"],
     "boosts": {"Dotplot": 0.5, "Spaghetti": 0.4}},
    # forest plot / CI
    {"kws": ["forest plot", "hazard ratio", "odds ratio", "confidence interval", "hr ", " or "],
     "boosts": {"DotLine": 0.8}},
    # gantt / timeline
    {"kws": ["gantt", "timeline", "treatment duration", "exposure period"],
     "boosts": {"Gantt": 0.8}},
    # bubble
    {"kws": ["bubble"],
     "boosts": {"Bubble": 0.80, "ScatterBubble2D": 0.80}},
    # bubble size as 3rd variable → Bubble
    {"kws": ["bubble size", "size as third", "size as 3rd"],
     "boosts": {"Bubble": 0.90}},
    # sized-by pattern → ScatterBubble2D
    {"kws": ["sized by", "size by"],
     "boosts": {"ScatterBubble2D": 0.90}},
    # lollipop
    {"kws": ["lollipop"],
     "boosts": {"Lollipop": 0.7}},
    # pareto
    {"kws": ["pareto", "80 20", "cumulative percent"],
     "boosts": {"Pareto": 0.75}},
    # qq
    {"kws": ["qq plot", "quantile quantile", "normality"],
     "boosts": {"QQ": 0.7}},
    # tag cloud / word cloud
    {"kws": ["word cloud", "tag cloud", "text frequency"],
     "boosts": {"TagCloud": 0.8}},
    # bar
    {"kws": ["count", "incidence", "ae count", "compare", "by soc", "by arm", "magnitude", "bar chart"],
     "boosts": {"Bar": 0.5}},
    # dumbbell / before-after
    {"kws": ["before after", "change from baseline", "paired", "dumbbell"],
     "boosts": {"Dumbbell": 0.75}},
    # tornado / sensitivity
    {"kws": ["sensitivity analysis", "tornado", "butterfly"],
     "boosts": {"Tornado": 1.35}},
    # ridgeline
    {"kws": ["ridgeline", "ridge plot", "ridge chart", "distributions across", "stacked density", "stacked kde"],
     "boosts": {"Ridgeline": 0.80}},
    # time series (sensor / monitoring)
    {"kws": ["time series", "timeseries", "sensor", "real-time readings", "monitoring data", "signal readings"],
     "boosts": {"TimeSeries": 0.75}},
    # streamgraph
    {"kws": ["streamgraph", "stream graph", "stacked stream", "composition over time", "stacked flow"],
     "boosts": {"Streamgraph": 1.20}},
    # CDF
    {"kws": ["cdf", "cumulative distribution", "empirical cdf", "ecdf", "ks test", "kolmogorov"],
     "boosts": {"CDF": 0.8}},
    # Cleveland dot plot / before-after
    {"kws": ["cleveland", "dot plot", "before after", "before-after", "paired change", "change per item"],
     "boosts": {"Cleveland": 0.75}},
    # radar / spider
    {"kws": ["radar", "spider chart", "spider plot", "radial chart", "skill dimensions", "performance across", "across dimensions"],
     "boosts": {"Radar": 0.7}},
    # bullet chart
    {"kws": ["bullet chart", "bullet graph", "actual vs target", "performance target", "kpi target", "target vs actual"],
     "boosts": {"Bullet": 0.8}},
    # contour / 2D density
    {"kws": ["contour", "density contour", "2d density", "kde contour", "contour plot"],
     "boosts": {"Contour": 0.75}},
    # bar-line / dual axis
    {"kws": ["bar-line", "bar line", "barline", "dual axis", "dual-axis", "trend line overlay", "bars with line"],
     "boosts": {"BarLine": 0.75}},
    # bump chart / rank changes
    {"kws": ["bump chart", "bump plot", "rank change", "ranking change", "ranking over time", "rank over time", "rank trajectory"],
     "boosts": {"Bump": 0.8}},
    # alluvial / state transitions
    {"kws": ["alluvial", "state transition", "response transition", "flow between states", "state change", "patient transition"],
     "boosts": {"Alluvial": 0.75}},
    # upset / set intersections
    {"kws": ["upset", "upset plot", "set intersect", "set overlap", "multi-set"],
     "boosts": {"Upset": 0.8}},
    # gene sets — upset for many sets, venn for small (<=3)
    {"kws": ["gene sets", "gene set"],
     "boosts": {"Upset": 0.5}},
    # three-set venn
    {"kws": ["three sets", "three gene", "overlap between three", "3 sets", "three groups overlap"],
     "boosts": {"Venn": 0.8}},
    # ribbon / band range
    {"kws": ["ribbon", "ribbon plot", "band range", "ribbon chart", "range band", "range per group"],
     "boosts": {"Ribbon": 0.8}},
    # parallel coordinates
    {"kws": ["parallel coordinates", "parallel axes", "multivariate profile", "parallel plot"],
     "boosts": {"ParallelCoordinates": 0.7}},
    # area chart
    {"kws": ["area chart", "area graph", "stacked area", "shaded area", "fill under"],
     "boosts": {"Area": 0.7}},
    # pairwise scatter matrix → SPLOM
    {"kws": ["pairwise", "scatter matrix", "pair plot", "pairplot", "pairs plot", "splom"],
     "boosts": {"SPLOM": 0.9}},
    # individual data points / small cohort → Dotplot
    {"kws": ["individual data points", "individual points", "small cohort", "each observation"],
     "boosts": {"Dotplot": 0.7}},
    # mean with error bars → DotLine
    {"kws": ["mean response", "mean with sd", "mean and sd", "mean ± sd", "mean with error"],
     "boosts": {"DotLine": 0.8}},
    # two continuous variables / hex binning
    {"kws": ["two continuous", "bivariate density", "hex bin", "hexplot"],
     "boosts": {"Hexplot": 0.9, "Contour": 0.5}},
    # compare distributions by group → Boxplot
    {"kws": ["compare distribution", "distribution by group", "group distribution", "distributions between"],
     "boosts": {"Boxplot": 0.5}},
    # adverse event hierarchy → Sunburst
    {"kws": ["preferred term", "ae hierarchy", "soc, preferred", "preferred term, and"],
     "boosts": {"Sunburst": 0.7}},
    # tournament bracket → TreeBracket
    {"kws": ["tournament", "bracket", "elimination bracket", "playoff bracket", "tree bracket"],
     "boosts": {"TreeBracket": 1.0}},
    # binplot
    {"kws": ["binplot", "bin scatter", "binned scatter", "bin plot"],
     "boosts": {"Binplot": 1.15}},
    # map / choropleth / geographic — only fire on explicit map request phrases,
    # NOT on substrings of 'heatmap' or 'treemap'
    {"kws": ["choropleth", "geographic map", "geospatial map",
             "world map", "usa map", "us map", "us states map", "usa states map",
             "country map", "state map", "pie map", "marker map",
             "map of the", "map of usa", "map of us ", "map of world",
             "map of country", "map of state", "map of region",
             "map showing", "map with", "on a map", "on the map",
             "enrollment map", "prevalence map", "incidence map",
             "geographic distribution", "geospatial distribution"],
     "boosts": {"Map": 1.0}},
    # intent starts with 'map' (e.g. "map of gdp by country", "map the us states")
    {"kws": ["map of", "map the ", "map: "],
     "boosts": {"Map": 1.0}},
    # explicit geographic terms in intent
    {"kws": ["latitude", "longitude"],
     "boosts": {"Map": 0.8}},
    # oncoprint / gene alteration matrix — specialized genomics chart, surfaces
    # only on explicit intent (mirrors Map's near-zero base score)
    {"kws": ["oncoprint", "onco print", "alteration matrix", "gene alteration",
             "mutation matrix", "somatic mutation", "copy number alteration",
             "cbioportal", "cancer genomics", "mutation landscape"],
     "boosts": {"Oncoprint": 1.0}},
]


# ---------------------------------------------------------------------------
# Scoring engine
# ---------------------------------------------------------------------------

def recommend_charts(
    column_types: dict[str, str],
    n_samples: int = 100,
    category_cardinalities: Optional[dict[str, int]] = None,
    intent: str = "",
) -> list[dict]:
    """
    Score all 53 chart types and return them sorted by descending score.

    Args:
        column_types:            {col_name: type_string}
        n_samples:               Row count (used for structural scoring)
        category_cardinalities:  Optional {col_name: n_unique} for category cols
        intent:                  Plain-English intent (used for Layer 3 boosts)

    Returns:
        List of dicts sorted by score desc:
        [{"graphType": str, "score": float, "factors": [...], "layer2_delta": float}, ...]
    """
    n_fac, n_num, n_time, n_bool, n_text = _count_types(column_types)

    # Subtract pure row-label columns (id/subject/patient/...) from the factor
    # count — they are row labels, not grouping variables. Charts like Boxplot/
    # Violin/Dotplot require a true grouping factor (arm, treatment, cell type…).
    all_factor_cols = [k for k, v in column_types.items() if v.lower() in _FACTOR_ALIASES]
    n_label_only = sum(
        1 for col in all_factor_cols
        if _col_matches(col, "subject") and not _col_matches(col, "group")
    )
    n_grouping_fac = max(0, n_fac - n_label_only)

    # Build category cardinality stats (from grouping factors only, excluding row-label cols)
    cat_cards: list[int] = []
    if category_cardinalities:
        grouping_cols = [
            k for k, v in column_types.items()
            if v.lower() in _FACTOR_ALIASES
            and not (_col_matches(k, "subject") and not _col_matches(k, "group"))
        ]
        cat_cards = [category_cardinalities[c] for c in grouping_cols if c in category_cardinalities]
    min_cat_unique = min(cat_cards) if cat_cards else 0
    max_cat_unique = max(cat_cards) if cat_cards else 0
    cat0 = cat_cards[0] if cat_cards else 0

    ctx = _Ctx(
        numeric=n_num,
        category=n_grouping_fac,
        datetime=n_time,
        boolean=n_bool,
        text=n_text,
        row_count=n_samples,
        min_cat_unique=min_cat_unique,
        max_cat_unique=max_cat_unique,
        cat0=cat0,
        has_high_variance=False,  # unknown without data; semantic layer can adjust
    )

    col_names = list(column_types.keys())
    semantic_adj = _compute_semantic_adjustments(col_names, column_types)

    # Layer 3: intent boosts
    intent_lower = intent.lower()
    intent_adj: dict[str, float] = {}
    for rule in _INTENT_BOOSTS:
        if any(kw in intent_lower for kw in rule["kws"]):
            for gt, delta in rule["boosts"].items():
                intent_adj[gt] = intent_adj.get(gt, 0.0) + delta

    results = []
    for scorer in _SCORERS:
        gt = scorer["graphType"]
        layer1 = scorer["fn"](ctx)
        l2 = semantic_adj.get(gt, 0.0)
        l3 = intent_adj.get(gt, 0.0)
        raw_score = layer1["score"] + l2 + l3
        final_score = _clamp(raw_score)
        results.append({
            "graphType":     gt,
            "score":         round(final_score, 4),
            "_raw":          round(raw_score, 4),
            "factors":       layer1["factors"],
            "layer2_delta":  round(l2, 4),
            "layer3_delta":  round(l3, 4),
        })

    results.sort(key=lambda r: r["_raw"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# LLM tiebreaker (optional)
# ---------------------------------------------------------------------------

# Call the LLM when the margin between #1 and #2 is smaller than this.
_LLM_TIEBREAK_THRESHOLD = 0.15


def _llm_tiebreak(
    candidates: list[dict],
    column_types: dict[str, str],
    n_samples: int,
    llm_complete: Callable[[str, str], str],
) -> tuple[str | None, str]:
    """
    Ask the LLM to choose among a short-list of ambiguous chart candidates.

    Args:
        candidates:   Top-N enriched entries each with at least
                      {graphType, score, description}.
        column_types: {col_name: type_string} for every column.
        n_samples:    Row count of the dataset.
        llm_complete: Callable(system, user) -> str.  The caller typically
                      passes llm_providers.complete wrapped to drop the
                      usage dict, or any function with that signature.

    Returns:
        (graphType, reason)  — graphType is one of the candidate names,
                               reason is the LLM's one-sentence explanation.
        (None, error_msg)    — on any failure (bad response, exception, …).
    """
    names_lines = "\n".join(
        f"  - {col} ({typ})" for col, typ in column_types.items()
    )
    cand_lines = "\n".join(
        f"  {i + 1}. {c['graphType']} (score={c['score']:.2f})"
        + (f": {c.get('description', '')}" if c.get("description") else "")
        for i, c in enumerate(candidates)
    )
    valid_names = [c["graphType"] for c in candidates]
    valid_list  = ", ".join(valid_names)

    system = (
        "You are a data visualization expert specializing in CanvasXpress charts. "
        "When asked to choose a chart type, reply with EXACTLY one chart name from "
        "the provided list, followed by a pipe | and a brief one-sentence reason. "
        "Example: Scatter2D | Best for showing correlation between two numeric variables."
    )
    user = (
        f"Dataset has {n_samples} rows and the following columns:\n{names_lines}\n\n"
        f"Candidate chart types (already ranked by structural scoring):\n{cand_lines}\n\n"
        f"Which is the most appropriate chart type? "
        f"Reply with exactly one of: {valid_list}"
    )

    try:
        raw = llm_complete(system, user).strip()
        if not raw:
            return None, "LLM returned empty response"
        if "|" in raw:
            chosen, reason = raw.split("|", 1)
            chosen = chosen.strip()
            reason = reason.strip()
        else:
            parts = raw.split()
            if not parts:
                return None, "LLM returned empty response"
            chosen = parts[0].rstrip(".,:;")
            reason = ""
        # exact match first
        if chosen in valid_names:
            return chosen, reason
        # case-insensitive fallback
        for name in valid_names:
            if name.lower() == chosen.lower():
                return name, reason
        return None, f"LLM returned unrecognised chart type: {chosen!r}"
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def select_chart(
    intent: str,
    column_types: dict[str, str],
    n_samples: Optional[int] = None,
    category_cardinalities: Optional[dict[str, int]] = None,
    llm_complete: Optional[Callable[[str, str], str]] = None,
) -> dict:
    """
    Recommend CanvasXpress graphType(s) from column types + analytical intent.

    Args:
        intent:       Plain-English description of what you want to show.
                      e.g. "AE counts by SOC across 3 treatment arms"
        column_types: {column_name: type} where type is one of:
                      numeric | factor | string | date | integer | binary |
                      boolean | text
        n_samples:    Optional row count — nudges Dotplot over Boxplot for small n
                      and Hexplot/Binplot for large n.
        category_cardinalities:
                      Optional {col_name: n_unique} for categorical columns.

    Returns:
        {
          intent             (str)  - echoed back
          column_summary     (dict) - {n_factor, n_numeric, n_time, n_bool, n_text}
          top_recommendation (dict) - best graphType with rationale
          alternatives       (list) - up to 4 other candidates
          generate_hint      (str)  - suggested description to pass to
                                      generate_canvasxpress_config
          tiebreak           (dict) - present only when LLM tiebreaker ran;
                                      {used: bool, chosen: str, reason: str}
        }
    """
    rows = n_samples if n_samples is not None else 100
    n_fac, n_num, n_time, n_bool, n_text = _count_types(column_types)

    ranked = recommend_charts(
        column_types=column_types,
        n_samples=rows,
        category_cardinalities=category_cardinalities,
        intent=intent,
    )

    # -----------------------------------------------------------------------
    # Optional LLM tiebreaker
    # -----------------------------------------------------------------------
    tiebreak_info: dict = {}
    if (
        llm_complete is not None
        and len(ranked) >= 2
        and (ranked[0].get("_raw", ranked[0]["score"]) - ranked[1].get("_raw", ranked[1]["score"])) < _LLM_TIEBREAK_THRESHOLD
    ):
        # Pass the top-3 candidates (or fewer if the list is short)
        tb_candidates = [
            {
                "graphType":   r["graphType"],
                "score":       r["score"],
                "description": CHART_CATALOGUE.get(r["graphType"], {}).get("description", ""),
            }
            for r in ranked[:3]
        ]
        chosen, reason = _llm_tiebreak(
            candidates=tb_candidates,
            column_types=column_types,
            n_samples=rows,
            llm_complete=llm_complete,
        )
        tiebreak_info = {"used": chosen is not None, "chosen": chosen, "reason": reason}
        if chosen is not None and chosen != ranked[0]["graphType"]:
            # Promote the LLM-chosen chart to the front of the list
            chosen_idx = next(
                (i for i, r in enumerate(ranked) if r["graphType"] == chosen), None
            )
            if chosen_idx is not None:
                ranked.insert(0, ranked.pop(chosen_idx))

    def _enrich(entry: dict) -> dict:
        gt = entry["graphType"]
        info = CHART_CATALOGUE.get(gt, {})
        out: dict = {
            "graphType":    gt,
            "score":        entry["score"],
            "category":     info.get("category", ""),
            "description":  info.get("description", ""),
            "clinical_use": info.get("clinical_use", ""),
            "next_step":    info.get(
                "next_step",
                f"generate_canvasxpress_config with description='{gt} chart'",
            ),
            "scoring_factors": entry["factors"],
        }
        if n_samples is not None and n_samples < 30 and gt == "Dotplot":
            out["note"] = (
                f"n_samples={n_samples} is small — Dotplot shows every observation clearly."
            )
        if n_samples is not None and n_samples > 5000 and gt in {"Scatter2D", "Dotplot"}:
            out["note"] = (
                f"n_samples={n_samples} is large — consider Hexplot or Binplot "
                "to avoid overplotting."
            )
        return out

    top  = _enrich(ranked[0])
    alts = [_enrich(r) for r in ranked[1:5]]

    # Build a ready-made description hint for generate_canvasxpress_config
    col_names    = list(column_types.keys())
    factor_cols  = [k for k, v in column_types.items()
                    if v.lower() in _FACTOR_ALIASES
                    and not (_col_matches(k, "subject") and not _col_matches(k, "group"))]
    # Exclude subject/ID columns (e.g. "Id", "PatientId") from numeric candidates
    # so they don't end up as chart axes.
    numeric_cols = [
        k for k, v in column_types.items()
        if v.lower() in _NUMERIC_ALIASES
        and not (_col_matches(k, "subject") and not _col_matches(k, "group"))
    ]
    time_cols    = [k for k, v in column_types.items() if v.lower() in _TIME_ALIASES]

    best_gt = ranked[0]["graphType"]
    if factor_cols and numeric_cols:
        axis_part = (
            f"over {time_cols[0]}" if time_cols else f"grouped by {factor_cols[0]}"
        )
        color_part = (
            f" colored by {factor_cols[1]}" if len(factor_cols) > 1 else ""
        )
        generate_hint = (
            f"{best_gt} chart of {numeric_cols[0]} {axis_part}"
            f"{color_part} — columns: {', '.join(col_names)}"
        )
    else:
        generate_hint = f"{best_gt} chart — columns: {', '.join(col_names)}"

    result: dict = {
        "intent":         intent,
        "column_summary": {
            "n_factor":  n_fac,
            "n_numeric": n_num,
            "n_time":    n_time,
            "n_bool":    n_bool,
            "n_text":    n_text,
        },
        "top_recommendation": top,
        "alternatives":       alts,
        "generate_hint":      generate_hint,
    }
    if tiebreak_info:
        result["tiebreak"] = tiebreak_info
    return result

