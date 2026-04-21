#!/usr/bin/env python3
"""
cx_knowledge.py
===============
CanvasXpress parameter knowledge skill for the MCP server.

Fetches SCHEMA.md and RULES.md from neuhausi/canvasxpress-LLM on GitHub,
parses out every parameter with its valid values, description, and applicable
graph types, then exposes that knowledge in three ways:

  1. MCP tool — query_canvasxpress_params(graph_type, param_name)
     Returns all supported parameters for a graph type, or the full
     definition of a single parameter including valid values.

  2. Prompt injection — get_param_snippet(graph_type)
     Returns a concise string of graph-type-specific parameters + valid
     values to append to the system prompt, tightening generation quality.

  3. Value validation — validate_param_values(config)
     Checks every string-valued config parameter against its known valid
     values and returns warnings for anything that looks wrong.

Fetch behaviour:
  - Primary  : GitHub raw URL (GITHUB_BASE/SCHEMA.md, RULES.md, CONTEXT.md)
  - Fallback : locally cached files in data/kb_cache/
  - Both fail: returns bundled minimal hardcoded schema (always works offline)

Configuration (env vars):
  CX_SCHEMA_TTL   cache TTL in seconds, default 3600 (1 hour)
  CX_SKIP_FETCH   set to "1" to always use cache / bundled schema
"""

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

log = logging.getLogger("cx-mcp.knowledge")

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

_HERE      = Path(__file__).parent
_BASE_DIR  = _HERE.parent
_CACHE_DIR = _BASE_DIR / "data" / "kb_cache"

GITHUB_BASE = "https://raw.githubusercontent.com/neuhausi/canvasxpress-LLM/main"
SCHEMA_TTL  = int(os.environ.get("CX_SCHEMA_TTL", "3600"))
SKIP_FETCH  = os.environ.get("CX_SKIP_FETCH", "").lower() in ("1", "true", "yes")

# Files we want from GitHub
_FETCH_FILES = ["SCHEMA.md", "RULES.md", "CONTEXT.md"]

# ---------------------------------------------------------------------------
# Bundled minimal schema (offline fallback)
# Keys: parameter name → {description, valid_values, graph_types, type}
# graph_types: list of applicable graph type names, or ["all"] for universal
# ---------------------------------------------------------------------------

_BUNDLED_SCHEMA: dict[str, dict] = {
    # ── Graph type (structural) ───────────────────────────────────────────────
    "graphType": {
        "description": "The type of chart to render.",
        "type": "string",
        "graph_types": ["all"],
        "valid_values": [
            "Alluvial","Area","AreaLine","Bar","BarLine","Boxplot","Bin","Binplot",
            "Bubble","Bullet","Bump","CDF","Chord","Circular","Cleveland","Contour",
            "Correlation","Density","Distribution","Donut","DotLine","Dotplot",
            "Dumbbell","Gantt","Heatmap","Hex","Hexplot","Histogram","KaplanMeier",
            "Line","Lollipop","Map","Meter","Network","ParallelCoordinates","Pareto",
            "Pie","QQ","Quantile","Radar","Ribbon","Ridgeline","Sankey","Scatter2D",
            "Scatter3D","ScatterBubble2D","Spaghetti","Stacked","StackedLine",
            "StackedPercent","StackedPercentLine","Streamgraph","Sunburst","TagCloud",
            "TimeSeries","Tornado","Tree","Treemap","Upset","Violin","Volcano",
            "Venn","Waterfall","WordCloud",
        ],
    },
    # ── Section 1 · Data Manipulation ────────────────────────────────────────
    "sortData":     {"description":"Sort rules applied before rendering.",             "type":"array", "graph_types":["all"],"valid_values":[]},
    "filterData":   {"description":"Filter rules applied before rendering.",           "type":"array", "graph_types":["all"],"valid_values":[]},
    "groupingFactors":  {"description":"Factor columns used to group and colour data.","type":"array",  "graph_types":["all"],"valid_values":[]},
    "hierarchy":        {"description":"Ordered list of hierarchy columns for Bubble/Tree/Sunburst.","type":"array","graph_types":["Bubble","Tree","Sunburst"],"valid_values":[]},
    "sankeyAxes":       {"description":"Ordered list of flow-axis columns for Sankey/Alluvial/Ribbon.","type":"array","graph_types":["Sankey","Alluvial","Ribbon"],"valid_values":[]},
    "labelBy":              {"description":"Column name used to label individual data points.",              "type":"string", "graph_types":["all"],"valid_values":[]},
    "labelSelect":          {"description":"Selection rules that determine which data points are labeled.",  "type":"array",  "graph_types":["all"],"valid_values":[]},
    "optimizeTextPosition": {"description":"Optimize label positions to avoid overlap between text items.", "type":"boolean","graph_types":["all"],"valid_values":[]},
    "xAxisTransform": {
        "description": "Mathematical transform applied to the x-axis scale.",
        "type": "string",
        "graph_types": ["all"],
        "valid_values": ["log2","log10","-log2","-log10","sqrt","percentile"],
    },
    "yAxisTransform": {
        "description": "Mathematical transform applied to the y-axis scale.",
        "type": "string",
        "graph_types": ["Scatter2D","Scatter3D","ScatterBubble2D","Spaghetti","Volcano","KaplanMeier"],
        "valid_values": ["log2","log10","-log2","-log10","sqrt","percentile"],
    },
    "transformData": {
        "description": "Transform applied to the underlying data values before rendering.",
        "type": "string",
        "graph_types": ["all"],
        "valid_values": ["log2","log10","-log2","-log10","zscore","percentile","sqrt"],
    },
    "setMaxX":        {"description":"Maximum value of the x-axis scale.",             "type":"numeric","graph_types":["all"],"valid_values":[]},
    "setMinX":        {"description":"Minimum value of the x-axis scale.",             "type":"numeric","graph_types":["all"],"valid_values":[]},
    "setMaxY":        {"description":"Maximum value of the y-axis scale (only when yAxis present).","type":"numeric","graph_types":["Scatter2D","Scatter3D","ScatterBubble2D","Spaghetti","Volcano","KaplanMeier"],"valid_values":[]},
    "setMinY":        {"description":"Minimum value of the y-axis scale (only when yAxis present).","type":"numeric","graph_types":["Scatter2D","Scatter3D","ScatterBubble2D","Spaghetti","Volcano","KaplanMeier"],"valid_values":[]},
    "setMax": {"description":"Maximum value of the data scale for 1D charts (bar, boxplot, etc.).", "type":"numeric","graph_types":["Bar","Boxplot","Violin","Histogram","Density","Dotplot","Ridgeline"],"valid_values":[]},
    "setMin": {"description":"Minimum value of the data scale for 1D charts (bar, boxplot, etc.).", "type":"numeric","graph_types":["Bar","Boxplot","Violin","Histogram","Density","Dotplot","Ridgeline"],"valid_values":[]},
    # ── Section 2 · Global Canvas / Themes ───────────────────────────────────
    "graphOrientation": {
        "description": "Orientation of bar-type charts.",
        "type": "string",
        "graph_types": ["Bar","Stacked","StackedPercent","Lollipop","Cleveland","Dumbbell","Waterfall"],
        "valid_values": ["horizontal","vertical"],
    },
    "theme": {
        "description": "Visual theme controlling overall chart aesthetics.",
        "type": "string",
        "graph_types": ["all"],
        "valid_values": [
            "bw","classic","cx","dark","economist","excel","ggblanket","ggplot",
            "gray","grey","highcharts","igray","light","linedraw","minimal","none",
            "ptol","solarized","stata","tableau","void0","wsj",
        ],
    },
    "colorScheme": {
        "description": "Named color palette applied to the chart.",
        "type": "string",
        "graph_types": ["all"],
        "valid_values": [
            "YlGn","YlGnBu","GnBu","BuGn","PuBuGn","PuBu","BuPu","RdPu","PuRd",
            "OrRd","YlOrRd","YlOrBr","Purples","Blues","Greens","Oranges","Reds",
            "Greys","PuOr","BrBG","PRGn","PiYG","RdBu","RdGy","RdYlBu","Spectral",
            "RdYlGn","Bootstrap","Economist","Excel","GGPlot","Solarized","PaulTol",
            "ColorBlind","Tableau","WallStreetJournal","Stata","BlackAndWhite","CanvasXpress",
        ],
    },
    # ── Section 3 · Titles / Subtitles / Citations ────────────────────────────
    "title":          {"description":"Chart title displayed above the plot.",          "type":"string", "graph_types":["all"],"valid_values":[]},
    "titleScaleFontFactor": {"description":"Scale factor for the chart title font size.",           "type":"numeric","graph_types":["all"],"valid_values":[]},
    "titleColor":           {"description":"Color of the chart title text.",                        "type":"string", "graph_types":["all"],"valid_values":[]},
    "titleFontStyle":       {"description":"Font style of the chart title.",                        "type":"string", "graph_types":["all"],"valid_values":["bold","italic","bold italic"]},
    "subtitle":                   {"description":"Chart subtitle displayed below the main title.",  "type":"string", "graph_types":["all"],"valid_values":[]},
    "subtitleScaleFontFactor":    {"description":"Scale factor for the subtitle font size.",        "type":"numeric","graph_types":["all"],"valid_values":[]},
    "subtitleColor":              {"description":"Color of the subtitle text.",                     "type":"string", "graph_types":["all"],"valid_values":[]},
    "subtitleFontStyle":          {"description":"Font style of the subtitle.",                     "type":"string", "graph_types":["all"],"valid_values":["bold","italic","bold italic"]},
    "citation":                   {"description":"Citation or attribution text shown at the bottom of the chart.", "type":"string", "graph_types":["all"],"valid_values":[]},
    "citationScaleFontFactor":    {"description":"Scale factor for the citation font size.",        "type":"numeric","graph_types":["all"],"valid_values":[]},
    "citationColor":              {"description":"Color of the citation text.",                     "type":"string", "graph_types":["all"],"valid_values":[]},
    "citationFontStyle":          {"description":"Font style of the citation.",                     "type":"string", "graph_types":["all"],"valid_values":["bold","italic","bold italic"]},
    # ── Section 4 · Axes ──────────────────────────────────────────────────────
    # X axis
    "xAxisTitle":     {"description":"Label for the x-axis.",                          "type":"string", "graph_types":["all"],"valid_values":[]},
    "xAxisTitleScaleFontFactor":  {"description":"Scale factor for the x-axis title font size.",   "type":"numeric","graph_types":["all"],"valid_values":[]},
    "xAxisTitleColor":            {"description":"Color of the x-axis title text.",                "type":"string", "graph_types":["all"],"valid_values":[]},
    "xAxisTextScaleFontFactor":   {"description":"Scale factor for x-axis tick label font size.",  "type":"numeric","graph_types":["all"],"valid_values":[]},
    "xAxisTextColor":             {"description":"Color of the x-axis tick label text.",           "type":"string", "graph_types":["all"],"valid_values":[]},
    # Y axis
    "yAxisTitle":     {"description":"Label for the y-axis (multi-dimensional charts).","type":"string", "graph_types":["Scatter2D","Scatter3D","ScatterBubble2D","Spaghetti","Volcano","KaplanMeier","Contour","Streamgraph","Bump"],"valid_values":[]},
    "yAxisTitleScaleFontFactor":  {"description":"Scale factor for the y-axis title font size.",   "type":"numeric","graph_types":["Scatter2D","Scatter3D","ScatterBubble2D","Spaghetti","Volcano","KaplanMeier","Contour","Streamgraph","Bump"],"valid_values":[]},
    "yAxisTitleColor":            {"description":"Color of the y-axis title text.",                "type":"string", "graph_types":["Scatter2D","Scatter3D","ScatterBubble2D","Spaghetti","Volcano","KaplanMeier","Contour","Streamgraph","Bump"],"valid_values":[]},
    "yAxisTextScaleFontFactor":   {"description":"Scale factor for y-axis tick label font size.",  "type":"numeric","graph_types":["Scatter2D","Scatter3D","ScatterBubble2D","Spaghetti","Volcano","KaplanMeier","Contour","Streamgraph","Bump"],"valid_values":[]},
    "yAxisTextColor":             {"description":"Color of the y-axis tick label text.",           "type":"string", "graph_types":["Scatter2D","Scatter3D","ScatterBubble2D","Spaghetti","Volcano","KaplanMeier","Contour","Streamgraph","Bump"],"valid_values":[]},
    # Z axis
    "zAxisTitle":     {"description":"Label for the z-axis (3D charts).",              "type":"string", "graph_types":["Scatter3D","ScatterBubble2D"],"valid_values":[]},
    # Sample axis
    "smpTitle":              {"description":"Sample axis label for single-dimensional and combined charts. Use instead of yAxisTitle for 1D/combined types.","type":"string","graph_types":["all"],"valid_values":[]},
    "smpTitleScaleFontFactor":{"description":"Scale factor for smpTitle font size.","type":"numeric","graph_types":["all"],"valid_values":[]},
    "smpTitleColor":         {"description":"Colour of the smpTitle text.","type":"string","graph_types":["all"],"valid_values":[]},
    "smpTitleFontStyle":     {"description":"Font style of the smpTitle text.","type":"string","graph_types":["all"],"valid_values":["normal","bold","italic","bold italic"]},
    "smpTextScaleFontFactor":{"description":"Scale factor for sample axis tick label font size.","type":"numeric","graph_types":["all"],"valid_values":[]},
    "smpTextColor":          {"description":"Colour of sample axis tick labels (1D/combined charts only).","type":"string","graph_types":["all"],"valid_values":[]},
    "smpTextFontStyle":      {"description":"Font style of sample axis tick labels.","type":"string","graph_types":["all"],"valid_values":["normal","bold","italic","bold italic"]},
    "smpTextRotate":         {"description":"Rotation angle for sample labels.","type":"numeric","graph_types":["Area","Bar","Boxplot","Dotplot","Line","Stacked","StackedPercent","Violin"],"valid_values":[]},
    # Variable axis
    "varTitle":              {"description":"Variable axis label for single-dimensional and combined charts. Use instead of xAxisTitle for 1D/combined types.","type":"string","graph_types":["all"],"valid_values":[]},
    "varTitleScaleFontFactor":{"description":"Scale factor for varTitle font size.","type":"numeric","graph_types":["all"],"valid_values":[]},
    "varTitleColor":         {"description":"Colour of the varTitle text.","type":"string","graph_types":["all"],"valid_values":[]},
    "varTitleFontStyle":     {"description":"Font style of the varTitle text.","type":"string","graph_types":["all"],"valid_values":["normal","bold","italic","bold italic"]},
    "varTextScaleFontFactor":{"description":"Scale factor for variable axis tick label font size.","type":"numeric","graph_types":["all"],"valid_values":[]},
    "varTextColor":          {"description":"Colour of variable axis tick labels (1D/combined charts only).","type":"string","graph_types":["all"],"valid_values":[]},
    "varTextFontStyle":      {"description":"Font style of variable axis tick labels.","type":"string","graph_types":["all"],"valid_values":["normal","bold","italic","bold italic"]},
    "varTextRotate":         {"description":"Rotation angle for variable labels.","type":"numeric","graph_types":["Heatmap","Correlation"],"valid_values":[]},
    # Sankey axis
    "sankeyTitleScaleFontFactor": {"description":"Scale factor for Sankey title font size.",       "type":"numeric","graph_types":["Sankey","Alluvial","Ribbon"],"valid_values":[]},
    "sankeyTitleColor":           {"description":"Color of the Sankey title text.",                "type":"string", "graph_types":["Sankey","Alluvial","Ribbon"],"valid_values":[]},
    "sankeyTextScaleFontFactor":  {"description":"Scale factor for Sankey label font size.",       "type":"numeric","graph_types":["Sankey","Alluvial","Ribbon"],"valid_values":[]},
    "sankeyTextColor":            {"description":"Color of the Sankey label text.",                "type":"string", "graph_types":["Sankey","Alluvial","Ribbon"],"valid_values":[]},
    # ── Section 5 · Aesthetics ────────────────────────────────────────────────
    "colorBy":          {"description":"Column whose values determine point/series colour.","type":"string","graph_types":["all"],"valid_values":[]},
    "shapeBy":          {"description":"Column whose values determine point shape.",   "type":"string", "graph_types":["Scatter2D","ScatterBubble2D","Spaghetti"],"valid_values":[]},
    "sizeBy":           {"description":"Column whose values scale point size.",        "type":"string", "graph_types":["ScatterBubble2D"],"valid_values":[]},
    "lineBy":       {"description":"Column defining line/edge grouping for Bump/Spaghetti.","type":"string","graph_types":["Bump","Spaghetti"],"valid_values":[]},
    "ridgeBy":          {"description":"Column whose values define ridgeline groups.",  "type":"string","graph_types":["Ridgeline"],"valid_values":[]},
    "ellipseBy":                {"description":"Column name to draw confidence ellipses around groups of data points.","type":"string","graph_types":["Scatter2D","ScatterBubble2D"],"valid_values":[]},
    "dataPointSizeScaleFactor":   {"description":"Uniform scale factor applied to all data point sizes.",     "type":"numeric","graph_types":["all"],"valid_values":[]},
    "dataPointSize":  {"description":"Radius of individual data points in pixels.",    "type":"numeric","graph_types":["Scatter2D","Scatter3D","ScatterBubble2D","Spaghetti","Contour"],"valid_values":[]},
    "showBoxplotOriginalData":  {"description":"Overlay original data points on boxplot.", "type":"boolean","graph_types":["Boxplot"],"valid_values":[]},
    "jitter":                   {"description":"Jitter overlapping data points.",          "type":"boolean","graph_types":["Boxplot","Violin","Dotplot"],"valid_values":[]},
    "sina":                       {"description":"Use Sina plot (density-aware) jitter for overlapping points.","type":"boolean","graph_types":["Boxplot","Violin","Dotplot"],"valid_values":[]},
    "binned":                     {"description":"Bin overlapping data points into stacked columns.",          "type":"boolean","graph_types":["Boxplot","Violin","Dotplot"],"valid_values":[]},
    "binAlignment":               {"description":"Horizontal alignment of data points within each bin.",       "type":"string", "graph_types":["Boxplot","Violin","Dotplot"],"valid_values":["left","center","right"]},
    "guidesShow":     {"description":"Show horizontal guide lines across the chart area.",    "type":"boolean","graph_types":["Bar","Boxplot","Violin","Histogram","Density","Dotplot","Line","Area"],"valid_values":[]},
    "guidesColor":    {"description":"Color of the guide lines.",                             "type":"string", "graph_types":["Bar","Boxplot","Violin","Histogram","Density","Dotplot","Line","Area"],"valid_values":[]},
    "guidesWidth":    {"description":"Width of the guide lines in pixels.",                   "type":"numeric","graph_types":["Bar","Boxplot","Violin","Histogram","Density","Dotplot","Line","Area"],"valid_values":[]},
    "guidesLineType": {"description":"Stroke style of the guide lines.",                      "type":"string", "graph_types":["Bar","Boxplot","Violin","Histogram","Density","Dotplot","Line","Area"],"valid_values":["solid","dotted","dashed","dotdash","longdash","twodash"]},
    "objectBorderColor": {"description":"Border/stroke color of chart objects such as bars and points.", "type":"string","graph_types":["Bar","Boxplot","Violin","Histogram","Density","Dotplot"],"valid_values":[]},
    # ── Section 6 · Legends / Fits ────────────────────────────────────────────
    "showLegend":               {"description":"Show/hide the legend.",                    "type":"boolean","graph_types":["all"],            "valid_values":[]},
    "legendPosition": {
        "description": "Position of the legend on the canvas.",
        "type": "string",
        "graph_types": ["all"],
        "valid_values": ["topRight","right","bottomRight","bottom","bottomLeft","left","topLeft","top"],
    },
    "legendColumns": {"description":"Number of columns used to lay out legend entries.",     "type":"numeric","graph_types":["all"],"valid_values":[]},
    "showRegressionFit":        {"description":"Overlay a regression line.",               "type":"boolean","graph_types":["Scatter2D","ScatterBubble2D","Contour"],"valid_values":[]},
    "showLoessFit":             {"description":"Overlay a LOESS smooth fit.",              "type":"boolean","graph_types":["Scatter2D","ScatterBubble2D"],"valid_values":[]},
    "showQuantileRegressionFit":  {"description":"Overlay a quantile regression line on scatter plots.", "type":"boolean","graph_types":["Scatter2D","ScatterBubble2D"],"valid_values":[]},
    "showConfidenceIntervals":  {"description":"Show confidence bands around fit lines.",  "type":"boolean","graph_types":["Scatter2D","ScatterBubble2D"],"valid_values":[]},
    "regressionType": {
        "description": "Type of regression line to overlay on scatter plots.",
        "type": "string",
        "graph_types": ["Scatter2D","ScatterBubble2D","Spaghetti","Contour"],
        "valid_values": ["linear","exponential","logarithmic","power","polynomial"],
    },
    "smpOverlays":         {"description":"Sample metadata columns shown as annotation tracks on heatmaps.","type":"array","graph_types":["Heatmap","Correlation"],"valid_values":[]},
    "varOverlays":         {"description":"Variable metadata columns shown as annotation tracks on heatmaps.","type":"array","graph_types":["Heatmap","Correlation"],"valid_values":[]},
    "decorations":  {"description":"List of line/point/text overlay objects.",         "type":"array", "graph_types":["all"],"valid_values":[]},
    # ── Section 7 · Graph-specific ────────────────────────────────────────────
    # Area
    "areaType": {
        "description": "How area series are drawn. REQUIRED for Area charts.",
        "type": "string",
        "graph_types": ["Area","AreaLine"],
        "valid_values": ["overlapping","stacked","percent"],
    },
    "areaStyle": {
        "description": "Style for area graphs. The options include solid which is the default, translucent, and outlined.",
        "type": "string",
        "graph_types": ["Area","AreaLine"],
        "valid_values": ["solid","translucent","outlined"],
    },
    # Line
    "lineType": {
        "description": "Interpolation style for line series.",
        "type": "string",
        "graph_types": ["Line","AreaLine","BarLine","DotLine","Spaghetti","TimeSeries"],
        "valid_values": ["rect","solid","spline","dotted","dashed","dotdash","longdash"],
    },
    "lineErrorType": {
        "description": "How error ranges are displayed on line charts.",
        "type": "string",
        "graph_types": ["Line","AreaLine","Spaghetti","TimeSeries"],
        "valid_values": ["bar","area"],
    },
    # Boxplot / Violin
    "boxplotWhiskersType":   {"description":"Style of boxplot whiskers.",                         "type":"string", "graph_types":["Boxplot"],"valid_values":["single","double","none"]},
    "showBoxplotIfViolin":   {"description":"Show inner boxplot inside violin plots.",             "type":"boolean","graph_types":["Violin"],"valid_values":[]},
    "violinTrim":               {"description":"Trim violin tails to data range.",         "type":"boolean","graph_types":["Violin"],"valid_values":[]},
    "boxplotNotched":           {"description":"Render notched boxplots.",                 "type":"boolean","graph_types":["Boxplot"],"valid_values":[]},
    "boxplotType": {
        "description": "Style of the boxplot whiskers.",
        "type": "string",
        "graph_types": ["Boxplot"],
        "valid_values": ["boxWhiskers","range"],
    },
    "violinScale": {
        "description": "How violin widths are scaled relative to each other.",
        "type": "string",
        "graph_types": ["Violin"],
        "valid_values": ["area","count","width"],
    },
    "boxplotConnect":        {"description":"Connect matched boxplot groups across categories with lines.", "type":"boolean","graph_types":["Boxplot"],"valid_values":[]},
    "boxplotMean":           {"description":"Show a mean indicator marker inside each boxplot.",   "type":"boolean","graph_types":["Boxplot"],"valid_values":[]},
    "boxplotMeanColor":      {"description":"Color of the mean indicator in boxplots.",            "type":"string", "graph_types":["Boxplot"],"valid_values":[]},
    "boxplotMedianColor":    {"description":"Color of the median line in boxplots.",               "type":"string", "graph_types":["Boxplot"],"valid_values":[]},
    "boxplotMedianWidth":    {"description":"Stroke width of the median line in boxplots.",        "type":"numeric","graph_types":["Boxplot"],"valid_values":[]},
    "boxplotOutliersColor":  {"description":"Color of outlier data points in boxplots.",           "type":"string", "graph_types":["Boxplot"],"valid_values":[]},
    "boxplotOutliersRatio":  {"description":"Size ratio of outlier points relative to normal data points.", "type":"numeric","graph_types":["Boxplot"],"valid_values":[]},
    "boxplotOutliersShape":  {"description":"Shape of outlier data points in boxplots.",           "type":"string", "graph_types":["Boxplot"],"valid_values":["circle","square","triangle","diamond","plus","minus","star","circleOpen","squareOpen","triangleOpen","diamondOpen"]},
    "showViolinBoxplot":        {"description":"Embed a boxplot inside each violin.",      "type":"boolean","graph_types":["Violin"],"valid_values":[]},
    "showViolinQuantiles":      {"description":"Show quantile lines on violin.",           "type":"boolean","graph_types":["Violin"],"valid_values":[]},
    # Histogram / Density
    "histogramType": {
        "description": "How multiple histogram series are combined. REQUIRED for Histogram charts.",
        "type": "string",
        "graph_types": ["Histogram"],
        "valid_values": ["dodged","staggered","stacked"],
    },
    "histogramStat":             {"description":"Statistical measure used for histogram bar heights.", "type":"string", "graph_types":["Histogram"],"valid_values":["count","density"]},
    "showHistogram":             {"description":"Overlay a histogram for specific data groups.",       "type":"boolean","graph_types":["Histogram"],"valid_values":[]},
    "showHistogramDensity":      {"description":"Overlay a density curve on the histogram.",          "type":"boolean","graph_types":["Histogram"],"valid_values":[]},
    "showFilledHistogramDensity":{"description":"Overlay a filled density curve on the histogram.",   "type":"boolean","graph_types":["Histogram"],"valid_values":[]},
    "showHistogramMedian":       {"description":"Show a median line in density plots.",               "type":"boolean","graph_types":["Density","Ridgeline"],"valid_values":[]},
    "showHistogramBars":         {"description":"Show histogram bars in density plots.",              "type":"boolean","graph_types":["Density","Ridgeline"],"valid_values":[]},
    "densityPosition": {
        "description": "How density curves are positioned relative to each other. REQUIRED for Density charts.",
        "type": "string",
        "graph_types": ["Density"],
        "valid_values": ["normal","stacked","filled"],
    },
    "densityKernel":             {"description":"Kernel function used for density estimation.",        "type":"string", "graph_types":["Density","Ridgeline"],"valid_values":["gaussian","rectangular","triangular","epanechnikov","quartic","biweight","cosine","optcosine"]},
    "showHistogramQuantiles":    {"description":"Show quantile lines in ridgeline / density plots.",  "type":"boolean","graph_types":["Ridgeline","Density"],"valid_values":[]},
    "showHistogramDataPoints":   {"description":"Show individual data points in ridgeline plots.",    "type":"boolean","graph_types":["Ridgeline","Density"],"valid_values":[]},
    # Contour
    "contourFilled":         {"description":"Render filled contour regions between isolines.",         "type":"boolean","graph_types":["Contour"],"valid_values":[]},
    "showContourDataPoints": {"description":"Overlay the underlying data points on contour plots.",    "type":"boolean","graph_types":["Contour"],"valid_values":[]},
    "showContourBands":      {"description":"Show filled band regions between contour isolines.",      "type":"boolean","graph_types":["Contour"],"valid_values":[]},
    "contourStat":           {"description":"Statistical method used to compute contour density.",     "type":"string", "graph_types":["Contour"],"valid_values":["density","ndensity","count"]},
    # Chord
    "chordThickness": {"description":"Thickness of chord ribbons in chord diagrams.",                  "type":"numeric","graph_types":["Chord"],"valid_values":[]},
    # Heatmap
    "variablesClustered":       {"description":"Hierarchical clustering of variables.",    "type":"boolean","graph_types":["Heatmap","Correlation"],"valid_values":[]},
    "samplesClustered":         {"description":"Hierarchical clustering of samples.",      "type":"boolean","graph_types":["Heatmap","Correlation"],"valid_values":[]},
    "heatmapIndicator":         {"description":"Show color scale bar on heatmap.",         "type":"boolean","graph_types":["Heatmap","Correlation"],"valid_values":[]},
    "heatmapIndicatorHistogram":      {"description":"Show a histogram inside the heatmap color-scale indicator.", "type":"boolean","graph_types":["Heatmap","Correlation"],"valid_values":[]},
    "heatmapIndicatorHistogramColor": {"description":"Color of the histogram bars inside the heatmap indicator.",  "type":"string", "graph_types":["Heatmap","Correlation"],"valid_values":[]},
    "heatmapIndicatorPosition":       {"description":"Position of the heatmap color-scale indicator.",             "type":"string", "graph_types":["Heatmap","Correlation"],"valid_values":["topLeft","top","topRight","right"]},
    "heatmapIndicatorHeight":         {"description":"Height of the heatmap color-scale indicator in pixels.",     "type":"numeric","graph_types":["Heatmap","Correlation"],"valid_values":[]},
    "heatmapIndicatorWidth":          {"description":"Width of the heatmap color-scale indicator in pixels.",      "type":"numeric","graph_types":["Heatmap","Correlation"],"valid_values":[]},
    "dendrogramSpace":{"description":"Pixels reserved for dendrograms on heatmaps.",  "type":"numeric","graph_types":["Heatmap","Correlation"],"valid_values":[]},
    # Bullet
    "bulletTargetVarName": {"description":"Column name containing the target value for bullet charts.", "type":"string","graph_types":["Bullet"],"valid_values":[]},
    "rangeStack":          {"description":"Columns defining the qualitative range band stack for bullet charts.", "type":"array","graph_types":["Bullet"],"valid_values":[]},
    # Circular
    "rAxis":           {"description":"Column mapped to the radial axis of circular charts.",          "type":"string", "graph_types":["Circular"],"valid_values":[]},
    "circularTrackName":{"description":"Display names for each track in circular charts.",             "type":"string", "graph_types":["Circular"],"valid_values":[]},
    "circularType":    {"description":"Layout type of the circular chart.",                            "type":"string", "graph_types":["Circular"],"valid_values":["normal","radar","sunburst","chord","bubble"]},
    "circularRotate":  {"description":"Rotation of the entire circular chart in degrees.",             "type":"numeric","graph_types":["Circular"],"valid_values":[]},
    # Venn
    "vennGroups":   {"description":"Array of group definitions for Venn diagrams.",    "type":"array", "graph_types":["Venn"],"valid_values":[]},
    # Kaplan-Meier
    "kmPvalue":                  {"description":"Show the log-rank p-value on Kaplan-Meier plots.",               "type":"boolean","graph_types":["KaplanMeier"],"valid_values":[]},
    "kmRiskTable":               {"description":"Column used to build the at-risk table on Kaplan-Meier plots.",  "type":"string", "graph_types":["KaplanMeier"],"valid_values":[]},
    "showKMConfidenceIntervals": {"description":"Show confidence interval bands on Kaplan-Meier plots.",          "type":"boolean","graph_types":["KaplanMeier"],"valid_values":[]},
    "showKMMedianSurvivalTime":  {"description":"Show a horizontal line at the median survival time.",            "type":"boolean","graph_types":["KaplanMeier"],"valid_values":[]},
    # Map
    "mapProjection":   {"description":"Cartographic projection used for map charts.",                  "type":"string", "graph_types":["Map"],"valid_values":["mercator","albers","orthographic"]},
    "mapColor":        {"description":"Default fill color for map regions with no data.",               "type":"string", "graph_types":["Map"],"valid_values":[]},
    "mapGraticuleShow":{"description":"Show latitude/longitude grid lines (graticule) on map charts.", "type":"boolean","graph_types":["Map"],"valid_values":[]},
    "useLeaflet":      {"description":"Use Leaflet.js for interactive tile-based map rendering.",       "type":"boolean","graph_types":["Map"],"valid_values":[]},
    # Streamgraph
    "scatterStreamType": {"description":"Layout style controlling how stream layers are stacked.",     "type":"string", "graph_types":["Streamgraph"],"valid_values":["mirror","ridge","proportional"]},
    # ── Section 8 · Facets ────────────────────────────────────────────────────
    "segregateSamplesBy":  {"description":"Columns used to facet samples into sub-plots.", "type":"array","graph_types":["all"],"valid_values":[]},
    "segregateVariablesBy":{"description":"Columns used to facet variables into sub-plots.","type":"array","graph_types":["all"],"valid_values":[]},
    "pivotBy":          {"description":"Column used to reshape data wide-to-long.",    "type":"string", "graph_types":["all"],"valid_values":[]},
    # ── Non-JS params (not in setAutocompleteObject) ──────────────────────────
    "xAxis":            {"description":"Column(s) to plot on the x-axis.",            "type":"array",  "graph_types":["all"],"valid_values":[]},
    "xAxis2":           {"description":"Second x-axis column for combined chart types.","type":"array", "graph_types":["AreaLine","BarLine","DotLine","Pareto","StackedLine","StackedPercentLine"],"valid_values":[]},
    "yAxis":            {"description":"Column(s) to plot on the y-axis (multi-dimensional charts only).","type":"array","graph_types":["Scatter2D","Scatter3D","ScatterBubble2D","Spaghetti","Volcano","KaplanMeier","Contour","Streamgraph","Bump"],"valid_values":[]},
    "zAxis":            {"description":"Column for the z-axis (3D / bubble size).",   "type":"array",  "graph_types":["Scatter3D","ScatterBubble2D"],"valid_values":[]},
    "stackBy":          {"description":"Column used to stack samples in bar charts.",  "type":"string", "graph_types":["Bar","Stacked","StackedPercent"],"valid_values":[]},
    "sortDir": {
        "description": "Sort direction for bars or samples.",
        "type": "string",
        "graph_types": ["Bar","Stacked","StackedPercent","Lollipop","Waterfall"],
        "valid_values": ["ascending","descending"],
    },
    "showDataPoints":           {"description":"Show individual data points on line.",     "type":"boolean","graph_types":["Line","AreaLine","Spaghetti"],"valid_values":[]},
    "barZero":                  {"description":"Force zero in bar charts with all-positive values.","type":"boolean","graph_types":["Bar","Stacked","StackedPercent"],"valid_values":[]},
    "background":     {"description":"Canvas background colour (any CSS colour string).","type":"string","graph_types":["all"],"valid_values":[]},
    "dumbbellType": {
        "description": "Style of the Dumbbell chart connector. REQUIRED for Dumbbell charts.",
        "type": "string",
        "graph_types": ["Dumbbell"],
        "valid_values": ["arrow","bullet","cleveland","connected","line","lineConnected","stacked"],
    },
    "ganttStart":   {"description":"Column containing task start dates.",              "type":"string","graph_types":["Gantt"],"valid_values":[]},
    "ganttEnd":     {"description":"Column containing task end dates.",                "type":"string","graph_types":["Gantt"],"valid_values":[]},
}


# ---------------------------------------------------------------------------
# In-memory cache
# ---------------------------------------------------------------------------

class _SchemaCache:
    def __init__(self) -> None:
        self._schema: dict[str, dict] = {}
        self._loaded_at: float = 0.0
        self._source: str = "not loaded"

    def is_fresh(self) -> bool:
        return bool(self._schema) and (time.time() - self._loaded_at) < SCHEMA_TTL

    def set(self, schema: dict[str, dict], source: str) -> None:
        self._schema   = schema
        self._loaded_at = time.time()
        self._source   = source
        log.info("cx_knowledge: schema loaded from %s (%d params)", source, len(schema))

    def get(self) -> dict[str, dict]:
        return self._schema

    @property
    def source(self) -> str:
        return self._source


_cache = _SchemaCache()


# ---------------------------------------------------------------------------
# GitHub fetch
# ---------------------------------------------------------------------------

def _fetch_url(url: str, timeout: int = 15) -> Optional[str]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "canvasxpress-mcp/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8")
    except Exception as e:
        log.debug("Fetch failed for %s: %s", url, e)
        return None


def _fetch_and_cache_files() -> dict[str, str]:
    """
    Download each target file from GitHub, cache locally, and return
    a dict of {filename: content}.  Returns {} if all downloads fail.
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, str] = {}
    for fname in _FETCH_FILES:
        cache_path = _CACHE_DIR / fname
        url        = f"{GITHUB_BASE}/{fname}"
        content    = _fetch_url(url)
        if content:
            cache_path.write_text(content, encoding="utf-8")
            results[fname] = content
            log.debug("cx_knowledge: fetched %s (%d chars)", fname, len(content))
        elif cache_path.exists():
            results[fname] = cache_path.read_text(encoding="utf-8")
            log.debug("cx_knowledge: using cached %s", fname)
        else:
            log.warning("cx_knowledge: could not fetch or find cache for %s", fname)
    return results


# ---------------------------------------------------------------------------
# Markdown parser
# ---------------------------------------------------------------------------

# Regex for valid-value lists like: "linear"|"exponential"|...
_VALUE_PATTERN = re.compile(r'"([^"]+)"')

# Regex for parameter headings like:  ### colorScheme  or  **colorScheme**
_PARAM_HEADING = re.compile(
    r'^(?:#{1,4}|[*]{2})\s*([a-zA-Z][a-zA-Z0-9_]+)(?:\s*[*]{2})?\s*$',
    re.MULTILINE,
)

# Graph type names for tagging
_ALL_GRAPH_TYPES = {
    "Alluvial","Area","AreaLine","Bar","BarLine","Boxplot","Bin","Binplot",
    "Bubble","Bullet","Bump","CDF","Chord","Circular","Cleveland","Contour",
    "Correlation","Density","Distribution","Donut","DotLine","Dotplot",
    "Dumbbell","Gantt","Heatmap","Hex","Hexplot","Histogram","KaplanMeier",
    "Line","Lollipop","Map","Meter","Network","ParallelCoordinates","Pareto",
    "Pie","QQ","Quantile","Radar","Ribbon","Ridgeline","Sankey","Scatter2D",
    "Scatter3D","ScatterBubble2D","Spaghetti","Stacked","StackedLine",
    "StackedPercent","StackedPercentLine","Streamgraph","Sunburst","TagCloud",
    "TimeSeries","Tornado","Tree","Treemap","Upset","Violin","Volcano",
    "Venn","Waterfall","WordCloud",
}


def _extract_graph_types(text: str) -> list[str]:
    """Return graph type names mentioned in a block of text."""
    found = [gt for gt in _ALL_GRAPH_TYPES if gt in text]
    return found if found else ["all"]


def _parse_schema_md(content: str) -> dict[str, dict]:
    """
    Parse SCHEMA.md into a dict of {param_name: {description, valid_values, graph_types, type}}.
    Heuristic: split on parameter headings, collect surrounding text as description
    and extract quoted values as valid_values.
    """
    schema: dict[str, dict] = {}

    # Split on H2/H3/H4 headings or bold **param** markers
    sections = re.split(r'\n(?=#{1,4} |\*\*[a-zA-Z])', content)

    for section in sections:
        lines = section.strip().splitlines()
        if not lines:
            continue

        # Extract parameter name from first line
        m = _PARAM_HEADING.match(lines[0].strip())
        if not m:
            continue
        param_name = m.group(1)

        # Skip obvious section headers (all caps, very short)
        if param_name.isupper() and len(param_name) < 4:
            continue

        body = "\n".join(lines[1:]).strip()

        # Description: first non-empty, non-code line
        description = ""
        for line in lines[1:]:
            line = line.strip()
            if line and not line.startswith("`") and not line.startswith("|"):
                description = line.lstrip("-: ").strip()
                break

        # Valid values: collect all "quoted" strings in the section
        valid_values = _VALUE_PATTERN.findall(body)
        # Filter out obvious non-values (column names, file paths, etc.)
        valid_values = [v for v in valid_values if len(v) < 40 and "/" not in v]

        # Determine type
        if valid_values:
            param_type = "string"
        elif re.search(r'\b(true|false|boolean)\b', body, re.I):
            param_type = "boolean"
        elif re.search(r'\b(integer|number|float|numeric)\b', body, re.I):
            param_type = "numeric"
        else:
            param_type = "string"

        graph_types = _extract_graph_types(body)

        schema[param_name] = {
            "description":  description or f"{param_name} parameter.",
            "type":         param_type,
            "valid_values": list(dict.fromkeys(valid_values)),  # deduplicate, preserve order
            "graph_types":  graph_types,
        }

    return schema


def _parse_rules_md(content: str, schema: dict[str, dict]) -> None:
    """
    Augment the schema with any additional valid_values found in RULES.md
    (e.g. areaType, densityPosition, histogramType rules sections).
    Mutates schema in-place.
    """
    for param, values in [
        ("areaType",        ["overlapping", "stacked", "percent"]),
        ("densityPosition", ["normal", "stacked", "filled"]),
        ("histogramType",   ["dodged", "staggered", "stacked"]),
        ("dumbbellType",    ["arrow", "bullet", "cleveland", "connected", "line", "lineConnected", "stacked"]),
    ]:
        if param not in schema:
            schema[param] = _BUNDLED_SCHEMA.get(param, {
                "description": f"{param} — see RULES.md",
                "type": "string", "valid_values": values, "graph_types": ["all"],
            })
        elif not schema[param].get("valid_values"):
            schema[param]["valid_values"] = values


# ---------------------------------------------------------------------------
# Schema loading (public entry point)
# ---------------------------------------------------------------------------

def load_schema(force: bool = False) -> dict[str, dict]:
    """
    Return the parameter schema dict.  Uses the in-memory cache unless
    stale or force=True.  Falls back gracefully through:
        GitHub → local cache → bundled minimal schema
    """
    if not force and _cache.is_fresh():
        return _cache.get()

    if SKIP_FETCH:
        log.info("cx_knowledge: CX_SKIP_FETCH=1 — using bundled schema")
        schema = dict(_BUNDLED_SCHEMA)
        _cache.set(schema, "bundled (skip_fetch)")
        return schema

    files = _fetch_and_cache_files()

    if not files:
        log.warning("cx_knowledge: no files available — using bundled schema")
        schema = dict(_BUNDLED_SCHEMA)
        _cache.set(schema, "bundled (no files)")
        return schema

    # Parse SCHEMA.md as the primary source
    schema: dict[str, dict] = {}
    if "SCHEMA.md" in files:
        schema = _parse_schema_md(files["SCHEMA.md"])
        log.debug("cx_knowledge: parsed %d params from SCHEMA.md", len(schema))

    # Augment with RULES.md
    if "RULES.md" in files:
        _parse_rules_md(files["RULES.md"], schema)

    # Merge bundled entries for any missing params
    for param, entry in _BUNDLED_SCHEMA.items():
        if param not in schema:
            schema[param] = entry

    source = "GitHub" if any(
        not (_CACHE_DIR / f).exists() for f in _FETCH_FILES if f in files
    ) else "cache"

    _cache.set(schema, source)
    return schema


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def get_params_for_graph_type(graph_type: str) -> dict[str, dict]:
    """
    Return all parameters that apply to the given graph type.
    Includes params tagged 'all' and params explicitly listing this graph type.
    """
    schema     = load_schema()
    gt_lower   = graph_type.lower()
    result     = {}
    for param, entry in schema.items():
        gts = [g.lower() for g in entry.get("graph_types", [])]
        if "all" in gts or gt_lower in gts:
            result[param] = entry
    return result


def get_param_info(param_name: str) -> Optional[dict]:
    """Return the full schema entry for a single parameter, or None if unknown."""
    return load_schema().get(param_name)


def get_param_snippet(graph_type: Optional[str] = None, max_params: int = 20) -> str:
    """
    Build a concise prompt snippet listing valid values for key parameters.
    Used by build_system_prompt() to inject live schema data.

    Returns a string ready to append to the system prompt, or '' if schema
    is unavailable or would add no useful information.
    """
    schema = load_schema()
    if not schema:
        return ""

    # Prioritise params with known valid_values and relevant to this graph type
    if graph_type:
        params = get_params_for_graph_type(graph_type)
    else:
        params = schema

    lines = []
    for param, entry in params.items():
        vals = entry.get("valid_values", [])
        if not vals:
            continue
        vals_str = " | ".join(f'"{v}"' for v in vals[:12])
        if len(vals) > 12:
            vals_str += f" ... ({len(vals)} total)"
        lines.append(f"  {param}: {vals_str}")
        if len(lines) >= max_params:
            break

    if not lines:
        return ""

    header = (
        f"## VALID PARAMETER VALUES"
        + (f" for {graph_type}" if graph_type else "")
        + f" (from canvasxpress-LLM, {_cache.source})\n"
    )
    return header + "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Value validation (extends existing column-ref validation)
# ---------------------------------------------------------------------------

def validate_param_values(config: dict) -> dict:
    """
    Check every string-valued config parameter against its known valid values.

    Returns:
        {
          "warnings":      list of warning strings,
          "invalid_values": dict of {param: {"given": value, "valid": [...]}}
        }

    Only parameters with a known, non-empty valid_values list are checked.
    Boolean and numeric params, and params with open-ended valid_values, are skipped.
    """
    schema        = load_schema()
    warnings:      list[str] = []
    invalid_values: dict[str, dict] = {}

    for param, value in config.items():
        entry = schema.get(param)
        if not entry:
            continue
        valid_vals = entry.get("valid_values", [])
        if not valid_vals or entry.get("type") != "string":
            continue

        # value may be a string or list of strings
        candidates = [value] if isinstance(value, str) else (
            [v for v in value if isinstance(v, str)] if isinstance(value, list) else []
        )

        bad = [v for v in candidates if v not in valid_vals]
        if bad:
            invalid_values[param] = {"given": bad, "valid": valid_vals}
            warnings.append(
                f"'{param}' has invalid value(s) {bad}. "
                f"Valid: {valid_vals[:8]}"
                + (" ..." if len(valid_vals) > 8 else "")
            )

    return {"warnings": warnings, "invalid_values": invalid_values}


# ---------------------------------------------------------------------------
# MCP tool handler (called from server.py)
# ---------------------------------------------------------------------------

def handle_query_params(
    graph_type: Optional[str] = None,
    param_name: Optional[str] = None,
) -> dict:
    """
    Core logic for the query_canvasxpress_params MCP tool.

    If param_name is given: return full info for that one parameter.
    If graph_type is given: return all parameters for that chart type.
    If both given: return param info scoped to that graph type.
    If neither: return all parameters (paginated summary).
    """
    schema = load_schema()

    # ── Single parameter lookup ───────────────────────────────────────────────
    if param_name:
        entry = schema.get(param_name)
        if not entry:
            # Fuzzy match — find close names
            close = [p for p in schema if param_name.lower() in p.lower()]
            return {
                "found":       False,
                "param":       param_name,
                "suggestions": close[:8],
                "schema_source": _cache.source,
            }
        result = {
            "found":        True,
            "param":        param_name,
            "description":  entry["description"],
            "type":         entry["type"],
            "valid_values": entry["valid_values"],
            "graph_types":  entry["graph_types"],
            "schema_source": _cache.source,
        }
        # If graph_type also given, note whether param applies
        if graph_type:
            gts = [g.lower() for g in entry["graph_types"]]
            result["applies_to_graph_type"] = (
                "all" in gts or graph_type.lower() in gts
            )
        return result

    # ── Graph-type parameter listing ──────────────────────────────────────────
    if graph_type:
        params = get_params_for_graph_type(graph_type)
        if not params:
            return {
                "graph_type":    graph_type,
                "param_count":   0,
                "params":        {},
                "schema_source": _cache.source,
                "note":          f"No parameters found for '{graph_type}'. Check the graph type name.",
            }
        # Slim down for readability — omit empty valid_values unless they're boolean
        slim = {}
        for p, e in params.items():
            slim[p] = {
                "description":  e["description"],
                "type":         e["type"],
                "valid_values": e["valid_values"],
            }
        return {
            "graph_type":   graph_type,
            "param_count":  len(slim),
            "params":       slim,
            "schema_source": _cache.source,
        }

    # ── Full schema summary ───────────────────────────────────────────────────
    summary = {}
    for p, e in schema.items():
        summary[p] = {
            "description":  e["description"],
            "type":         e["type"],
            "valid_values": e["valid_values"][:6] if e["valid_values"] else [],
            "has_more":     len(e["valid_values"]) > 6,
            "graph_types":  e["graph_types"][:4],
        }
    return {
        "param_count":   len(summary),
        "params":        summary,
        "schema_source": _cache.source,
        "tip":           "Pass graph_type= to filter, or param_name= for full detail on one param.",
    }


# ---------------------------------------------------------------------------
# Startup: warm the cache
# ---------------------------------------------------------------------------

def filter_unknown_params(config: dict) -> tuple[dict, list[str]]:
    """
    Remove any config keys not present in the known parameter schema.

    Primary defence against hallucinated parameter names. Called after every
    LLM response is parsed, before the config is returned to the caller.

    Returns:
        (filtered_config, removed_keys)
    """
    schema = load_schema()

    # Keys always allowed regardless of schema (structural / catch-all)
    _ALWAYS_ALLOWED = {
        "graphType", "data", "events",
        "xAxis", "xAxis2", "yAxis", "zAxis",
        "ganttStart", "ganttEnd",
        # Sorting and Filtering
        "sortData", "filterData",
        # Grouping and Hierarchy
        "groupingFactors", "hierarchy", "sankeyAxes",
        # Labeling and Selecting
        "labelBy", "labelSelect", "optimizeTextPosition",
        # Axis and Data Transformations
        "xAxisTransform", "yAxisTransform", "transformData",
        # Range and Scales
        "setMinX", "setMaxX", "setMinY", "setMaxY", "setMin", "setMax",
        # Graph Orientation
        "graphOrientation",
        # Themes and color schemes
        "theme", "colorScheme",
        # Graph Title
        "title", "titleScaleFontFactor", "titleColor", "titleFontStyle",
        # Graph Subtitle
        "subtitle", "subtitleScaleFontFactor", "subtitleColor", "subtitleFontStyle",
        # Citation
        "citation", "citationScaleFontFactor", "citationColor", "citationFontStyle",
        # X Axis Titles & Ticks
        "xAxisTitle", "xAxisTitleScaleFontFactor", "xAxisTitleColor", "xAxisTextScaleFontFactor", "xAxisTextColor",
        # Y Axis Titles & Ticks
        "yAxisTitle", "yAxisTitleScaleFontFactor", "yAxisTitleColor", "yAxisTextScaleFontFactor", "yAxisTextColor",
        # Z Axis Titles & Ticks
        "zAxisTitle", "zAxisTitleScaleFontFactor", "zAxisTitleColor", "zAxisTextScaleFontFactor", "zAxisTextColor",
        # Sample Titles & Labels
        "smpTitle", "smpTitleScaleFontFactor", "smpTitleColor", "smpTitleFontStyle",
        "smpTextScaleFontFactor", "smpTextColor", "smpTextFontStyle", "smpTextRotate",
        # Variable Titles & Labels
        "varTitle", "varTitleScaleFontFactor", "varTitleColor", "varTitleFontStyle",
        "varTextScaleFontFactor", "varTextColor", "varTextFontStyle", "varTextRotate",
        # Sankey Titles & Labels
        "sankeyTitleScaleFontFactor", "sankeyTitleColor", "sankeyTextScaleFontFactor", "sankeyTextColor",
        # Aesthetics by Column Mapping
        "colorBy", "shapeBy", "sizeBy", "lineBy", "ridgeBy", "ellipseBy",
        # Data point sizing & layout styles
        "dataPointSizeScaleFactor", "dataPointSize", "showBoxplotOriginalData",
        "jitter", "sina", "binned", "binAlignment",
        # Guides & object borders
        "guidesShow", "guidesColor", "guidesWidth", "guidesLineType", "objectBorderColor",
        # Background
        "background",
        # Legends
        "showLegend", "legendPosition", "legendColumns",
        # Fits
        "showRegressionFit", "showLoessFit", "showQuantileRegressionFit", "showConfidenceIntervals", "regressionType",
        # Data Overlays
        "smpOverlays", "varOverlays",
        # Line Decorations
        "decorations",
        # Area & Line styles
        "areaType", "areaStyle", "lineType",
        # Boxplot & Violin plots
        "boxplotWhiskersType", "showBoxplotIfViolin", "violinTrim", "boxplotNotched", "boxplotType", "violinScale",
        "boxplotConnect", "boxplotMean", "boxplotMeanColor", "boxplotMedianColor", "boxplotMedianWidth",
        "boxplotOutliersColor", "boxplotOutliersRatio", "boxplotOutliersShape",
        "showViolinBoxplot", "showViolinQuantiles",
        # Distributions (Density, Histogram, Ridgeline)
        "histogramType", "histogramStat", "showHistogram",
        "showHistogramDensity", "showFilledHistogramDensity", "showHistogramMedian",
        "showHistogramBars", "densityPosition", "densityKernel", "showHistogramQuantiles", "showHistogramDataPoints",
        # Contour Plots
        "contourFilled", "showContourDataPoints", "showContourBands", "contourStat",
        # Chord
        "chordThickness",
        # Clustering, Heatmap, Circular, Venn, Bullet
        "variablesClustered", "samplesClustered",
        "heatmapIndicator", "heatmapIndicatorHistogram", "heatmapIndicatorHistogramColor",
        "heatmapIndicatorPosition", "heatmapIndicatorHeight", "heatmapIndicatorWidth", "dendrogramSpace",
        "bulletTargetVarName", "rangeStack",
        "rAxis", "circularTrackName", "circularType", "circularRotate",
        "vennGroups",
        # Kaplan-Meier
        "kmPvalue", "kmRiskTable", "showKMConfidenceIntervals", "showKMMedianSurvivalTime",
        # Map
        "mapId", "mapPropertyId", "mapProjection", "mapColor", "mapGraticuleShow", "useLeaflet",
        "topoJSON", "legendOrder", "sizeBy",
        # Streamgraph
        "scatterStreamType",
        # Faceting and Pivoting
        "segregateSamplesBy", "segregateVariablesBy", "pivotBy",
        ## Other
        "showDataPoints", "lineErrorType", "dumbbellType", "sortDir", "barZero", "stackBy",
    }

    filtered: dict = {}
    removed:  list[str] = []

    for key, value in config.items():
        if key in schema or key in _ALWAYS_ALLOWED:
            filtered[key] = value
        else:
            removed.append(key)
            log.warning(
                "cx_knowledge: removed unknown parameter '%s' (value=%s)",
                key, repr(value)[:80],
            )

    if removed:
        log.info(
            "cx_knowledge: stripped %d unknown param(s): %s",
            len(removed), removed,
        )

    return filtered, removed


def warm_cache() -> None:
    """Pre-load the schema at server startup so the first request is fast."""
    try:
        schema = load_schema()
        log.info(
            "cx_knowledge: schema warmed — %d params from %s",
            len(schema), _cache.source,
        )
    except Exception as e:
        log.warning("cx_knowledge: warm_cache failed: %s", e)
