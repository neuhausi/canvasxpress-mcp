# CanvasXpress Chart Rendering Skill

Generate publication-quality data visualizations using CanvasXpress and render them as PNG images that can be displayed inline.

## Overview

This skill enables you to:
1. Generate CanvasXpress chart configurations from natural language descriptions
2. Render charts to PNG using the `canvasxpress-cli` tool
3. Display the resulting images inline for the user

It operates in **hybrid mode**: uses the CanvasXpress MCP server when reachable, otherwise falls back to embedded knowledge.

---

## Prerequisites

- **Node.js** ≥ 18
- **canvasxpress-cli** with dependencies installed
- **Playwright browsers** installed (the CLI uses headless Chromium)

## Setup (run once if not already installed)

Before rendering, verify the CLI is available. If any step fails, run the installation:

```bash
# 1. Check if canvasxpress-cli exists (it's a sibling directory in the same repo)
CLI_DIR="$(dirname "$(pwd)")/canvasxpress-cli"
if [ ! -f "$CLI_DIR/bin/canvasxpress" ]; then
  echo "canvasxpress-cli not found at $CLI_DIR"
  echo "Clone the isaac-mcp-server repo which contains both canvasxpress-mcp and canvasxpress-cli:"
  echo "  git clone https://github.com/nicolesmith/isaac-mcp-server.git"
  exit 1
fi

# 2. Install npm dependencies
cd "$CLI_DIR"
if [ ! -d node_modules ]; then
  npm install
fi

# 3. Install Playwright Chromium browser
npx playwright install chromium
```

To verify the setup works:
```bash
cd "$CLI_DIR" && ./bin/canvasxpress png -o ./output/ -c '{"graphType":"Bar"}' -d '{"y":{"vars":["V"],"smps":["A","B"],"data":[[1,2]]}}' --timeout 8000
ls -la ./output/cX.png  # Should exist and be > 0 bytes
```

If Node.js is not installed, install it first:
```bash
curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
source ~/.bashrc
nvm install 22
```

---

## Quick Start — Rendering a Chart

```bash
# Minimal invocation (CLI_DIR is the canvasxpress-cli sibling directory):
cd "$CLI_DIR"
./bin/canvasxpress png -o ./output/ -c '{"graphType":"Bar"}' -d '{"y":{"vars":["Score"],"smps":["A","B","C"],"data":[[10,20,30]]}}' --timeout 8000

# Output: ./output/cX.png (800×800 RGBA PNG)
```

To locate the CLI directory, find it relative to this skill file:
```bash
# If this skill is at <repo>/canvasxpress-mcp/skills/canvasxpress-chart-rendering.md
# then the CLI is at <repo>/canvasxpress-cli/
CLI_DIR="$(cd "$(dirname "$0")/../../canvasxpress-cli" 2>/dev/null && pwd)"
# Or search for it:
CLI_DIR=$(find ~ -path "*/canvasxpress-cli/bin/canvasxpress" -printf '%h\n' 2>/dev/null | head -1 | sed 's|/bin$||')
```

**IMPORTANT**: Always invoke via `./bin/canvasxpress` (not `node index.js`). The CLI uses `process.argv[1]` to resolve its template HTML path.

After rendering, **open the image in VS Code** so the user can see it:
```bash
code /absolute/path/to/output/cX.png
```
This opens the PNG in VS Code's built-in image viewer as a new editor tab.

### CLI Flags

| Flag | Short | Description | Default |
|------|-------|-------------|---------|
| `--config` | `-c` | JSON config string | Boxplot default |
| `--data` | `-d` | JSON data string | Sample data |
| `--output` | `-o` | Output directory | `./` |
| `--width` | `-x` | Image width (px) | 800 |
| `--height` | `-y` | Image height (px) | 800 |
| `--timeout` | `-t` | Render timeout (ms) | 2500 |
| `--input` | `-i` | Input HTML file | — |
| `--target` | — | Target canvas ID | — |
| `--browser` | `-b` | Show browser (debug) | false |

**Important**: Always use `--timeout 8000` (or higher) for complex charts.

---

## Path 1: MCP Server (Preferred)

When the CanvasXpress MCP server is available at `http://0.0.0.0:8100/mcp`, use its tools for best results. The server has 3000+ curated few-shot examples and vector retrieval.

### Available MCP Tools

| Tool | Purpose |
|------|---------|
| `generate_canvasxpress_config` | Generate config from natural language description |
| `modify_canvasxpress_config` | Modify an existing config with plain English instructions |
| `select_canvasxpress_chart` | Recommend the best chart type for given data columns |
| `list_chart_types` | List all supported chart types with descriptions |
| `query_canvasxpress_params` | Look up parameters, valid values, and descriptions |
| `get_axes_info` | Get axis assignment rules for a chart type |
| `explain_config_property` | Explain any config property |
| `generate_km_config` | Specialized Kaplan-Meier survival plot generation |

### Workflow with MCP Server

1. **If unsure which chart type**: Call `select_canvasxpress_chart` with column types and intent
2. **Generate config**: Call `generate_canvasxpress_config` with description (and optionally headers/data/column_types)
3. **Render**: Pass the returned `config` and user's `data` to the CLI
4. **Display**: Show the PNG inline

### MCP Tool Parameters

#### `generate_canvasxpress_config`
```json
{
  "description": "Heatmap of gene expression with RdBu color scheme",
  "headers": ["Gene", "Sample1", "Sample2", "Treatment"],
  "column_types": {"Gene": "string", "Sample1": "numeric", "Sample2": "numeric", "Treatment": "factor"},
  "data": [["Gene","Sample1","Sample2","Treatment"],["BRCA1",1.2,3.4,"Control"]],
  "temperature": 0.0
}
```

#### `select_canvasxpress_chart`
```json
{
  "intent": "AE counts by SOC across treatment arms",
  "column_types": {"SOC": "factor", "Treatment": "factor", "AE_Count": "numeric"},
  "n_samples": 150,
  "llm_first": false
}
```

#### `modify_canvasxpress_config`
```json
{
  "config": {"graphType": "Bar", "xAxis": ["Score"]},
  "instruction": "add a title 'My Chart' and switch to dark theme"
}
```

---

## Path 2: Standalone (Embedded Knowledge)

When the MCP server is not available, generate configs directly using this embedded knowledge.

### Supported Graph Types

**Single-dimensional** (xAxis only, NEVER yAxis):
Bar, Line, Area, Boxplot, Violin, Heatmap, Pie, Donut, Stacked, StackedPercent, Histogram, Density, Dotplot, Lollipop, Waterfall, Cleveland, Dumbbell, Ridgeline, Treemap, Sankey, Chord, Alluvial, Venn, Radar, WordCloud, CDF

**Multi-dimensional** (both xAxis AND yAxis required):
Scatter2D, Scatter3D, ScatterBubble2D, Volcano, Spaghetti, Contour, Streamgraph, Bump, KaplanMeier, TimeSeries

**Combined** (xAxis + xAxis2, NEVER yAxis):
BarLine, AreaLine, DotLine, Pareto, StackedLine, StackedPercentLine

### Critical Axis Rules

#### Single-Dimensional Charts
- `xAxis` = the **NUMERIC** variable(s) being plotted (bar heights, point positions, distribution values)
- Samples = the **CATEGORICAL** labels (gene names, groups, time points) — assigned automatically from data
- Use `smpTitle` to label the categorical axis (NOT yAxisTitle)
- **NEVER** use `yAxis` or `yAxisTitle` on single-dimensional charts

#### Multi-Dimensional Charts
- **MUST** include both `xAxis` and `yAxis`
- Scatter3D and ScatterBubble2D also require `zAxis`
- Use `xAxisTitle` and `yAxisTitle` for axis labels

#### Combined Charts
- `xAxis` for primary numeric series, `xAxis2` for secondary
- **NEVER** use `yAxis`; use `smpTitle` for the categorical axis

### Data Format

CanvasXpress expects data in this structure:
```json
{
  "y": {
    "vars": ["Variable1", "Variable2"],
    "smps": ["Sample1", "Sample2", "Sample3"],
    "data": [[1.2, 3.4, 5.6], [7.8, 9.0, 1.1]]
  },
  "x": {
    "Treatment": ["Control", "Drug", "Control"]
  },
  "z": {
    "Category": ["Gene", "Protein"]
  }
}
```

- `y.vars` — variable/row names (one per row in `data`)
- `y.smps` — sample/column names (one per column in `data`)
- `y.data` — numeric matrix (vars × smps)
- `x` — sample-level annotations (factors for grouping/coloring)
- `z` — variable-level annotations

### Graph Type Selection Guide

| User Intent | graphType | Notes |
|------------|-----------|-------|
| Scatter, PCA, UMAP, tSNE | Scatter2D | Multi-dim |
| Bar chart, bar graph | Bar | Single-dim |
| Box plot, distribution | Boxplot | Single-dim |
| Violin plot | Violin | Single-dim |
| Heatmap, expression matrix | Heatmap | Single-dim |
| Line chart, trends | Line | Single-dim |
| Histogram, frequency | Histogram | Single-dim |
| Volcano plot | Volcano | Multi-dim |
| Survival, Kaplan-Meier | KaplanMeier | Multi-dim; xAxis=time, yAxis=event |
| Pie chart | Pie | Single-dim |
| Bubble chart | ScatterBubble2D | Multi-dim; needs zAxis |
| Correlation matrix | Correlation | Single-dim |
| Sankey, flow diagram | Sankey | Single-dim; needs sankeyAxes |
| Network graph | Network | Special |
| Ridgeline, joy plot | Ridgeline | Single-dim; use ridgeBy |
| Geographic, choropleth | Map | Special; needs mapId |

### Required Parameters by Graph Type

| graphType | Required Param | Valid Values |
|-----------|---------------|--------------|
| Area | `areaType` | "overlapping", "stacked", "percent" |
| Density | `densityPosition` | "normal", "stacked", "filled" |
| Histogram | `histogramType` | "dodged", "staggered", "stacked" |
| Map | `mapId` | See Map section below |
| Ridgeline | `ridgeBy` | column name (NOT groupingFactors) |
| KaplanMeier | `xAxis` + `yAxis` | time column, event column (0/1) |

### Key Parameters

| Parameter | Purpose | Applies To |
|-----------|---------|-----------|
| `graphType` | Chart type (REQUIRED) | All |
| `xAxis` | Numeric variable name(s) | All |
| `yAxis` | Second axis variable(s) | Multi-dim only |
| `groupingFactors` | Categorical columns for grouping/color | 1D charts |
| `colorBy` | Column for color mapping | Scatter, Spaghetti |
| `shapeBy` | Column for point shapes | Scatter |
| `sizeBy` | Column for point/symbol size | ScatterBubble2D |
| `ellipseBy` | Column for confidence ellipses | Scatter2D, Scatter3D |
| `segregateSamplesBy` | Column for faceting | Most charts |
| `colorScheme` | Named color palette | All |
| `theme` | Visual theme | All |
| `title` | Chart title | All |
| `showLegend` | Show/hide legend | All |
| `graphOrientation` | "vertical" or "horizontal" | Bar, Boxplot, etc. |
| `smpTitle` | Label for sample/categorical axis | 1D charts |
| `xAxisTitle` | Label for x-axis | Multi-dim charts |
| `yAxisTitle` | Label for y-axis | Multi-dim charts |
| `transformData` | Data transform | All |
| `filterData` | Filter rows/samples | All |
| `sortData` | Sort data | Most (not Scatter, Histogram, etc.) |

### Color Schemes

Available: YlGn, YlGnBu, GnBu, BuGn, PuBuGn, PuBu, BuPu, RdPu, PuRd, OrRd, YlOrRd, YlOrBr, Purples, Blues, Greens, Oranges, Reds, Greys, PuOr, BrBG, PRGn, PiYG, RdBu, RdGy, RdYlBu, Spectral, RdYlGn, Bootstrap, Economist, Excel, GGPlot, Solarized, PaulTol, ColorBlind, Tableau, WallStreetJournal, Stata, BlackAndWhite, CanvasXpress

### Themes

Available: bw, classic, cx, dark, economist, excel, ggblanket, ggplot, gray, grey, highcharts, igray, light, linedraw, minimal, none, ptol, solarized, stata, tableau, void0, wsj

### Decorations

```json
{
  "decorations": [
    {"type": "line", "value": 2.0, "color": "#e74c3c", "width": 1, "label": "Threshold"},
    {"type": "point", "value": 8.5, "color": "#e67e22", "label": "Marker"}
  ]
}
```

For **multi-dimensional** charts (Scatter2D, Volcano), use `"x"` and `"y"` instead of `"value"`:
```json
{
  "decorations": [
    {"type": "line", "x": 2.0, "color": "#e74c3c", "label": "FC cutoff"},
    {"type": "line", "y": 1.3, "color": "#7f8c8d", "label": "p=0.05"}
  ]
}
```

### Data Transforms

| Parameter | Values |
|-----------|--------|
| `transformData` | "log2", "log10", "-log2", "-log10", "zscore", "percentile", "sqrt" |
| `xAxisTransform` | "log2", "log10", "-log2", "-log10", "sqrt", "percentile" |
| `yAxisTransform` | Same (multi-dim only) |

### Filter & Sort

**Filter** (show only matching data):
```json
{"filterData": [["guess", "Treatment", "like", "Control"]]}
```
Multiple filters (AND): `[["guess","Col1","like","A"],["guess","Col2","different","B"]]`

**Sort**:
```json
{"sortData": [["var", "var", "Expression"]]}
```
For simple bar sorting: `{"sortDir": "ascending"}` or `"descending"`

### Map Charts

Always include `mapId`. Inference from description:

| Description keyword | mapId |
|-------------------|-------|
| world, countries, global | "World" |
| world continents | "WorldContinents" |
| africa | "Africa" |
| asia | "Asia" |
| europe | "Europe" |
| north america | "NorthAmerica" |
| south america | "SouthAmerica" |
| oceania | "Oceania" |
| US states, united states | "USAStates" |
| US counties | "USACounties" |
| canada | "CAN" |
| australia | "AUS" |
| united kingdom | "GBR" |

**Map data**: First column = geographic IDs (ISO-3 codes for world, 2-letter for US states, 5-digit FIPS for counties). Remaining columns = numeric values. Do NOT assign any column to xAxis/yAxis/groupingFactors for maps.

**Map ID column values**:
- World/continents: ISO 3-letter codes ("USA", "GBR", "FRA", etc.)
- WorldContinents: "Africa", "Asia", "Europe", "NorthAmerica", "Oceania", "SouthAmerica"
- USAStates: 2-letter codes ("CA", "TX", "NY", etc.)
- USACounties: 5-digit FIPS ("06037", "48113", etc.)
- CAN: Province names with `mapPropertyId: "prov_name_en"`
- AUS: State names with `mapPropertyId: "STATE_NAME"`
- GBR: HASC codes with `mapPropertyId: "HASC_2"` or names with `mapPropertyId: "NAME_2"`

---

## Complete Workflow Example

### User asks: "Show me a scatter plot of gene expression with PCA"

**Step 1**: Determine chart type → Scatter2D (PCA keyword)

**Step 2**: Build config:
```json
{
  "graphType": "Scatter2D",
  "xAxis": ["PC1"],
  "yAxis": ["PC2"],
  "xAxisTitle": "PC1",
  "yAxisTitle": "PC2",
  "colorBy": "CellType",
  "ellipseBy": "CellType",
  "title": "PCA of Gene Expression",
  "colorScheme": "Tableau"
}
```

**Step 3**: Build data:
```json
{
  "y": {
    "vars": ["PC1", "PC2"],
    "smps": ["Cell1", "Cell2", "Cell3", "Cell4"],
    "data": [[1.2, -0.5, 2.1, -1.8], [0.8, 1.5, -0.3, -1.2]]
  },
  "x": {
    "CellType": ["TypeA", "TypeA", "TypeB", "TypeB"]
  }
}
```

**Step 4**: Render:
```bash
cd "$CLI_DIR"
./bin/canvasxpress png -o ./output/ \
  -c '{"graphType":"Scatter2D","xAxis":["PC1"],"yAxis":["PC2"],"xAxisTitle":"PC1","yAxisTitle":"PC2","colorBy":"CellType","ellipseBy":"CellType","title":"PCA of Gene Expression","colorScheme":"Tableau"}' \
  -d '{"y":{"vars":["PC1","PC2"],"smps":["Cell1","Cell2","Cell3","Cell4"],"data":[[1.2,-0.5,2.1,-1.8],[0.8,1.5,-0.3,-1.2]]},"x":{"CellType":["TypeA","TypeA","TypeB","TypeB"]}}' \
  --timeout 8000
```

**Step 5**: Open the image in VS Code for the user to see:
```bash
code "$CLI_DIR/output/cX.png"
```

### User asks: "Make a bar chart of sales by region"

**Step 1**: Chart type → Bar (single-dimensional)

**Step 2**: Config:
```json
{
  "graphType": "Bar",
  "xAxis": ["Sales"],
  "smpTitle": "Region",
  "title": "Sales by Region",
  "colorScheme": "Tableau"
}
```

**Step 3**: Data:
```json
{
  "y": {
    "vars": ["Sales"],
    "smps": ["East", "West", "North", "South"],
    "data": [[150, 230, 180, 95]]
  }
}
```

**Step 4**: Render and display.

---

## Common Pitfalls

1. **Never use yAxis on single-dimensional charts** — use `smpTitle` for the categorical axis label
2. **Always include `mapId` for Map charts** — it won't render without it
3. **Use `--timeout 8000`** — the default 2500ms is too short for complex charts
4. **Escape JSON properly** in shell commands — use single quotes around the JSON strings
5. **xAxis on 1D charts = the NUMERIC variable** — categorical labels are samples, not xAxis
6. **For Ridgeline, use `ridgeBy`** — not `groupingFactors`
7. **For KaplanMeier**: xAxis = time column, yAxis = event column (0/1), colorBy = grouping
8. **Config JSON must be valid** — no trailing commas, no comments
9. **Output is always `cX.png`** in the specified output directory
10. **Data matrix shape**: `y.data` is vars×smps (rows=variables, columns=samples)

---

## Checking MCP Server Availability

Before choosing a path, check if the MCP server is reachable:

```bash
curl -s -o /dev/null -w "%{http_code}" http://0.0.0.0:8100/mcp 2>/dev/null
```

- HTTP 200 or 405 → server is up, use Path 1 (MCP tools)
- Connection refused or timeout → use Path 2 (embedded knowledge)

---

## Output

The CLI always produces `cX.png` in the output directory. After rendering:
1. Verify the file exists and has non-zero size
2. **Open the PNG in VS Code** so the user can see it:
   ```bash
   code /absolute/path/to/output/cX.png
   ```
   This opens the image in VS Code's built-in image viewer tab. Do NOT just print the file path — the user expects to SEE the chart.
3. If the user wants modifications, adjust the config and re-render
