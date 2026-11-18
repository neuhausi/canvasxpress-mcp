# CanvasXpress plugin for Claude

Interactive scientific charts and dashboards from plain English, powered by
[CanvasXpress](https://www.canvasxpress.org) — a mature open JavaScript charting library
built for scientific and bioinformatics data, with 60+ chart types: heatmaps with
dendrograms, volcano and PCA plots, Kaplan-Meier survival curves, violin/box plots,
networks, genome browsers, choropleth and cartogram maps, Sankey diagrams, and more.

## What this plugin contains

- **A remote MCP connector** (`.mcp.json`) pointing at the hosted CanvasXpress MCP server
  at `https://www.canvasxpress.org/mcp`. The server turns natural-language descriptions
  into validated CanvasXpress JSON configs, renders live interactive chart views in chat
  (MCP Apps), builds multi-panel linked dashboards from a dataset or a CSV/TSV link, and
  answers parameter/documentation lookups. No account, sign-up, or API key is required.
- **One skill** (`skills/canvasxpress-charts/`) that guides Claude through the intended
  tool flow: pick a chart type, generate, modify, validate, render, and when to use the
  Kaplan-Meier, map, and dashboard tools.

## What it runs, sends, and fetches

This plugin contains **no scripts, hooks, commands, or local executables** — only the
remote server reference and a Markdown skill. When Claude calls the connector's tools,
the chart description and any tabular data you provide (pasted rows, or a file link you
ask it to read) are sent over HTTPS to the CanvasXpress server to generate or validate
the config; see the [privacy policy](https://www.canvasxpress.org/privacy.html) for
logging and retention details (tool-call logs are deleted after at most 90 days). The
server applies a per-address rate limit.

## Install

In Claude Code (2.1.275 or later):

```
/plugin install canvasxpress --marketplace neuhausi/canvasxpress-mcp
```

## Usage

Ask Claude for a chart or dashboard in plain English:

- "Clustered heatmap of 40 genes by 12 samples with RdBu colors and dendrograms on both axes"
- "Kaplan-Meier survival curve comparing two treatment arms with a risk table"
- "Suggest a dashboard from https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
- "Choropleth map of the US colored by rate per state"

## Links

- Documentation: https://github.com/neuhausi/canvasxpress-mcp
- Website: https://www.canvasxpress.org
- Support: https://www.canvasxpress.org/contact.html
- License: MIT
