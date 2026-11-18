---
name: canvasxpress-charts
description: Create interactive scientific charts and dashboards with the CanvasXpress MCP tools. Use when the user asks for a chart, plot, heatmap, volcano plot, PCA, violin/box plot, Kaplan-Meier survival curve, choropleth or cartogram map, network, or a multi-chart dashboard from tabular data (pasted CSV/TSV or a link).
---

# CanvasXpress charts and dashboards

The bundled CanvasXpress connector turns plain-English descriptions into validated
CanvasXpress JSON configs and live interactive chart views. Follow this flow instead of
hand-writing configs — the server validates against the real parameter schema and strips
hallucinated parameters.

## Single chart

1. **Unsure which chart type fits?** Call `select_canvasxpress_chart` with the intent and
   column names/types. Skip this when the user named the chart type.
2. **Generate** with `generate_canvasxpress_config`: pass the plain-English description,
   plus `headers` and `data` rows when the user supplied data. The result is a ready
   `config` (and renders a live interactive chart view inline when the host supports MCP
   Apps).
3. **Specialized generators**: `generate_km_config` for Kaplan-Meier / survival analysis,
   `create_map_config` for choropleth, cartogram, and other maps (world, continents, US
   states, custom topoJSON).
4. **Modify, don't regenerate**: for follow-up changes ("make it horizontal", "RdBu
   colors") call `modify_canvasxpress_config` with the existing config and the
   instruction — it preserves everything else.
5. **Validate / render**: `validate_canvasxpress_config` lints a config against the
   published schema; `render_canvasxpress_config` returns a PNG rendered by the real
   engine in headless Chromium (useful when no interactive view is available).

## Dashboards

For any multi-chart dashboard request, call `suggest_canvasxpress_dashboard` — ONE call,
and the data goes IN the call: pass the dataset, or `url` with any link to a CSV/TSV file
(GitHub, Dropbox, Google Drive/Sheets, S3 links are accepted as-is). It returns a complete
linked dashboard (KPI cards, filters, cross-filtered panels). Use
`compose_canvasxpress_dashboard` only to hand-pick specific charts or mix datasets, and
`validate_dashboard_spec` to check a hand-edited spec. `publish_canvasxpress_dashboard`
publishes a finished spec as a shareable page (ask the user before publishing).

## Reference lookups

- `list_chart_types` — all 60+ supported graph types.
- `query_canvasxpress_params` / `explain_config_property` — parameter names, valid
  values, and docs; use before guessing a parameter.
- `get_axes_info` — which axis keys are valid for a graph type (call before axis changes).
- `get_minimal_parameters` — the minimal required config for a graph type.
- `explain_canvasxpress_r` / `explain_canvasxpress_ggplot` /
  `explain_canvasxpress_dashboards` — guides for R, ggplot2 conversion, and the
  dashboards npm package, when the user wants code for their own app.

## Notes

- No account or API key is needed; the server is public.
- Chart descriptions and any data passed to the tools are sent to the CanvasXpress
  server to generate the config (see https://www.canvasxpress.org/privacy.html).
- Keep pasted datasets reasonably sized; for large files prefer passing a link via
  `suggest_canvasxpress_dashboard`'s `url` parameter.
