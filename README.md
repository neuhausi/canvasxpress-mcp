# CanvasXpress MCP Server

Natural language → CanvasXpress JSON configs, served over HTTP on port 8100.

Describe a chart in plain English. Get back a ready-to-use CanvasXpress JSON config
object ready to pass directly to `new CanvasXpress()`. No CanvasXpress expertise required.

```
"Clustered heatmap with RdBu colors and dendrograms on both axes"
"Volcano plot with log2 fold change on x-axis and -log10 p-value on y-axis"
"Violin plot of gene expression by cell type, Tableau colors"
"Survival curve for two treatment groups"
"PCA scatter plot colored by Treatment with regression ellipses"
```

Supports four LLM backends: **Anthropic API**, **Amazon Bedrock**, **Ollama** (local),
and **OpenAI-compatible** APIs including corporate gateways.

---

## How it works

1. Your description is matched against few-shot examples using **semantic vector search** (sqlite-vec)
2. The top 6 most relevant examples are included as context (RAG)
3. A **tiered system prompt** is assembled from the canvasxpress-LLM knowledge base — only the content relevant to your request is included
4. The configured LLM generates a validated CanvasXpress JSON config
5. Hallucinated parameter names are **stripped** against the known schema
6. If headers/data are provided, all column references are **validated** against them
7. The config is returned ready to pass to `new CanvasXpress()`

---

## Project structure

```
canvasxpress-mcp/
│
├── src/
│   ├── server.py           — FastMCP HTTP server (main entry point)
│   ├── llm_providers.py    — Unified LLM backend (Anthropic, Bedrock, Ollama, OpenAI)
│   ├── cx_knowledge.py     — Parameter knowledge skill (fetch, parse, validate, inject)
│   ├── cx_survival.py      — Kaplan-Meier skill (generate, detect columns, validate, annotate)
│   └── cx_selector.py      — Chart type selection skill (deterministic scorer + optional LLM tiebreaker)
│
├── data/
│   ├── few_shot_examples.json  — RAG examples (add more to improve accuracy)
│   └── embeddings.db           — sqlite-vec vector index (built by build_index.py)
│
├── build_index.py          — builds the vector index from few_shot_examples.json
│
├── test_client.py          — Python test client
├── test_client.pl          — Perl test client
├── test_client.mjs         — Node.js test client (Node 18+)
│
├── USAGE.md                — usage guide (production, SSH tunnel, local)
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Build the vector index (one-time)

```bash
python build_index.py
```

Re-run whenever you add or change `data/few_shot_examples.json`. If you skip this
step the server still works — it falls back to text-similarity matching and logs a warning.

### 3. Configure your LLM provider

```bash
# Quickstart — Anthropic (default)
export ANTHROPIC_API_KEY="sk-ant-..."
```

See the [LLM providers](#llm-providers) section for all four backends.

### 4. Start the server

```bash
python src/server.py
```

Server starts at `http://localhost:8100`. The MCP protocol endpoint is at `/mcp`.
REST endpoints are at `/generate`, `/modify`, `/km`, etc.

**Debug mode** — full reasoning trace per request:

```bash
CX_DEBUG=1 python src/server.py
```

**Run in background (production):**

```bash
nohup python src/server.py > /tmp/cx-server.log 2>&1 &
```

---

## REST endpoints

All endpoints accept `GET` (query parameters) or `POST` (JSON body).
Endpoints that interact with the LLM also support **JSONP** via a `callback=` parameter
for direct integration with CanvasXpress's `askLLM()` function.

| Endpoint | Tool | Required params |
|----------|------|----------------|
| `GET /generate` | Generate a new config | `description` |
| `GET /modify` | Modify an existing config | `config`, `instruction` |
| `GET /km` | Kaplan-Meier config | at least one of: `description`, `headers`, `data`, `config` |
| `GET /params` | Query parameter schema | none (optional: `graph_type`, `param_name`, `refresh`) |
| `GET /axes` | Axis assignment rules | `graph_type` |
| `GET /select` | Recommend a chart type (deterministic + LLM) | `intent`, `column_types` (optional: `llm_first`) |
| `GET /explain` | Explain a config property | `property` |
| `GET /explain-r` | CanvasXpress in R | none (optional: `topic`) |
| `GET /explain-ggplot` | CanvasXpress ggplot2 bridge | none (optional: `topic`) |
| `GET /minimal-params` | Minimal required parameters | `graph_type` |
| `GET /map` | Map visualization config | `map_id` |
| `GET /map` | Map visualization config | `map_id` |
| `POST /feedback` | Rate a tool call thumbs up/down | `request_id`, `rating` |
| `GET /feedback/export` | Export call log | — (optional: `tool`, `rated_only`, `limit`) |
| `POST /feedback/purge` | Delete call log rows (**admin key required**) | — (optional: `tool`, `rated_only`) |
| `GET /ui` | Browser form UI | — |

### Common parameters (all LLM endpoints)

| Parameter | Description |
|-----------|-------------|
| `description` | Plain English chart description. Alias: `prompt`, `q` |
| `headers` | Comma-separated column names: `Gene, Expression, Treatment` |
| `column_types` | Column types: `Gene=string, Expression=numeric, Treatment=factor` |
| `data` | JSON array of arrays (first row = headers): `[["Gene","Expr"],["BRCA1",1.2]]` |
| `temperature` | LLM creativity 0.0–1.0 (default 0.0 = deterministic) |
| `callback` | JSONP callback name — set automatically by CanvasXpress |
| `target` | CanvasXpress chart target ID — passed through to JSONP response |
| `client_id` | CanvasXpress client ID — passed through to JSONP response |

### Examples

```bash
# Generate a config
curl -s "http://localhost:8100/generate?description=Violin+plot+of+expression+by+treatment\
&headers=Expression,Treatment&column_types=Expression=numeric,Treatment=factor"

# Modify an existing config
curl -s "http://localhost:8100/modify?\
config=%7B%22graphType%22%3A%22Heatmap%22%7D\
&instruction=change+colorScheme+to+Spectral+and+add+a+title"

# Kaplan-Meier config from headers
curl -s "http://localhost:8100/km?\
description=OS+curve+by+treatment+arm\
&headers=PatientID,OS_Time,OS_Status,Treatment"

# Query all parameters for a graph type
curl -s "http://localhost:8100/params?graph_type=Heatmap"

# Look up a single parameter
curl -s "http://localhost:8100/params?param_name=colorScheme"

# Axis assignment rules for a chart type
curl -s "http://localhost:8100/axes?graph_type=Scatter2D"

# Recommend a chart type (deterministic scorer + automatic LLM tiebreaker)
curl -s "http://localhost:8100/select?\
intent=show+expression+distribution+by+cell+type\
&column_types=Expression=numeric,CellType=factor"

# Let the LLM pick the chart type from the full catalogue first (llm_first)
curl -s "http://localhost:8100/select?\
intent=show+expression+distribution+by+cell+type\
&column_types=Expression=numeric,CellType=factor&llm_first=true"

# Explain a config property
curl -s "http://localhost:8100/explain?property=groupingFactors"

# Minimal required parameters

# Map config — world choropleth
curl -s "http://localhost:8100/map?map_id=World&color_scheme=Blues&title=World+GDP"

# Map config — US states pie chart
curl -s "http://localhost:8100/map?map_id=USAStates\
&color_by=Winner&size_by=Total\
&topo_json=https://www.canvasxpress.org/data/json/usa-albers-states.json"
curl -s "http://localhost:8100/minimal-params?graph_type=KaplanMeier"

# Map config — world choropleth
curl -s "http://localhost:8100/map?map_id=World&color_scheme=Blues&title=World+GDP"

# Map config — US states pie chart
curl -s "http://localhost:8100/map?map_id=USAStates\
&color_by=Winner&size_by=Total\
&topo_json=https://www.canvasxpress.org/data/json/usa-albers-states.json"
```

### CanvasXpress integration (JSONP)

Set `llmServiceURL` in your CanvasXpress config. CanvasXpress will append `generate`
and add all required JSONP parameters automatically:

**Production (canvasxpress.org):**
```javascript
cx.llmServiceURL = "https://www.canvasxpress.org/";
```

**Local development via SSH tunnel:**
```bash
# Run this once in a terminal and leave it open
ssh -L 8100:127.0.0.1:8100 canvasxpress@canvasxpress.org -N
```
```javascript
cx.llmServiceURL = "http://localhost:8100/";
```

**Local server:**
```javascript
cx.llmServiceURL = "http://localhost:8100/";
```

---

## Apache proxy configuration

To expose the server through Apache on a production host, add this to your
VirtualHost include directory. The `ProxyPass / !` line **must be last**.

```apache
# /etc/apache2/conf.d/userdata/ssl/2_4/canvasxpress/canvasxpress.org/mcp-proxy.conf
<Location /mcp>
    PassengerEnabled Off
    ProxyPass http://127.0.0.1:8100/mcp
    ProxyPassReverse http://127.0.0.1:8100/mcp
</Location>
<Location /generate>
    PassengerEnabled Off
    ProxyPass http://127.0.0.1:8100/generate
    ProxyPassReverse http://127.0.0.1:8100/generate
</Location>
<Location /modify>
    PassengerEnabled Off
    ProxyPass http://127.0.0.1:8100/modify
    ProxyPassReverse http://127.0.0.1:8100/modify
</Location>
<Location /km>
    PassengerEnabled Off
    ProxyPass http://127.0.0.1:8100/km
    ProxyPassReverse http://127.0.0.1:8100/km
</Location>
<Location /params>
    PassengerEnabled Off
    ProxyPass http://127.0.0.1:8100/params
    ProxyPassReverse http://127.0.0.1:8100/params
</Location>
<Location /axes>
    PassengerEnabled Off
    ProxyPass http://127.0.0.1:8100/axes
    ProxyPassReverse http://127.0.0.1:8100/axes
</Location>
<Location /select>
    PassengerEnabled Off
    ProxyPass http://127.0.0.1:8100/select
    ProxyPassReverse http://127.0.0.1:8100/select
</Location>
<Location /explain-r>
    PassengerEnabled Off
    ProxyPass http://127.0.0.1:8100/explain-r
    ProxyPassReverse http://127.0.0.1:8100/explain-r
</Location>
<Location /explain-ggplot>
    PassengerEnabled Off
    ProxyPass http://127.0.0.1:8100/explain-ggplot
    ProxyPassReverse http://127.0.0.1:8100/explain-ggplot
</Location>
<Location /explain>
    PassengerEnabled Off
    ProxyPass http://127.0.0.1:8100/explain
    ProxyPassReverse http://127.0.0.1:8100/explain
</Location>
<Location /minimal-params>
    PassengerEnabled Off
    ProxyPass http://127.0.0.1:8100/minimal-params
    ProxyPassReverse http://127.0.0.1:8100/minimal-params
</Location>
<Location /map>
    PassengerEnabled Off
    ProxyPass http://127.0.0.1:8100/map
    ProxyPassReverse http://127.0.0.1:8100/map
</Location>
<Location /feedback>
    PassengerEnabled Off
    ProxyPass http://127.0.0.1:8100/feedback
    ProxyPassReverse http://127.0.0.1:8100/feedback
</Location>
<Location /ui>
    PassengerEnabled Off
    ProxyPass http://127.0.0.1:8100/ui
    ProxyPassReverse http://127.0.0.1:8100/ui
</Location>
<Location /favicon.ico>
    PassengerEnabled Off
    ProxyPass http://127.0.0.1:8100/favicon.ico
    ProxyPassReverse http://127.0.0.1:8100/favicon.ico
</Location>
```

> **Note:** `explain-r` and `explain-ggplot` must appear **before** `explain` — Apache
> matches first-wins, so more specific paths go first. `PassengerEnabled Off` and
> `ProxyPass` must be inside the same `<Location>` block; top-level `ProxyPass`
> directives are intercepted by Passenger before they can fire.

To write the file in one shot on the production server (run as root):

```bash
cat > /etc/apache2/conf.d/userdata/ssl/2_4/canvasxpress/canvasxpress.org/mcp-proxy.conf << 'EOF'
<Location /mcp>
    PassengerEnabled Off
    ProxyPass http://127.0.0.1:8100/mcp
    ProxyPassReverse http://127.0.0.1:8100/mcp
</Location>
<Location /generate>
    PassengerEnabled Off
    ProxyPass http://127.0.0.1:8100/generate
    ProxyPassReverse http://127.0.0.1:8100/generate
</Location>
<Location /modify>
    PassengerEnabled Off
    ProxyPass http://127.0.0.1:8100/modify
    ProxyPassReverse http://127.0.0.1:8100/modify
</Location>
<Location /km>
    PassengerEnabled Off
    ProxyPass http://127.0.0.1:8100/km
    ProxyPassReverse http://127.0.0.1:8100/km
</Location>
<Location /params>
    PassengerEnabled Off
    ProxyPass http://127.0.0.1:8100/params
    ProxyPassReverse http://127.0.0.1:8100/params
</Location>
<Location /axes>
    PassengerEnabled Off
    ProxyPass http://127.0.0.1:8100/axes
    ProxyPassReverse http://127.0.0.1:8100/axes
</Location>
<Location /select>
    PassengerEnabled Off
    ProxyPass http://127.0.0.1:8100/select
    ProxyPassReverse http://127.0.0.1:8100/select
</Location>
<Location /explain-r>
    PassengerEnabled Off
    ProxyPass http://127.0.0.1:8100/explain-r
    ProxyPassReverse http://127.0.0.1:8100/explain-r
</Location>
<Location /explain-ggplot>
    PassengerEnabled Off
    ProxyPass http://127.0.0.1:8100/explain-ggplot
    ProxyPassReverse http://127.0.0.1:8100/explain-ggplot
</Location>
<Location /explain>
    PassengerEnabled Off
    ProxyPass http://127.0.0.1:8100/explain
    ProxyPassReverse http://127.0.0.1:8100/explain
</Location>
<Location /minimal-params>
    PassengerEnabled Off
    ProxyPass http://127.0.0.1:8100/minimal-params
    ProxyPassReverse http://127.0.0.1:8100/minimal-params
</Location>
<Location /map>
    PassengerEnabled Off
    ProxyPass http://127.0.0.1:8100/map
    ProxyPassReverse http://127.0.0.1:8100/map
</Location>
<Location /feedback>
    PassengerEnabled Off
    ProxyPass http://127.0.0.1:8100/feedback
    ProxyPassReverse http://127.0.0.1:8100/feedback
</Location>
<Location /ui>
    PassengerEnabled Off
    ProxyPass http://127.0.0.1:8100/ui
    ProxyPassReverse http://127.0.0.1:8100/ui
</Location>
<Location /favicon.ico>
    PassengerEnabled Off
    ProxyPass http://127.0.0.1:8100/favicon.ico
    ProxyPassReverse http://127.0.0.1:8100/favicon.ico
</Location>
EOF

apachectl configtest && service httpd restart
```

---

## MCP tools

The server exposes the following tools over the MCP protocol (used by AI assistants
such as Claude Desktop) and as REST endpoints (used by web pages and scripts directly).

### `generate_canvasxpress_config`

Generate a new CanvasXpress config from a plain English description.

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `description` | string | ✅ | Plain English chart description |
| `headers` | string[] | ❌ | Column names from your dataset |
| `data` | array[][] | ❌ | Full data array — first row = headers. Overrides `headers` |
| `column_types` | object | ❌ | Map of column → type (`string`/`numeric`/`factor`/`date`) |
| `temperature` | float | ❌ | LLM creativity 0–1 (default 0.0) |

**Response:**

```json
{
  "config":         { "graphType": "Violin", "xAxis": ["Expression"], "groupingFactors": ["Treatment"] },
  "valid":          true,
  "warnings":       [],
  "invalid_refs":   {},
  "headers_used":   ["Expression", "Treatment"],
  "types_used":     { "Expression": "numeric", "Treatment": "factor" },
  "removed_params": [],
  "success":        true,
  "datetime":       "Fri, 10 Apr 2026 19:00:00 GMT",
  "request_id":     "3fa85f64-5717-4562-b3fc-2c963f66afa6"
}
```

| Field | Description |
|-------|-------------|
| `config` | The CanvasXpress JSON config — pass to `new CanvasXpress()` |
| `valid` | `true` if all column references exist in the provided headers |
| `warnings` | Column reference or parameter value warnings |
| `removed_params` | Parameter names the LLM invented that were stripped |
| `success` | Same as `valid` — included for CanvasXpress `callbackLLM` compatibility |
| `request_id` | UUID for this call — use with `POST /feedback` to submit a rating |

---

### `modify_canvasxpress_config`

Modify an existing config using a plain English instruction.

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `config` | object | ✅ | The existing CanvasXpress JSON config to modify |
| `instruction` | string | ✅ | Plain English description of the change to apply |
| `headers` | string[] | ❌ | Column names for validating new column references |
| `data` | array[][] | ❌ | Full data array. Overrides `headers` |
| `column_types` | object | ❌ | Map of column → type |
| `temperature` | float | ❌ | LLM creativity 0–1 (default 0.0) |

**Response:**

```json
{
  "config":         { "graphType": "Heatmap", "colorScheme": "Spectral", "title": "My Heatmap", "xAxis": ["Gene"] },
  "prompt":         "change colorScheme to Spectral and add a title",
  "valid":          true,
  "warnings":       [],
  "invalid_refs":   {},
  "headers_used":   ["Gene"],
  "types_used":     { "Gene": "string" },
  "removed_params": [],
  "changes":        { "added": ["title"], "removed": [], "changed": ["colorScheme"] },
  "success":        true,
  "datetime":       "Fri, 10 Apr 2026 19:00:00 GMT",
  "tool":           "modify_canvasxpress_config",
  "request_id":     "7c9e6679-7425-40c3-bba5-0ee65d4ef6a4"
}
```

| Field | Description |
|-------|-------------|
| `config` | The updated CanvasXpress JSON config |
| `prompt` | The instruction echoed back |
| `changes` | Keys added, removed, and changed relative to the input config |
| `request_id` | UUID for this call — use with `POST /feedback` to submit a rating |

**Example instructions:**
```
"add a title My Heatmap"
"change the color scheme to Tableau"
"remove the legend"
"switch to dark theme"
"add groupingFactors for the Treatment column"
"set y-axis min to 0 and max to 100"
"add a horizontal reference line at y = 1.5"
```

---

### `generate_km_config`

Generate, validate, and detect columns for Kaplan-Meier survival plots.
Accepts any combination of description, headers, data, and existing config.

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `description` | string | ❌ | Plain English KM description |
| `headers` | string[] | ❌ | Column names from your dataset |
| `data` | array[][] | ❌ | Full data array — enables column detection |
| `config` | object | ❌ | Existing KM config to validate and fix |
| `temperature` | float | ❌ | LLM creativity 0–1 (default 0.0) |

At least one argument must be provided.

**Response:**

```json
{
  "config": {
    "graphType": "KaplanMeier",
    "xAxis": ["OS_Time"],
    "yAxis": ["OS_Status"],
    "groupingFactors": ["Treatment"],
    "xAxisTitle": "Time (months)",
    "yAxisTitle": "Survival Probability",
    "colorScheme": "Tableau",
    "showLegend": true
  },
  "valid": true,
  "errors": [],
  "warnings": [],
  "suggestions": [],
  "column_detection": {
    "time_col":   "OS_Time",
    "event_col":  "OS_Status",
    "group_cols": ["Treatment"],
    "confidence": "high",
    "notes": []
  },
  "valid":      true,
  "success":    true,
  "datetime":   "Fri, 10 Apr 2026 19:00:00 GMT",
  "tool":       "generate_km_config",
  "request_id": "2b9e6679-9c0b-4ef8-bb6b-6bb9bd380a55"
}
```

---

### `query_canvasxpress_params`

Query the CanvasXpress parameter knowledge base — fetched live from the
[canvasxpress-LLM](https://github.com/neuhausi/canvasxpress-LLM) GitHub repo
with automatic local cache fallback.

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `graph_type` | string | ❌ | Chart type — returns all parameters for this type |
| `param_name` | string | ❌ | Parameter name — returns full definition and valid values |
| `refresh` | boolean | ❌ | Force re-fetch from GitHub (default `false`) |

Pass either, both, or neither (returns full schema summary).

| Env var | Default | Description |
|---------|---------|-------------|
| `CX_SCHEMA_TTL` | `3600` | Schema cache TTL in seconds |
| `CX_SKIP_FETCH` | `0` | Set to `1` to always use bundled schema, no GitHub fetch |

**Response — single parameter (`param_name=colorScheme`):**

```json
{
  "found":        true,
  "param":        "colorScheme",
  "description":  "Color palette applied to the chart.",
  "type":         "string",
  "valid_values": ["Blues", "Reds", "RdBu", "Tableau", "CanvasXpress", "..."],
  "graph_types":  ["Bar", "Heatmap", "Scatter2D", "Violin", "..."],
  "schema_source": "github",
  "tool":         "get_chart_parameters",
  "valid":        true,
  "datetime":     "Fri, 10 Apr 2026 19:00:00 GMT",
  "request_id":   "a0eebc99-9c0b-4ef8-bb6b-6bb9bd380a11"
}
```

**Response — graph type (`graph_type=Heatmap`):**

```json
{
  "graph_type":   "Heatmap",
  "param_count":  42,
  "params": {
    "colorScheme":          { "description": "Color palette.", "type": "string", "valid_values": ["RdBu", "Blues", "..."] },
    "samplesClustered":     { "description": "Cluster columns with dendrogram.", "type": "boolean", "valid_values": [] },
    "variablesClustered":   { "description": "Cluster rows with dendrogram.", "type": "boolean", "valid_values": [] }
  },
  "schema_source": "github",
  "tool":         "get_chart_parameters",
  "valid":        true,
  "datetime":     "Fri, 10 Apr 2026 19:00:00 GMT",
  "request_id":   "b1ddbc00-1d1c-5fg9-cc7c-7cc0ce491b22"
}
```

---

### `get_axes_info`

Return axis assignment rules for a given graph type: which axes are valid,
which are forbidden, and which axis title parameter to use.

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `graph_type` | string | ✅ | CanvasXpress chart type e.g. `Bar`, `Scatter2D`, `BarLine` |

**Response:**

```json
{
  "graph_type":       "Scatter2D",
  "category":         "multi_dim",
  "valid_axes":       ["xAxis", "yAxis"],
  "invalid_axes":     [],
  "axis_title_param": "xAxisTitle / yAxisTitle",
  "notes":            "Scatter2D requires both xAxis (numeric) and yAxis (numeric). Use xAxisTitle and yAxisTitle for axis labels. Never use smpTitle.",
  "schema_snippet":   "xAxis — columns for x-axis; yAxis — columns for y-axis ...",
  "tool":             "suggest_axes",
  "valid":            true,
  "datetime":         "Fri, 10 Apr 2026 19:00:00 GMT",
  "request_id":       "c2eecd11-2e2d-6gh0-dd8d-8dd1df502c33"
}
```

| Field | Description |
|-------|-------------|
| `category` | `single_dim`, `multi_dim`, or `combined` |
| `valid_axes` | Axis keys that apply to this chart type |
| `invalid_axes` | Axis keys that must NOT be used |
| `axis_title_param` | Correct axis title parameter (`smpTitle` vs `xAxisTitle/yAxisTitle`) |

---

### `select_canvasxpress_chart`

The **chart-type recommender** ("wizard" / Tufte mode). Given column metadata and a
plain English intent, it recommends the most appropriate chart type, returns a ranked
list of candidates with rationale, attaches a ready-to-use `minimal_config` to each,
and emits a `generate_hint` you can pipe straight into `generate_canvasxpress_config`.

The recommendation blends three tiers, fastest first:

1. **Deterministic scorer** *(no LLM, no API key)* — a structured first pass over the
   53-chart catalogue, scoring each candidate on (a) column-type counts + row count +
   cardinality, (b) semantic regex over the column *names*, and (c) keyword boosts
   derived from the `intent` string. This always runs and is the default result.
2. **LLM tiebreaker** *(automatic)* — when the top two candidates score within `0.15`
   of each other, the configured LLM disambiguates among the top three and its choice
   is promoted to the top. The one-sentence rationale is returned in the `tiebreak`
   field. If the LLM is unavailable or errors, the deterministic ranking stands.
3. **LLM-first mode** *(opt-in via `llm_first=true`)* — the LLM freely chooses the best
   chart type from the full catalogue (with descriptions) given the columns, intent,
   and row count; its pick becomes `top_recommendation` and the deterministic scorer
   fills the `alternatives`.

No raw data is required — just column names, types, an optional row count, and the
intent string.

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `intent` | string | ✅ | Plain English description of what you want to show |
| `column_types` | object | ✅ | Map of column name → type (`string`/`numeric`/`factor`/`date`) |
| `n_samples` | integer | ❌ | Optional number of rows — used to refine recommendations |
| `llm_first` | boolean | ❌ | When `true`, the LLM chooses the top chart from the full catalogue before the scorer runs (default `false`) |

**Response:**

```json
{
  "intent": "show expression distribution by cell type",
  "column_summary": { "n_factor": 1, "n_numeric": 1, "n_time": 0, "n_bool": 0, "n_text": 0 },
  "top_recommendation": {
    "graphType":       "Violin",
    "score":           0.9,
    "description":     "Kernel density distribution of a numeric variable split by a categorical grouping factor.",
    "clinical_use":    "Gene expression by cell type, biomarker distribution by cohort.",
    "next_step":       "generate_canvasxpress_config with description='Violin chart of Expression grouped by CellType'",
    "scoring_factors": ["1 numeric + 1 factor — ideal for distribution by group"],
    "minimal_config":  { "graphType": "Violin", "xAxis": ["Expression"], "groupingFactors": ["CellType"] }
  },
  "alternatives": [
    {
      "graphType":      "Boxplot",
      "score":          0.78,
      "description":    "Box-and-whisker summary statistics by group.",
      "clinical_use":   "Quick distribution summary when n is sufficient.",
      "scoring_factors": ["1 numeric + 1 factor — suitable for grouped distribution"],
      "minimal_config": { "graphType": "Boxplot", "xAxis": ["Expression"], "groupingFactors": ["CellType"] }
    }
  ],
  "generate_hint": "Violin chart of Expression grouped by CellType — columns: Expression, CellType",
  "tiebreak":       { "used": false, "chosen": null, "reason": "" },
  "valid":          true,
  "warnings":       [],
  "llm_first":      false,
  "type_source":    "explicit",
  "tool":           "select_canvasxpress_chart",
  "datetime":       "Fri, 10 Apr 2026 19:00:00 GMT",
  "request_id":     "d3ffde22-3f3e-7hi1-ee9e-9ee2eg613d44"
}
```

| Field | Description |
|-------|-------------|
| `top_recommendation` | Best chart type with score, rationale, and `minimal_config` |
| `alternatives` | Up to 4 other ranked candidates, each also with `minimal_config` (up to 3 in `llm_first` mode) |
| `generate_hint` | Ready-made description to pass to `generate_canvasxpress_config` |
| `minimal_config` | Minimal axis config ready to use — attach to the `generate` call |
| `tiebreak` | Present when the LLM tiebreaker ran: `{ used, chosen, reason }` |
| `llm_first` | Echoes whether LLM-first mode was requested |
| `type_source` | How column types were resolved: `explicit`, `inferred`, or `merged` |

---

### `explain_config_property`

Return a plain English explanation of any CanvasXpress configuration property.

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `property` | string | ✅ | Config property name e.g. `colorScheme`, `groupingFactors`, `decorations` |

**Response:**

```json
{
  "property":    "groupingFactors",
  "explanation": "**`groupingFactors`** — Array of column names used to group/color data. e.g. ['Treatment', 'CellType']",
  "tool":        "explain_canvasxpress_property",
  "valid":       true,
  "datetime":    "Fri, 10 Apr 2026 19:00:00 GMT",
  "request_id":  "e4ggef33-4g4f-8ij2-ff0f-0ff3fh724e55"
}
```

---

### `explain_canvasxpress_r`

Usage guide for CanvasXpress in R — installation, basic usage, data formats,
Shiny integration, R Markdown, and the ggplot2 bridge.

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `topic` | string | ❌ | Filter by topic: `installation`, `basic`, `shiny`, `rmarkdown`, `data`, `config` |

**Response (no topic — full guide):**

```json
{
  "overview":          "The canvasXpress R package wraps the CanvasXpress JavaScript library as an htmlwidget ...",
  "sections": {
    "installation": { "title": "Installation", "content": "Install from CRAN: install.packages('canvasXpress') ..." },
    "basic":        { "title": "Basic Usage",   "content": "The main function is canvasXpress() ..." },
    "data":         { "title": "Data Format",   "content": "CanvasXpress in R expects data in one of two orientations ..." },
    "config":       { "title": "Configuration Parameters", "content": "All CanvasXpress JSON config parameters map directly ..." },
    "shiny":        { "title": "Using CanvasXpress in Shiny", "content": "CanvasXpress integrates with Shiny via canvasXpressOutput() ..." },
    "rmarkdown":    { "title": "Using CanvasXpress in R Markdown / Quarto", "content": "..." }
  },
  "available_topics": ["installation", "basic", "data", "config", "shiny", "rmarkdown"],
  "tool":             "explain_canvasxpress_r",
  "valid":            true,
  "datetime":         "Fri, 10 Apr 2026 19:00:00 GMT",
  "request_id":       "f5hhfg44-5h5g-9jk3-gg1g-1gg4gi835f66"
}
```

**Response (with `topic=shiny`):**

```json
{
  "topic":   "shiny",
  "section": { "title": "Using CanvasXpress in Shiny", "content": "CanvasXpress integrates with Shiny via canvasXpressOutput() ..." },
  "available_topics": ["installation", "basic", "data", "config", "shiny", "rmarkdown"],
  "tool":             "explain_canvasxpress_r",
  "valid":            true,
  "datetime":         "Fri, 10 Apr 2026 19:00:00 GMT",
  "request_id":       "g6iigh55-6i6h-0kl4-hh2h-2hh5hj946g77"
}
```

---

### `explain_canvasxpress_ggplot`

Usage guide for the CanvasXpress ggplot2 bridge — convert any ggplot2 object to an
interactive CanvasXpress widget with a single function call.

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `topic` | string | ❌ | Filter by topic: `installation`, `geoms`, `example` |

**Response (no topic — full guide):**

```json
{
  "sections": {
    "overview":      { "title": "Overview",      "content": "canvasXpress() accepts a ggplot2 object directly ..." },
    "installation":  { "title": "Installation",  "content": "install.packages('canvasXpress') ..." },
    "geoms":         { "title": "Supported Geoms", "content": "geom_point → Scatter2D, geom_bar → Bar ..." },
    "example":       { "title": "Full Example",   "content": "library(ggplot2); library(canvasXpress) ..." }
  },
  "available_topics": ["overview", "installation", "geoms", "example"],
  "tool":             "explain_ggplot_to_canvasxpress",
  "valid":            true,
  "datetime":         "Fri, 10 Apr 2026 19:00:00 GMT",
  "request_id":       "h7jjih66-7j7i-1lm5-ii3i-3ii6ik057h88"
}
```

---

### `get_minimal_parameters`

Return the minimal set of required parameters for a specific chart type.

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `graph_type` | string | ✅ | CanvasXpress chart type e.g. `Scatter2D`, `Heatmap` |

**Response:**

```json
{
  "graphType":           "KaplanMeier",
  "required_parameters": ["graphType", "xAxis", "yAxis"],
  "tool":                "get_minimal_params",
  "valid":               true,
  "datetime":            "Fri, 10 Apr 2026 19:00:00 GMT",
  "request_id":          "i8kkji77-8k8j-2mn6-jj4j-4jj7jl168i99"
}
```

| Field | Description |
|-------|-------------|
| `graphType` | The requested chart type |
| `required_parameters` | Minimum set of parameters that must be populated for a valid config |

---

### `create_map_config`

Generate a CanvasXpress map (choropleth, pie, or marker) config without an LLM call.
Supports world maps, continent maps, country maps, US state/county maps, and custom
topoJSON maps. Optionally overlays pie charts per region, proportional sizing, and
geocoded marker pins.

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `map_id` | string | ✅ | Map identifier — see table below |
| `data` | array[][] | ❌ | CSV-style data; first row = headers, first column = geographic IDs |
| `title` | string | ❌ | Chart title |
| `color_scheme` | string | ❌ | Color palette e.g. `Blues`, `RdBu`, `YlOrRd` |
| `color_by` | string | ❌ | Column name to color regions/symbols by |
| `size_by` | string | ❌ | Column name whose values scale symbol/pie size per region |
| `decorations` | object | ❌ | Decoration overlays — currently supports `pie` (see below) |
| `topo_json` | string | ❌ | URL to a custom topoJSON file |
| `legend_order` | object | ❌ | Map of column → ordered list of values for legend display |
| `markers` | array | ❌ | List of marker pin dicts (see below) |

**`map_id` values:**

| `map_id` | Coverage | First-column ID format |
|----------|----------|-----------------------|
| `World` | All countries | ISO 3-letter codes (`USA`, `FRA`, `CHN`, …) |
| `WorldContinents` | 6 continents | Continent names (`Africa`, `Asia`, `Europe`, …) |
| `Africa`, `Asia`, `Europe`, `NorthAmerica`, `SouthAmerica`, `Oceania` | One continent | ISO 3-letter codes |
| `USAStates` | 50 US states + DC | 2-letter codes (`CA`, `TX`, `NY`, …) |
| `USACounties` | US counties | 5-digit FIPS codes (`06037`, `48113`, …) |
| `albersStatesPie` | US states, Albers projection | 2-letter codes — use when Albers projection is explicitly requested |
| ISO 3-letter code (`CAN`, `GBR`, `AUS`, …) | Country sub-regions | Feature property values — set `mapPropertyId` in config |
| 2-letter US state code (`CA`, `TX`, `NY`, …) | State counties | County names or FIPS codes |

**Pie overlay (`decorations.pie`):**

```json
"decorations": {
  "pie": {
    "smps":   ["Democrat", "Republican", "Libertarian", "Other"],
    "colors": ["blue", "red", "yellow", "green"],
    "size":   2.5
  }
}
```

`decorations.pie.size` is a float multiplier that controls how large each pie is drawn.
`size_by` (top-level) is a column name that scales each pie proportionally to its data value.
These two work independently — both can be used together.

**Marker pins (`markers` list):**

Each marker dict supports three ways to specify location:

| Key | Description |
|-----|-------------|
| `lat` + `lng` | Explicit decimal-degree coordinates |
| `zip` | US ZIP code — resolved via free zippopotam.us API |
| `location` | Any city, address, or landmark — geocoded via Nominatim (OpenStreetMap) |

Optional fields: `label` (string), `color` (default `"red"`), `shape` (`teardrop`\|`circle`\|`star`\|`square`, default `teardrop`), `size` (int 1–10, default `4`).

**Response:**

```json
{
  "config": {
    "graphType":    "Map",
    "mapId":        "USAStates",
    "colorBy":      "Winner",
    "sizeBy":       "Total",
    "decorations":  { "pie": [{"smps": ["Democrat","Republican","Libertarian","Other"], "colors": ["blue","red","yellow","green"], "size": 2.5}] },
    "legendOrder":  { "Winner": ["Republican","Democrat"] },
    "title":        "2000 Presidential Elections"
  },
  "valid":        true,
  "warnings":     [],
  "map_id":       "USAStates",
  "headers_used": ["Id","Total","Democrat","Republican","Libertarian","Other","State","Winner"],
  "tool":         "create_map_config",
  "datetime":     "Fri, 10 Apr 2026 19:00:00 GMT",
  "request_id":   "j9lllk88-9l9k-3no7-kk5k-5kk8km279j00"
}
```

**Example prompts for `generate_canvasxpress_config`:**
```
"Show GDP by country on a world map using Blues color scheme"
"US states map colored by unemployment rate"
"Create a map of France showing population by region"
"Pie map of the 2000 US Presidential Election — Democrat, Republican, Libertarian, Other slices per state"
"World map with markers at Paris, Tokyo, New York, and Sydney"
"USA states map with markers at ZIP codes 10001, 90210, and 60601"
```

---

## Call logging & feedback

Every tool call is automatically logged to `data/call_log.db` (a separate SQLite
database from the vector index). Each response includes a `request_id` UUID that
can be used to submit thumbs-up/down feedback.

### Submit feedback

```bash
curl -X POST http://localhost:8100/feedback \
  -H "Content-Type: application/json" \
  -d '{"request_id": "<uuid from response>", "rating": 1, "comment": "Perfect"}'
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `request_id` | string | ✅ | UUID from the tool response |
| `rating` | integer | ✅ | `1` = thumbs up, `-1` = thumbs down |
| `comment` | string | ❌ | Optional free-text note |

### Export call log

```bash
# All calls (latest 500)
curl "http://localhost:8100/feedback/export"

# Only rated calls for one tool
curl "http://localhost:8100/feedback/export?tool=select_canvasxpress_chart&rated_only=true"
```

| Parameter | Description |
|-----------|-------------|
| `tool` | Filter by tool name (optional) |
| `rated_only` | `true` — return only rows that have a rating |
| `limit` | Max rows to return (default `500`) |

### Purge call log (admin only)

Requires the `X-Admin-Key` header. See [ADMIN_KEY](#environment-variables) below.

```bash
# Delete all rows
curl -X POST http://localhost:8100/feedback/purge \
  -H "X-Admin-Key: your-secret-key"

# Delete only rated rows for one tool
curl -X POST http://localhost:8100/feedback/purge \
  -H "X-Admin-Key: your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"tool": "generate_canvasxpress_config", "rated_only": true}'
```

| Body field | Description |
|------------|-------------|
| `tool` | Delete only rows for this tool name (optional) |
| `rated_only` | `true` — delete only rows that have a rating (optional) |

Omitting both deletes **all** rows. Returns `{"success": true, "deleted": N}`.

> **Security** — without a correct `X-Admin-Key` header the endpoint returns
> `403 Forbidden`. The comparison uses `hmac.compare_digest` to prevent
> timing-based attacks. Set `ADMIN_KEY` in `.env` for a persistent key; if not
> set, a random UUID is generated per restart and printed to the server log.

### Manage the call log locally (CLI)

`manage_call_log.py` is a standalone CLI script for managing `data/call_log.db`
directly on the server — no HTTP key required.

> **Note** — `data/call_log.db` is in `.gitignore` and is never overwritten by
> `git pull`. The server uses `CREATE TABLE IF NOT EXISTS`, so the file is only
> ever opened and appended to, never recreated on restart.

```bash
# Summary: row counts by tool, rated/unrated, 👍/👎
python manage_call_log.py stats

# Export everything to a timestamped JSON file
python manage_call_log.py export --out backup_$(date +%Y%m%d).json

# Export only rated calls as CSV
python manage_call_log.py export --rated-only --format csv --out rated.csv

# Export only calls for one tool
python manage_call_log.py export --tool generate_canvasxpress_config

# Purge only rated calls (keeps unrated history)
python manage_call_log.py purge --rated-only

# Purge everything without a prompt (e.g. in a cron job after export)
python manage_call_log.py purge --yes

# Use a different database path
python manage_call_log.py --db /path/to/other/call_log.db stats
```

**`stats`** — prints a per-tool breakdown table with total, rated, 👍, 👎 counts and first-call timestamp.

**`export`** options:

| Option | Description |
|--------|-------------|
| `--tool TOOL` | Filter by tool name (partial match) |
| `--rated-only` | Only rows that have a rating |
| `--limit N` | Max rows to export (default: all) |
| `--format json\|csv` | Output format (default: `json`) |
| `--out FILE` | Write to file instead of stdout |

**`purge`** options:

| Option | Description |
|--------|-------------|
| `--tool TOOL` | Delete only rows for this tool (partial match) |
| `--rated-only` | Delete only rows that have a rating |
| `--yes` | Skip the confirmation prompt |

---

## LLM providers

### Anthropic (default)

```bash
export LLM_PROVIDER=anthropic          # optional — this is the default
export ANTHROPIC_API_KEY="sk-ant-..."
export LLM_MODEL=claude-sonnet-4-20250514  # optional
python src/server.py
```

### Amazon Bedrock

```bash
pip install boto3
export LLM_PROVIDER=bedrock
export AWS_REGION=us-east-1
# Uses your existing AWS credentials (IAM role, SSO profile, or explicit keys)
python src/server.py
```

### Ollama (local, no API key)

```bash
ollama serve
ollama pull llama3.2
export LLM_PROVIDER=ollama
export LLM_MODEL=llama3.2
python src/server.py
```

### OpenAI / corporate gateway

```bash
pip install openai
export LLM_PROVIDER=openai
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="https://api.your-company.com/openai/v1"
export LLM_MODEL=gpt-4o
python src/server.py
```

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `anthropic` | LLM backend: `anthropic`, `bedrock`, `ollama`, `openai` |
| `LLM_MODEL` | provider default | Model name / ID |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `AWS_REGION` | `us-east-1` | AWS region for Bedrock |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OPENAI_API_KEY` | — | OpenAI / gateway API key |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible endpoint |
| `MCP_HOST` | `0.0.0.0` | Server bind host |
| `MCP_PORT` | `8100` | Server port |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |
| `ADMIN_KEY` | auto-generated | Secret key required for `POST /feedback/purge`. Auto-generated and logged if not set |
| `CX_DEBUG` | `0` | Set to `1` for full debug trace |
| `CX_SCHEMA_TTL` | `3600` | Schema cache TTL in seconds |
| `CX_SKIP_FETCH` | `0` | Set to `1` to skip GitHub schema fetch |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model for vector search |

All variables can also be set in a `.env` file in the project root.

---

## Troubleshooting

**Updating the production server from the repo**

Run these steps every time you pull a new version:

```bash
cd canvasxpress-mcp/
./server.sh stop
git fetch origin
git reset --hard origin/main
git apply --whitespace=nowarn canvasxpress-ctypes-fix.patch
source .venv/bin/activate
pip install -r requirements.txt
./server.sh start
```

> `git reset --hard` discards any local changes and makes the working tree match
> the remote exactly. If you have local config files (`.env`, `data/call_log.db`)
> they are untracked and will not be touched.
>
> If `src/server.py` is missing after the patch step (patch did not restore it),
> recover it with:
> ```bash
> git restore src/server.py
> ```

---

**`No module named 'dotenv'`**
```bash
pip install python-dotenv
```

**`No module named 'starlette'`**
```bash
pip install starlette fastmcp
```

**Port 8100 already in use**
```bash
lsof -ti :8100 | xargs kill -9   # macOS/Linux
```

**Homepage shows "It works! Python 3.12"**
Remove the Passenger configuration from `/home/canvasxpress/public_html/.htaccess`.
Delete the lines between `CLOUDLINUX PASSENGER CONFIGURATION BEGIN` and
`CLOUDLINUX PASSENGER CONFIGURATION END`.

**404 from the browser but curl works**
Check that `llmServiceURL` does not include the port number in the path.
Correct: `"https://www.canvasxpress.org/"` — incorrect: `"https://www.canvasxpress.org:8100/"`.

**500 error on invalid descriptions**
Upgrade to the latest `server.py` — graceful error handling was added so invalid
prompts return a 200 with `valid: false` and a helpful message instead of a 500.

**`removed_params` is non-empty**
The LLM generated parameter names not in the CanvasXpress schema. They were
automatically stripped. The config is still valid — refine the description if needed.

**New tool added — Apache returns 404 for the new endpoint**
Every time a new tool (and its REST endpoint) is added to `server.py`, the Apache
proxy config must be updated **as root**. Add a new `<Location>` block for the
new endpoint, keeping `PassengerEnabled Off` and `ProxyPass` together inside it:

```apache
<Location /new-endpoint>
    PassengerEnabled Off
    ProxyPass http://127.0.0.1:8100/new-endpoint
    ProxyPassReverse http://127.0.0.1:8100/new-endpoint
</Location>
```

Then reload Apache:

```bash
apachectl configtest && service httpd restart
```

> See the [Apache proxy configuration](#apache-proxy-configuration) section above
> for the complete current config to overwrite the file from scratch if needed.

## MCP Apps support

MCP Apps ([SEP-1865](https://raw.githubusercontent.com/modelcontextprotocol/ext-apps/main/specification/2026-01-26/apps.mdx)) is an MCP extension that lets hosts embed a server-supplied iframe directly inside the conversation UI, enabling inline interactive chart rendering.

### Which tools declare a UI resource

Four chart-producing tools carry `_meta.ui.resourceUri = "ui://canvasxpress/chart"`:

| Tool | Purpose |
|------|---------|
| `generate_canvasxpress_config` | Generate a new chart config from plain English |
| `modify_canvasxpress_config` | Modify an existing chart config |
| `generate_km_config` | Generate a Kaplan-Meier survival plot config |
| `create_map_config` | Generate a choropleth / map config |

When a supporting host calls any of these tools it will load the `ui://canvasxpress/chart` resource, render it in a sandboxed iframe, and post the tool result to the iframe via the `ui/notifications/tool-result` postMessage notification. The iframe parses the JSON config and calls `new CanvasXpress()` to display the chart inline.

### Known supporting hosts

- Claude Desktop (Anthropic)
- ChatGPT / OpenAI Apps SDK
- VS Code (with MCP Apps extension)
- Goose (Block)

### Testing locally

```bash
# 1. Import check
.venv/bin/python -c "import sys; sys.path.insert(0, 'src'); import server; print('OK')"

# 2. Unit tests (offline, no LLM calls, < 10 s)
.venv/bin/python -m pytest tests/ -x -q

# 3. Verify resource content in-process
.venv/bin/python -c "
import sys, asyncio; sys.path.insert(0, 'src')
import server
async def main():
    r = await server.mcp.read_resource('ui://canvasxpress/chart')
    print(r.contents[0].mime_type, len(r.contents[0].content), 'bytes')
asyncio.run(main())
"
```

All existing REST endpoints (`/generate`, `/modify`, `/select`, `/km`, etc.) and any MCP clients that do not support MCP Apps are completely unaffected — the `_meta.ui.resourceUri` annotation is silently ignored by non-supporting hosts.

### CDN versioning note

The HTML view loads CanvasXpress from `https://www.canvasxpress.org/dist/canvasXpress.min.js`. No versioned CDN URL is offered by CanvasXpress upstream (verified 2026-05-28). Users who require reproducibility should self-host a pinned copy of `canvasXpress.min.js` and update the `<script src>` in `src/ui/cx_chart_view.html`.
