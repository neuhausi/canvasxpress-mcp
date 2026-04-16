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
│   └── cx_selector.py      — Chart type selection skill (deterministic, no LLM)
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
python3.11 -m venv .venv
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
| `GET /select` | Recommend a chart type | `intent`, `column_types` |
| `GET /explain` | Explain a config property | `property` |
| `GET /explain-r` | CanvasXpress in R | none (optional: `topic`) |
| `GET /explain-ggplot` | CanvasXpress ggplot2 bridge | none (optional: `topic`) |
| `GET /minimal-params` | Minimal required parameters | `graph_type` |
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

# Recommend a chart type
curl -s "http://localhost:8100/select?\
intent=show+expression+distribution+by+cell+type\
&column_types=Expression=numeric,CellType=factor"

# Explain a config property
curl -s "http://localhost:8100/explain?property=groupingFactors"

# Minimal required parameters
curl -s "http://localhost:8100/minimal-params?graph_type=KaplanMeier"
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

# Disable Passenger for all MCP proxy paths
<Location /generate>
    PassengerEnabled Off
</Location>
<Location /modify>
    PassengerEnabled Off
</Location>
<Location /km>
    PassengerEnabled Off
</Location>
<Location /params>
    PassengerEnabled Off
</Location>
<Location /axes>
    PassengerEnabled Off
</Location>
<Location /select>
    PassengerEnabled Off
</Location>
<Location /explain>
    PassengerEnabled Off
</Location>
<Location /explain-r>
    PassengerEnabled Off
</Location>
<Location /explain-ggplot>
    PassengerEnabled Off
</Location>
<Location /minimal-params>
    PassengerEnabled Off
</Location>
<Location /ui>
    PassengerEnabled Off
</Location>

# Proxy MCP paths to the Python server on port 8100
ProxyPass        /generate        http://127.0.0.1:8100/generate
ProxyPassReverse /generate        http://127.0.0.1:8100/generate
ProxyPass        /modify          http://127.0.0.1:8100/modify
ProxyPassReverse /modify          http://127.0.0.1:8100/modify
ProxyPass        /km              http://127.0.0.1:8100/km
ProxyPassReverse /km              http://127.0.0.1:8100/km
ProxyPass        /params          http://127.0.0.1:8100/params
ProxyPassReverse /params          http://127.0.0.1:8100/params
ProxyPass        /axes            http://127.0.0.1:8100/axes
ProxyPassReverse /axes            http://127.0.0.1:8100/axes
ProxyPass        /select          http://127.0.0.1:8100/select
ProxyPassReverse /select          http://127.0.0.1:8100/select
ProxyPass        /explain         http://127.0.0.1:8100/explain
ProxyPassReverse /explain         http://127.0.0.1:8100/explain
ProxyPass        /explain-r       http://127.0.0.1:8100/explain-r
ProxyPassReverse /explain-r       http://127.0.0.1:8100/explain-r
ProxyPass        /explain-ggplot  http://127.0.0.1:8100/explain-ggplot
ProxyPassReverse /explain-ggplot  http://127.0.0.1:8100/explain-ggplot
ProxyPass        /minimal-params  http://127.0.0.1:8100/minimal-params
ProxyPassReverse /minimal-params  http://127.0.0.1:8100/minimal-params
ProxyPass        /ui              http://127.0.0.1:8100/ui
ProxyPassReverse /ui              http://127.0.0.1:8100/ui

# Block the root from being proxied — serve the website normally
# MUST be the last ProxyPass rule
ProxyPass        /  !
```

After editing:

```bash
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
  "datetime":       "Fri, 10 Apr 2026 19:00:00 GMT"
}
```

| Field | Description |
|-------|-------------|
| `config` | The CanvasXpress JSON config — pass to `new CanvasXpress()` |
| `valid` | `true` if all column references exist in the provided headers |
| `warnings` | Column reference or parameter value warnings |
| `removed_params` | Parameter names the LLM invented that were stripped |
| `success` | Same as `valid` — included for CanvasXpress `callbackLLM` compatibility |

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

Response includes all `generate_canvasxpress_config` fields plus:

```json
{
  "changes": {
    "added":   ["title"],
    "removed": [],
    "changed": ["colorScheme"]
  }
}
```

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
  }
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

---

### `get_axes_info`

Return axis assignment rules for a given graph type: which axes are valid,
which are forbidden, and which axis title parameter to use.

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `graph_type` | string | ✅ | CanvasXpress chart type e.g. `Bar`, `Scatter2D`, `BarLine` |

---

### `select_canvasxpress_chart`

Recommend the most appropriate chart type given column metadata and a plain
English intent. Deterministic — no LLM call. Returns a ranked list of candidates
with rationale and a ready-made description hint to pass to `generate_canvasxpress_config`.

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `intent` | string | ✅ | Plain English description of what you want to show |
| `column_types` | object | ✅ | Map of column name → type (`string`/`numeric`/`factor`/`date`) |
| `n_samples` | integer | ❌ | Optional number of rows — used to refine recommendations |

---

### `explain_config_property`

Return a plain English explanation of any CanvasXpress configuration property.

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `property` | string | ✅ | Config property name e.g. `colorScheme`, `groupingFactors`, `decorations` |

---

### `explain_canvasxpress_r`

Usage guide for CanvasXpress in R — installation, basic usage, data formats,
Shiny integration, R Markdown, and the ggplot2 bridge.

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `topic` | string | ❌ | Filter by topic: `installation`, `basic`, `shiny`, `rmarkdown`, `data`, `config` |

---

### `explain_canvasxpress_ggplot`

Usage guide for the CanvasXpress ggplot2 bridge — convert any ggplot2 object to an
interactive CanvasXpress widget with a single function call.

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `topic` | string | ❌ | Filter by topic: `installation`, `geoms`, `example` |

---

### `get_minimal_parameters`

Return the minimal set of required parameters for a specific chart type.

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `graph_type` | string | ✅ | CanvasXpress chart type e.g. `Scatter2D`, `Heatmap` |

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
| `CX_DEBUG` | `0` | Set to `1` for full debug trace |
| `CX_SCHEMA_TTL` | `3600` | Schema cache TTL in seconds |
| `CX_SKIP_FETCH` | `0` | Set to `1` to skip GitHub schema fetch |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model for vector search |

All variables can also be set in a `.env` file in the project root.

---

## Troubleshooting

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
