# CanvasXpress MCP — Usage Guide

This guide covers three ways to use the CanvasXpress LLM service to generate
chart configurations from plain English descriptions.

| Mode | URL | When to use |
|------|-----|-------------|
| **Production service** | `https://www.canvasxpress.org/` | Simplest — no server to run |
| **Client → Production** | `http://localhost:8100/` via SSH tunnel | Local development against production LLM |
| **Client → Local server** | `http://localhost:8100/` | Full local stack, own API keys |

---

## Mode 1 — Production service at canvasxpress.org

The MCP server runs on `canvasxpress.org` and is exposed via Apache at
`https://www.canvasxpress.org/generate`. No installation required.

### CanvasXpress integration

Set `llmServiceURL` to the production root before initialising your chart:

```javascript
var cx = new CanvasXpress({
    renderTo: "myChart",
    data: { ... },
    config: { graphType: "Bar" },
    llmServiceURL: "https://www.canvasxpress.org/"
});
```

CanvasXpress will append `generate` to form the full endpoint URL:
```
https://www.canvasxpress.org/generate?callback=CanvasXpress.callbackLLM&...
```

### Direct REST call (fetch)

```javascript
const params = new URLSearchParams({
    description: "Violin plot of expression grouped by treatment, Tableau colors",
    headers:     "Sample, Expression, Treatment",
    column_types:"Expression=numeric, Treatment=factor",
    temperature: "0"
});

const response = await fetch(`https://www.canvasxpress.org/generate?${params}`);
const result   = await response.json();
console.log(result.config);   // ready to pass to new CanvasXpress()
```

### Direct REST call (curl)

```bash
curl -s "https://www.canvasxpress.org/generate?\
description=Clustered+heatmap+with+RdBu+colors\
&headers=Gene,Control1,Control2,Drug1\
&column_types=Gene=string,Control1=numeric,Control2=numeric,Drug1=numeric"
```

### Modify an existing config

```bash
curl -s "https://www.canvasxpress.org/modify?\
config=%7B%22graphType%22%3A%22Heatmap%22%7D\
&instruction=change+colorScheme+to+Spectral+and+add+a+title"
```

### Response format

Both `/generate` and `/modify` return the same structure:

```json
{
  "config": {
    "graphType": "Violin",
    "xAxis": ["Expression"],
    "groupingFactors": ["Treatment"],
    "colorScheme": "Tableau"
  },
  "valid":          true,
  "warnings":       [],
  "invalid_refs":   {},
  "headers_used":   ["Sample", "Expression", "Treatment"],
  "types_used":     {"Expression": "numeric", "Treatment": "factor"},
  "removed_params": [],
  "success":        true,
  "prompt":         "Violin plot of expression grouped by treatment, Tableau colors",
  "datetime":       "Fri, 10 Apr 2026 19:00:00 GMT"
}
```

| Field | Description |
|-------|-------------|
| `config` | The CanvasXpress JSON config — pass this to `new CanvasXpress()` |
| `valid` | `true` if all column references exist in the provided headers |
| `warnings` | Column reference or parameter value warnings |
| `removed_params` | Parameter names the LLM invented that were stripped |
| `success` | Same as `valid` — included for CanvasXpress `callbackLLM` compatibility |

---

## Mode 2 — Local client pointing to the production service

Use this during local development when you want to test your page locally but
use the production LLM without running a local server or managing API keys.

### Step 1 — Open an SSH tunnel

Run this once in a terminal and leave it open:

```bash
ssh -L 8100:127.0.0.1:8100 canvasxpress@canvasxpress.org -N
```

With a named SSH config entry (add to `~/.ssh/config`):

```
Host cxmcp
    HostName canvasxpress.org
    User canvasxpress
    LocalForward 8100 127.0.0.1:8100
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

Then just run:

```bash
ssh -N cxmcp
```

### Step 2 — Test the tunnel

```bash
curl -s "http://localhost:8100/generate?description=test+heatmap" | python3 -m json.tool
```

You should see a JSON response. If you do, the tunnel is working.

### Step 3 — Point your local page at the tunnel

```javascript
var cx = new CanvasXpress({
    renderTo: "myChart",
    data: { ... },
    config: { graphType: "Bar" },
    llmServiceURL: "http://localhost:8100/"
});
```

### Closing the tunnel

```bash
pkill -f "ssh.*8100"
```

### Notes

- The tunnel forwards your local port 8100 to the server's internal port 8100 over SSH.
  No firewall changes are needed — port 8100 is not publicly exposed.
- The production LLM and API key are used — you are not charged locally.
- This is the recommended mode for local development.

---

## Mode 3 — Local client pointing to a local MCP server

Run the full stack locally with your own LLM API key. Best for development,
testing prompt changes, or working offline with Ollama.

### Step 1 — Clone and install

```bash
git clone https://github.com/neuhausi/canvasxpress-mcp.git
cd canvasxpress-mcp
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2 — Build the vector index

```bash
python build_index.py
```

Re-run whenever `data/few_shot_examples.json` changes.

### Step 3 — Configure your LLM provider

**Anthropic (default):**

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

**Amazon Bedrock:**

```bash
pip install boto3
export LLM_PROVIDER=bedrock
export AWS_REGION=us-east-1
```

**Ollama (fully local, no API key):**

```bash
# Install Ollama from https://ollama.com, then:
ollama pull llama3
export LLM_PROVIDER=ollama
export LLM_MODEL=llama3
```

**OpenAI or compatible:**

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY="sk-..."
```

### Step 4 — Start the server

```bash
python src/server.py
```

Server starts at `http://localhost:8100`. Verify:

```bash
curl -s "http://localhost:8100/generate?description=bar+chart" | python3 -m json.tool
```

### Step 5 — Point your local page at the local server

```javascript
var cx = new CanvasXpress({
    renderTo: "myChart",
    data: { ... },
    config: { graphType: "Bar" },
    llmServiceURL: "http://localhost:8100/"
});
```

### Debug mode

Print the full reasoning trace for every request:

```bash
CX_DEBUG=1 python src/server.py
```

---

## API reference

### GET /generate

Generate a new CanvasXpress config from a plain English description.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `description` | ✅ | Plain English chart description. Alias: `prompt`, `q` |
| `headers` | ❌ | Comma-separated column names: `Gene, Expression, Treatment` |
| `column_types` | ❌ | Column types: `Gene=string, Expression=numeric, Treatment=factor` |
| `data` | ❌ | JSON array of arrays (first row = headers): `[["Gene","Expr"],["BRCA1",1.2]]` |
| `temperature` | ❌ | LLM creativity 0.0–1.0 (default 0.0 = deterministic) |
| `callback` | ❌ | JSONP callback name (set automatically by CanvasXpress) |
| `target` | ❌ | CanvasXpress chart target ID (set automatically by CanvasXpress) |
| `client_id` | ❌ | CanvasXpress client ID (set automatically by CanvasXpress) |

**Example:**

```
GET /generate?description=Violin+plot+of+expression+by+cell+type
              &headers=CellID,Expression,CellType
              &column_types=CellID=string,Expression=numeric,CellType=factor
```

---

### GET /modify

Modify an existing config using a plain English instruction.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `config` | ✅ | Existing CanvasXpress JSON config (URL-encoded) |
| `instruction` | ✅ | Plain English modification: `change colorScheme to Tableau` |
| `headers` | ❌ | Column names for validating any new column references |
| `column_types` | ❌ | Column types |
| `temperature` | ❌ | 0.0–1.0 (default 0.0) |

**Example:**

```
GET /modify?config={"graphType":"Heatmap","xAxis":["Gene"]}
            &instruction=add+a+title+Expression+Heatmap+and+switch+to+dark+theme
```

---

### GET /ui

Browser-based form for testing generate and modify without writing code.

```
https://www.canvasxpress.org/ui
http://localhost:8100/ui       (local server)
```

---

## Common descriptions

```
"Clustered heatmap with RdBu colors and dendrograms on both axes"
"Volcano plot with log2 fold change on x-axis and -log10 p-value on y-axis"
"Violin plot of gene expression grouped by cell type, Tableau colors"
"PCA scatter plot of PC1 vs PC2 colored by Treatment with regression ellipses"
"Kaplan-Meier survival curve for two treatment groups"
"Stacked percent bar chart of market share by year and company"
"Bar chart of expression values filtered to Control group only"
"Horizontal bar chart sorted descending by value"
"Scatter plot with vertical threshold lines at log2FC ±2 and significance line at 1.3"
```

## Troubleshooting

**"It works! Python 3.12" appears on my website**
Remove the Passenger configuration from `/home/canvasxpress/public_html/.htaccess`.
Look for and delete the lines between `CLOUDLINUX PASSENGER CONFIGURATION BEGIN`
and `CLOUDLINUX PASSENGER CONFIGURATION END`.

**`curl` to port 8100 gives "Connection refused"**
Port 8100 is not publicly exposed — this is intentional. Use the SSH tunnel
(Mode 2) for local development.

**JSONP callback not firing in the browser**
Open DevTools → Network and check that the `/generate` response starts with
`CanvasXpress.callbackLLM(` and has `Content-Type: application/javascript`.
If it returns plain JSON, the `callback=` parameter is not being sent —
check that `llmServiceURL` ends with a trailing slash.

**`removed_params` is non-empty**
The LLM generated parameter names that don't exist in the CanvasXpress schema.
They were automatically stripped. The config is still valid — the removed names
are shown so you can refine the description if needed.

**Column validation warnings**
The generated config references column names that weren't found in your `headers`.
Either the LLM inferred column names from the description (which may not match
your actual data), or the headers weren't passed in. Add `headers=` to your
request for accurate validation.
