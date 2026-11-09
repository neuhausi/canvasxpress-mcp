# Changelog

All notable changes to canvasxpress-mcp. Releases are tagged `vX.Y.Z`; pin a tag rather than a commit.

## [Unreleased]

### Added
- **Public-deployment controls: bearer-token auth + cost kill-switch (for an internet-facing endpoint that
  ChatGPT Apps / Gemini Connected Apps can connect to).** `CX_PUBLIC=1` is a fail-closed public profile:
  bearer auth is required and the server refuses to start without a token, so a public endpoint can never
  accidentally run open. On a non-public instance it is additive and opt-in — auth enforces only when
  `CX_AUTH_TOKENS`/`CX_AUTH_TOKENS_FILE` is set, otherwise the endpoint behaves exactly as before, so local,
  SSH-tunnel and existing public JSONP installs (and the Claude connector) are unchanged. When enforced, every
  route is gated except the exempt prefixes (`CX_AUTH_EXEMPT_PATHS`, default `/health`) and CORS preflight, with
  an optional per-token rate limit (`CX_RATE_LIMIT_PER_MIN` → `429`). Cost controls pause generation when spend
  runs high: a per-UTC-day ceiling (`CX_DAILY_COST_LIMIT_USD`) and a manual kill switch (`CX_DISABLE_GENERATION`),
  both surfaced through a single generation chokepoint. All LLM generation flows through the guard; the daily
  tally reuses the existing per-request cost accounting. 11 new offline tests (`tests/test_public_auth.py`).
- **One rule for getting data in: "drag the file here".** `suggest_canvasxpress_dashboard` called with no data
  returns `{drop: true}` and the dashboard view shows a drop zone; the person drags the file (or 2 to 6 related
  files) onto the panel, the view sends it to the server in 60 KB pieces through the host (`tools/call`
  `upload_canvasxpress_file`, the one channel a host lets the view use: Claude Desktop blocks a fetch from the
  iframe), gets a link back, calls the tool again with it (`url`, or the new `urls` for several files) and draws
  the result. No rows pass through
  the model, so a file on the computer, a file attached to the chat and data made by an analysis all take seconds.
  The tool text is now that one rule: `dataset` or `url` when there is one, otherwise call with nothing and tell
  the user to drag the file onto the panel.
- **Behind-the-scenes upload from a local file.** `tools/local-upload/cx_upload.py` is a stdlib-only local MCP
  server (stdio; also a Desktop Extension via its `manifest.json`) with one tool, `upload_csv(path)`: it reads the
  file on the user's machine, POSTs it to `/mcp/upload` and returns the link; the remote tool text says to use it
  first. The rows never pass through the model, and no client sandbox policy is involved.
- **A link instead of the rows.** `suggest_canvasxpress_dashboard(url=...)`: the server fetches the CSV/TSV itself
  (GitHub blob, Dropbox, Google Drive and Google Sheets share links are turned into direct-download links; http(s)
  on public hosts only), so the model types one link and the dashboard is built in well under a second on any
  client, including those whose sandbox blocks an upload from the model's shell. `GET /mcp/upload` is now a page
  where a person drops a local file and gets such a link (`/mcp/file/<id>`, 30 minutes); the tool text tells the
  model to ask for that link rather than type out a file of more than ~300 rows.
- **Upload instead of typing the rows.** `POST /mcp/upload` (raw body or multipart; under the `/mcp` path a proxy
  already forwards) holds a CSV/TSV for 30 minutes and returns a `file_id`; `suggest_canvasxpress_dashboard(file_id=...)`
  reads the file on the server, so the tool call is a few bytes and the model never writes the data out (a 1,500-row
  file took the model 6-8 minutes to type as JSON). `csv` (the file's raw text) is the fallback without a shell. The
  tool text says which to use, in that order, without deliberating.
- **Large dashboards reach the view.** A dashboard result carries the complete table, so it can run to hundreds of
  KB, and Claude replaces a tool result that large with an error text the view cannot read (a blank panel). The tool
  now returns a short result (under 2 KB: template, panels, `result_id`, `parts`) and holds the full one for 30
  minutes; the view fetches it in pieces through the host (`get_canvasxpress_dashboard_result`) and draws it. A
  result the host mangled still shows one line in the panel instead of nothing.
- **More forgiving input.** `suggest_canvasxpress_dashboard` also reads a table sent as `{"headers": [...], "rows": [[...]]}`
  or as a list of row objects (the shapes models often send), for `data` and for every file in `tables`; the tool text
  now names the accepted shapes.
- **Several related files, one dashboard.** New `tables` argument on `suggest_canvasxpress_dashboard`: 2 to 6 complete
  files (for example orders, customers and products). The file with the most rows is the base and keeps every row; the
  others are looked up on a column name they share (that column must identify one row of the lookup file, so a patient
  file is never multiplied by its visits). One dashboard is built from the joined table, the header lists the files and
  the insights strip ignores columns that came from a lookup file. Errors say which file could not be related and why.
- **Header panel.** Every suggested dashboard starts with a full-width text panel: the title in bold, larger text and a
  short description (kind of table, complete row and column count, the charts it holds), like the Cohort Explorer.
- **Over time.** A table with an ISO date column gets a panel of its leading measure over time (real date axis, limits
  from the data's range) and a month or year filter. Added after the template is chosen; the view repeats it for a full
  file it loads (`derived` in the result). Slash dates are never guessed.
- **What stands out.** An insights strip computed in the dashboard view from the complete data (never a sample): declared
  limits, a group far from the overall average (for example the one lane with low % Mapped) and extreme values.
- **KPI row.** Every suggested dashboard starts with one row of up to four round cards: Rows, then the average of up to
  three measures (the template's own metrics for QC and survival tables). Headers are humanized column names; the
  cards follow the filters. Added after the template is chosen, so the detection is unchanged.
- **Always the complete data, never a sample.** The tool text asks for every row; a call whose `total_rows` is larger than
  the rows sent is refused and the model is told to send them all. The dashboard view has no file loader, no banner and
  no "Load full file" button: it draws what the tool returned.
- `tests/test_dashboard_view_insights.py` checks the insights maths (a planted bad group, pure noise, limits, outliers),
  and that the view's name humanizing matches the server's.
- `tests/test_detection_pinned.py` pins the survival / QC / batch / group vocabulary, the thresholds and the template
  chosen for the reference tables. `tests/test_dashboard_view_sample.py` tests that a sample is never drawn and the full-file load in a
  browser (set `CX_TEST_ENGINE` and `CX_TEST_DASHBOARDS`).

### Changed
- **`suggest_canvasxpress_dashboard`: retry, then force.** The tool description no longer tells a model to fall back to
  an inline HTML widget (`inline_widget`) when the dashboard view is not shown. It now says: on an error or
  `suggested: false`, read the message, correct the arguments and call again, up to 5 attempts, then call once more with
  the new `force=true`. Never build the dashboard in a widget.
- `force=true` skips the templates and builds a Filters panel plus bar charts: a Bar of the first numeric column or,
  for a table with only discrete columns, a count bar per discrete column (up to 4). Template `minimal`.
- Error messages say what is wrong: an object without `y` (such as `{headers, rows}`) names the accepted shapes; a table
  with no numeric columns points at `force=true`. The "no template fits" message asks for a retry and then `force`.
  The tool still accepts only a list of lists or a `{y, x}` object.

### Fixed
- **Chart view in strict MCP Apps hosts (Claude).** `ui://canvasxpress/chart` never sent `ui/notifications/initialized`
  after `ui/initialize`, so a strict host never delivered the tool result and charts from `generate_canvasxpress_config`,
  `modify_canvasxpress_config`, `generate_km_config` and `create_map_config` did not render. The dashboard view already
  did this; the chart view now matches. A stand-in strict host in `tests/test_chart_view_host.py` reproduces the failure.
- **Map charts.** The view resources now declare `connectDomains: ["https://www.canvasxpress.org"]` (CSP `connect-src`)
  alongside `resourceDomains`; map topology is fetched from `https://www.canvasxpress.org/data/maps/<id>.json`, which a
  host blocks otherwise. Applies to both the chart and the dashboard view.

### Changed
- `pyproject.toml` allows `fastmcp>=3.2.3,<3.4`. FastMCP 3.3.1 passes the test suite and `scripts/verify_live.py` with
  output identical to 3.2.3; `requirements.txt` still pins 3.2.3. FastMCP 4.x is not supported yet.

### Added
- `data/few_shot_supplement.json`: the nine hand-written Meter, Bullet and Bar examples that used to sit at the end of
  `few_shot_examples.json`. `scripts/generate_few_shots.py` appends them when all graph types are generated
  (`--no-supplement`, `--supplement PATH`). The shipped corpus is unchanged.

## [0.1.0] - 2026-10-30

First tagged release. All new behaviour is opt-in; defaults are unchanged except where noted.

### Added
- `GET /health`: liveness plus `retrieval: vector|fallback` (and a reason), with no LLM call or fetch.
- Startup probe of the vector index, and `CX_REQUIRE_VECTOR=1` to refuse to start without it.
- `CX_DISABLED_TOOLS` and `CX_STRICT_TOOLS` to remove tools that cannot work on a host.
- `CX_STRICT_REST=1`: no CORS headers; `/feedback/export` requires `X-Admin-Key`.
- `CX_KB_BASE_URL` (pin the knowledge base to a tag or mirror) and `CX_STATE_DIR` for the knowledge-base cache, with the
  old `data/kb_cache` still read as a fallback.
- `build_index.py --examples/--output/--model`; the index records model, dimension, count and a corpus hash, and the
  server warns on a mismatch.
- `scripts/generate_few_shots.py`: regenerate the few-shot corpus with the engine's `generateExamples`, seeded and
  reproducible.
- `pyproject.toml` with per-provider extras (`anthropic`, `bedrock`, `openai`, `gemini`, `ollama`, `render`, `apsw`,
  `dev`) and a `canvasxpress-mcp` console script.
- GitHub Actions: CI (tests and an index smoke build) and a tag-triggered release.

### Changed
- **Bedrock:** the call retries once without `temperature` when the model rejects it (Claude Sonnet 5 and 5.5), and
  remembers the model.
- **`ADMIN_KEY` unset** now disables the admin endpoints (`/feedback/purge`). Previously a random key was generated and
  printed to the log. Set `ADMIN_KEY` to enable them.
- REST handlers run blocking LLM and tool calls on a worker thread instead of the event loop.
- Knowledge-base cache writes tolerate a read-only or full directory (logged as a warning).
- Older corpus snapshots moved to `data/archive/` (nothing read them).
