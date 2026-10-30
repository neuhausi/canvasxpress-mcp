# Changelog

All notable changes to canvasxpress-mcp. Releases are tagged `vX.Y.Z`; pin a tag rather than a commit.

## [Unreleased]

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
