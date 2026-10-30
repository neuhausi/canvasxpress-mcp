# Changelog

All notable changes to canvasxpress-mcp. Releases are tagged `vX.Y.Z`; pin a tag rather than a commit.

## [Unreleased]

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
