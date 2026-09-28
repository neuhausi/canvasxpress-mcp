# generate-config eval

Scores the `generate_canvasxpress_config` path (retrieval → prompt → LLM →
post-processing) on fixed task sets. Use it to check whether a prompt,
retrieval or knowledge change actually helps before shipping it.

The server module is imported in-process with `.env` applied, so a run measures
exactly what `src/` and the configured model produce. Eval traffic goes to
`eval/runs/<label>/state/`, never to the local call log.

## Task sets

| Set | Source | Labels | Use |
|---|---|---|---|
| `evolve` | `few_shot_examples.json`, 2 per graphType (128) | expected config | tune against |
| `test` | same pool, 2 more per graphType (128) | expected config | report only, never tune on |
| `ood` | 100 real user descriptions from `data/call_log.db` | none | check that changes generalise |

All `evolve` and `test` items are removed from retrieval at eval time, along
with any pool entry that shares their description or config (the pool stores
153 exact duplicates). `data/embeddings.db` is not modified.

`eval/data/` and `eval/runs/` are gitignored: `ood` contains real user text.

## Scoring

Per trial, 0 when no config comes back. Otherwise:

- **labeled** (`evolve`, `test`): `0.35·graphType match + 0.45·pair F1 + 0.20·columns valid`.
  Pair F1 is over (key, value) pairs, excluding graphType, with numeric
  tolerance and `[x] == x`.
- **ood**: mean of columns valid, schema clean, and intent match. Intent is
  scored only when the description names a chart outright ("heatmap",
  "bar chart", ...).

"Columns valid" means no invalid column references. "Schema clean" means no
errors against the published config schema. `cx_knowledge` value warnings are
recorded but not scored (their allowed lists are stale for some params).
Every sub-check is saved per trial, so the weights can change without re-running.

## Commands

From the repo root:

```bash
../.venv/bin/python eval/build_sets.py                       # (re)build sets, deterministic
../.venv/bin/python eval/run_eval.py --set evolve --label base-a
../.venv/bin/python eval/run_eval.py --set test --label base-a --k 2 --limit 10
../.venv/bin/python eval/compare.py base-a base-b --set evolve
```

`run_eval.py` is resume-safe: re-running a label skips trials already recorded.
Cost with Sonnet 4.6 is roughly $0.01–0.03 per call.

## Reading a comparison

Run the unchanged server twice (`base-a`, `base-b`) and compare them first: that
difference is the noise band. A later change counts only when its gain on
`evolve` clears that band, `test` does not drop, and `ood` does not drop.
