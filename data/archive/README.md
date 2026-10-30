# Archived few-shot snapshots

Earlier snapshots of the few-shot corpus. **Nothing reads these files**: the server and
`build_index.py` use only `data/few_shot_examples.json`.

| File | Examples | Record keys |
|---|---|---|
| `few_shot_examples-106.json` | 106 | `id, type, description, config` |
| `few_shot_examples-3366.json` | 3,366 | `id, data, header, config, description` |
| `few_shot_examples-5040.json` | 5,040 | `id, data, header, description, config` |

They overlap with the current corpus rather than extending it, so they are kept for
reference and for comparing retrieval quality across corpus generations (see `eval/`).

To index one of them anyway:

    python build_index.py --examples data/archive/few_shot_examples-5040.json --output /tmp/idx.db
