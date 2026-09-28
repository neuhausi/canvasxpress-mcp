"""
Shared plumbing for the generate-config eval: paths, server import, leakage-safe
retrieval and synthetic data.

The server module is imported in-process (not over HTTP) so the eval measures the
exact generate path production runs, without writing to the production call log.
"""
import json
import os
import random
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPO = EVAL_DIR.parent
DATA_DIR = EVAL_DIR / "data"
RUNS_DIR = EVAL_DIR / "runs"

SETS = ("evolve", "test", "ood")

# Extra candidates pulled from the index so that, after the held-out items and
# their duplicates are dropped, retrieval still returns a full top_k.
_RETRIEVE_SLACK = 40


def config_key(config: dict) -> str:
    """Canonical JSON of a config, used to spot duplicate pool entries."""
    return json.dumps(config, sort_keys=True)


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def load_server(state_dir: Path):
    """Import src/server.py with .env applied and state redirected to `state_dir`.

    CX_STATE_DIR must be set before import: it moves the call-log DB out of data/
    so eval traffic never lands in the local usage metrics.
    """
    state_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CX_STATE_DIR"] = str(state_dir)
    sys.path.insert(0, str(REPO / "src"))
    from dotenv import load_dotenv
    load_dotenv(REPO / ".env", override=False)
    os.environ["CX_STATE_DIR"] = str(state_dir)
    import server  # noqa: E402
    return server


def install_leakage_filter(server, banned_descriptions: set, banned_configs: set) -> None:
    """Wrap server.retrieve_examples so held-out items never reach the prompt.

    A pool entry is dropped when its description OR its config matches a held-out
    item; the second check catches the pool's exact duplicates (same pair stored
    twice), which would otherwise leak the answer under a different id.
    The production embeddings.db is not modified.
    """
    original = server.retrieve_examples

    def filtered(query: str, top_k: int = 6) -> list[dict]:
        pulled = original(query, top_k + _RETRIEVE_SLACK)
        kept = [
            ex for ex in pulled
            if ex["description"] not in banned_descriptions
            and config_key(ex["config"]) not in banned_configs
        ]
        return kept[:top_k]

    server.retrieve_examples = filtered


# ---------------------------------------------------------------------------
# Synthetic data (for lint and render checks; the pool stores no real data)
# ---------------------------------------------------------------------------

_FACTOR_HINTS = (
    "group", "category", "type", "set", "region", "class", "cluster", "treatment",
    "condition", "status", "sex", "gender", "stage", "label", "factor", "species",
    "responder", "party", "segment", "tissue", "batch", "cohort", "arm",
)


def guess_type(name: str) -> str:
    low = name.lower()
    if any(h in low for h in _FACTOR_HINTS):
        return "factor"
    if any(h in low for h in ("date", "time", "day", "month", "year", "quarter")):
        return "date"
    if low in ("id", "sample", "name", "gene", "smp", "var"):
        return "string"
    return "numeric"


def synth_data(headers: list[str], column_types: dict | None, n_rows: int = 12,
               seed: int = 0) -> list[list]:
    """CSV-style array-of-arrays with plausible values for each column type."""
    rng = random.Random(seed)
    types = [(column_types or {}).get(h) or guess_type(h) for h in headers]
    rows = [list(headers)]
    for i in range(n_rows):
        row = []
        for t in types:
            if t == "numeric":
                row.append(round(rng.uniform(1, 100), 2))
            elif t == "factor":
                row.append("ABC"[i % 3])
            elif t == "date":
                row.append("2026-%02d-01" % (i % 12 + 1))
            else:
                row.append("S%d" % (i + 1))
        rows.append(row)
    return rows
