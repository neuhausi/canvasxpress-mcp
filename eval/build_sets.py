"""
Build the eval task sets (deterministic; re-running reproduces them exactly).

    evolve  labeled, from few_shot_examples.json   -> what a prompt or retrieval search may tune on
    test    labeled, from few_shot_examples.json   -> never used for tuning; reported only
    ood     unlabeled, real user descriptions      -> generalisation check (validity only)

Labeled items are stratified by graphType (PER_TYPE each for evolve and test).
Every evolve/test item is excluded from retrieval at eval time (see common.py),
together with any pool entry that shares its description or config.

ood comes from the LOCAL call log (data/call_log.db) and contains real user
text: eval/data/ is gitignored and must never be committed or published.

Run from repo root:  ../.venv/bin/python eval/build_sets.py [--per-type 2] [--ood 100]
"""
import argparse
import collections
import json
import random
import sqlite3

from common import DATA_DIR, REPO, write_jsonl

SEED = 20260928


def labeled_sets(per_type: int) -> tuple[list[dict], list[dict]]:
    pool = json.load(open(REPO / "data" / "few_shot_examples.json"))
    # Collapse exact duplicates so one logical item cannot land in both splits.
    unique, seen = [], set()
    for ex in pool:
        key = (ex["description"], json.dumps(ex["config"], sort_keys=True))
        if key not in seen:
            seen.add(key)
            unique.append(ex)
    by_type = collections.defaultdict(list)
    for ex in unique:
        by_type[ex["config"].get("graphType", "?")].append(ex)

    rng = random.Random(SEED)
    evolve, test = [], []
    for graph_type in sorted(by_type):
        items = sorted(by_type[graph_type], key=lambda e: e["id"])
        rng.shuffle(items)
        for split, chunk in ((evolve, items[:per_type]),
                             (test, items[per_type:2 * per_type])):
            for ex in chunk:
                split.append({
                    "task_id": "fs-%d" % ex["id"],
                    "description": ex["description"],
                    "headers": ex["header"],
                    "column_types": None,
                    "expected": ex["config"],
                })
    return evolve, test


def _parse_headers(raw):
    """Same rules as the server's HTTP layer: list, JSON array string, or comma list."""
    if isinstance(raw, list):
        return [str(h).strip() for h in raw if str(h).strip()]
    raw = (raw or "").strip()
    if raw.startswith("["):
        return json.loads(raw)
    return [h.strip() for h in raw.split(",") if h.strip()]


def _parse_column_types(raw):
    """Same rules as the server's HTTP layer: dict, JSON object string, or k=v list."""
    if isinstance(raw, dict) or raw is None:
        return raw
    raw = raw.strip()
    if not raw:
        return None
    if raw.startswith("{"):
        return json.loads(raw)
    result = {}
    for item in raw.split(","):
        k, _, v = item.strip().partition("=")
        if k.strip() and v.strip():
            result[k.strip()] = v.strip()
    return result or None


def ood_set(n: int) -> list[dict]:
    con = sqlite3.connect("file:%s?mode=ro" % (REPO / "data" / "call_log.db"), uri=True)
    rows = con.execute(
        "SELECT id, request FROM tool_calls "
        "WHERE tool='generate_canvasxpress_config' AND status=200 ORDER BY id"
    ).fetchall()
    tasks, seen = [], set()
    for row_id, request in rows:
        try:
            req = json.loads(request)
        except (TypeError, ValueError):
            continue
        desc = (req.get("description") or "").strip()
        try:
            headers = _parse_headers(req.get("headers"))
            column_types = _parse_column_types(req.get("column_types"))
        except ValueError:
            continue
        if not desc or not headers or desc in seen:
            continue
        seen.add(desc)
        tasks.append({
            "task_id": "log-%s" % str(row_id)[:12],
            "description": desc,
            "headers": headers,
            "column_types": column_types,
            "expected": None,
        })
    random.Random(SEED).shuffle(tasks)
    return sorted(tasks[:n], key=lambda t: t["task_id"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-type", type=int, default=2)
    ap.add_argument("--ood", type=int, default=100)
    args = ap.parse_args()

    evolve, test = labeled_sets(args.per_type)
    ood = ood_set(args.ood)
    for name, rows in (("evolve", evolve), ("test", test), ("ood", ood)):
        write_jsonl(DATA_DIR / ("%s.jsonl" % name), rows)
        print("%-7s %4d tasks -> %s" % (name, len(rows), DATA_DIR / ("%s.jsonl" % name)))


if __name__ == "__main__":
    main()
