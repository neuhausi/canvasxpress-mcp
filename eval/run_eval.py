"""
Run the generate path over one task set, k trials per task, and grade each trial.

    ../.venv/bin/python eval/run_eval.py --set evolve --label base-a --k 1
    ../.venv/bin/python eval/run_eval.py --set test --label base-a --k 1 --limit 10

Writes eval/runs/<label>/<set>.jsonl (one record per task x trial; resume-safe:
re-running skips trials already recorded) and eval/runs/<label>/<set>.summary.json.
Model, provider and prompts are whatever src/ and .env currently hold.
"""
import argparse
import json
import re
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from common import (DATA_DIR, RUNS_DIR, config_key, install_leakage_filter,
                    load_jsonl, load_server, synth_data)
from grade import grade

# Keywords that name a chart outright. Others in the server's table ("correlation",
# "density", "survival", "pca", ...) often describe the data, not the chart, so they
# only count as intent when followed by chart/plot/graph/map/diagram.
_UNAMBIGUOUS = {
    "heatmap", "heat map", "scatterplot", "boxplot", "box plot", "violin", "volcano",
    "barplot", "histogram", "sankey", "alluvial", "venn", "treemap", "tree map",
    "donut", "lollipop", "waterfall", "ridgeline", "gantt", "tornado",
}
_CHART_WORD = r"\s+(chart|plot|graph|map|diagram)\b"


def intent_graph_type(server, description: str):
    low = description.lower()
    for kw in sorted(server._GRAPH_TYPE_KEYWORDS, key=len, reverse=True):
        pattern = r"\b" + re.escape(kw) + r"\b"
        if not re.search(pattern, low):
            continue
        if kw in _UNAMBIGUOUS or any(w in kw for w in ("chart", "plot", "graph")):
            return server._GRAPH_TYPE_KEYWORDS[kw]
        if re.search(pattern + _CHART_WORD, low):
            return server._GRAPH_TYPE_KEYWORDS[kw]
        return None
    return None


def run_one(server, task: dict, trial: int, temperature: float) -> dict:
    import cx_validate
    started = time.time()
    error = None
    try:
        result = server.generate_canvasxpress_config(
            description=task["description"],
            headers=task["headers"],
            column_types=task.get("column_types"),
            temperature=temperature,
        )
    except Exception as exc:  # noqa: BLE001 - a crash is a scored failure, not a stop
        result, error = {"config": {}}, "%s: %s" % (type(exc).__name__, exc)
    config = result.get("config") or {}
    data = synth_data(task["headers"], task.get("column_types"), seed=trial)
    lint = cx_validate.lint_figure({"data": data, "config": config}) if config else {}
    checks = grade(task, result, lint, intent_graph_type(server, task["description"]))
    usage = result.get("usage") or {}
    return {
        "task_id": task["task_id"],
        "trial": trial,
        "reward": checks.pop("reward"),
        "checks": checks,
        "config": config,
        "warnings": result.get("warnings", []),
        "invalid_refs": result.get("invalid_refs") or {},
        "removed_params": result.get("removed_params", []),
        "cost_usd": usage.get("cost_usd", 0.0),
        "tokens_in": usage.get("input_tokens", 0),
        "tokens_out": usage.get("output_tokens", 0),
        "cache_read": usage.get("cache_read_input_tokens", 0),
        "seconds": round(time.time() - started, 2),
        "error": error,
    }


def summarize(records: list[dict], tasks: list[dict], label: str, set_name: str,
              model: str) -> dict:
    by_task = {}
    for r in records:
        by_task.setdefault(r["task_id"], []).append(r["reward"])
    task_means = [statistics.mean(v) for v in by_task.values()]
    n = len(task_means)

    def mean_of(fn):
        vals = [fn(r) for r in records]
        vals = [v for v in vals if v is not None]
        return round(statistics.mean(vals), 4) if vals else None

    summary = {
        "label": label, "set": set_name, "model": model,
        "tasks": n, "tasks_in_set": len(tasks), "trials": len(records),
        "S": round(statistics.mean(task_means), 4) if n else None,
        # Standard error over tasks: the spread a different task sample would show.
        "S_stderr": round(statistics.stdev(task_means) / n ** 0.5, 4) if n > 1 else None,
        "parsed": mean_of(lambda r: float(r["checks"]["parsed"])),
        "header_valid": mean_of(lambda r: float(r["checks"]["header_valid"])),
        "schema_clean": mean_of(lambda r: float(r["checks"]["schema_clean"])),
        "value_warning_rate": mean_of(lambda r: float(bool(r["checks"]["value_warnings"]))),
        "errors": sum(1 for r in records if r["error"]),
        "cost_usd": round(sum(r["cost_usd"] for r in records), 4),
        "cost_per_call_usd": round(sum(r["cost_usd"] for r in records) / len(records), 5)
        if records else None,
        "tokens_in_mean": mean_of(lambda r: r["tokens_in"]),
        "tokens_out_mean": mean_of(lambda r: r["tokens_out"]),
        "seconds_mean": mean_of(lambda r: r["seconds"]),
    }
    if any(r["checks"].get("pairs") for r in records):
        summary.update({
            "graph_type_ok": mean_of(lambda r: float(r["checks"]["graph_type_ok"])),
            "pair_f1": mean_of(lambda r: r["checks"]["pairs"]["f1"]),
            "pair_precision": mean_of(lambda r: r["checks"]["pairs"]["precision"]),
            "pair_recall": mean_of(lambda r: r["checks"]["pairs"]["recall"]),
            "key_recall": mean_of(lambda r: r["checks"]["pairs"]["key_recall"]),
        })
    else:
        summary["intent_ok"] = mean_of(
            lambda r: None if r["checks"]["intent_ok"] is None else float(r["checks"]["intent_ok"]))
        summary["intent_tasks"] = sum(1 for r in records if r["checks"]["intent_ok"] is not None)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["evolve", "test", "ood"])
    ap.add_argument("--label", required=True, help="run name, e.g. base-a")
    ap.add_argument("--k", type=int, default=1, help="trials per task")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="first N tasks only (smoke)")
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    tasks = load_jsonl(DATA_DIR / ("%s.jsonl" % args.set))
    if args.limit:
        tasks = tasks[:args.limit]
    held_out = load_jsonl(DATA_DIR / "evolve.jsonl") + load_jsonl(DATA_DIR / "test.jsonl")

    run_dir = RUNS_DIR / args.label
    server = load_server(run_dir / "state")
    install_leakage_filter(
        server,
        {t["description"] for t in held_out},
        {config_key(t["expected"]) for t in held_out},
    )

    out_path = run_dir / ("%s.jsonl" % args.set)
    done = set()
    if out_path.exists():
        for r in load_jsonl(out_path):
            done.add((r["task_id"], r["trial"]))
    todo = [(t, i) for t in tasks for i in range(args.k) if (t["task_id"], i) not in done]
    print("%s/%s: %d tasks x k=%d, %d to run (%d already done), model %s"
          % (args.label, args.set, len(tasks), args.k, len(todo), len(done), server.MODEL),
          flush=True)

    lock = threading.Lock()
    with open(out_path, "a") as out, ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run_one, server, t, i, args.temperature) for t, i in todo]
        for n, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            with lock:
                out.write(json.dumps(rec) + "\n")
                out.flush()
            if n % 10 == 0 or n == len(futures):
                print("  %d/%d" % (n, len(futures)), flush=True)

    wanted = {t["task_id"] for t in tasks}
    records = [r for r in load_jsonl(out_path) if r["task_id"] in wanted]
    summary = summarize(records, tasks, args.label, args.set, server.MODEL)
    (run_dir / ("%s.summary.json" % args.set)).write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
