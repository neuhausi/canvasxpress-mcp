"""
Compare two runs on the same set: paired per-task difference with a bootstrap
95% interval, plus the tasks that moved most.

    ../.venv/bin/python eval/compare.py base-a base-b --set evolve

Comparing two runs of the UNCHANGED server estimates the noise band: any later
change whose gain sits inside that band has not been shown to help.
"""
import argparse
import random
import statistics

from common import RUNS_DIR, load_jsonl


def task_means(label: str, set_name: str) -> dict:
    by_task = {}
    for r in load_jsonl(RUNS_DIR / label / ("%s.jsonl" % set_name)):
        by_task.setdefault(r["task_id"], []).append(r["reward"])
    return {k: statistics.mean(v) for k, v in by_task.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--set", required=True)
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()

    a, b = task_means(args.a, args.set), task_means(args.b, args.set)
    common = sorted(set(a) & set(b))
    diffs = [b[t] - a[t] for t in common]
    mean = statistics.mean(diffs)
    rng = random.Random(0)
    boot = sorted(
        statistics.mean(rng.choice(diffs) for _ in diffs) for _ in range(2000))
    lo, hi = boot[49], boot[1949]

    print("%s: %d paired tasks" % (args.set, len(common)))
    print("  S(%s) = %.4f   S(%s) = %.4f" % (
        args.a, statistics.mean(a[t] for t in common),
        args.b, statistics.mean(b[t] for t in common)))
    print("  delta = %+.4f   95%% CI [%+.4f, %+.4f]" % (mean, lo, hi))
    print("  tasks changed: %d up, %d down, %d same" % (
        sum(d > 1e-9 for d in diffs), sum(d < -1e-9 for d in diffs),
        sum(abs(d) <= 1e-9 for d in diffs)))
    moved = sorted(zip(diffs, common))
    if moved and moved[0][0] < 0:
        print("  biggest drops:", ", ".join(
            "%s %+.2f" % (t, d) for d, t in moved[:args.top] if d < 0))
    if moved and moved[-1][0] > 0:
        print("  biggest gains:", ", ".join(
            "%s %+.2f" % (t, d) for d, t in reversed(moved[-args.top:]) if d > 0))


if __name__ == "__main__":
    main()
