"""
Scoring for one generate result. Pure functions; every sub-check is kept in the
record so a reward can be re-weighted later without re-running the model.

Labeled reward (evolve/test), 0 when no config came back, else
    0.35 * graphType match + 0.45 * pair F1 + 0.20 * columns-valid
Unlabeled reward (ood), 0 when no config came back, else the mean of
    columns-valid, schema-clean, and intent match (only when the description
    names a chart type outright; see run_eval.intent_graph_type).

columns-valid = the tool reported no invalid column references (invalid_refs).
The tool's own `valid` flag is NOT used: it also fails on value warnings from
cx_knowledge, whose allowed lists are stale for some params (e.g. colorScheme
rejects WallStreetJournal3, a real scheme), and that would score correct
configs as wrong. schema-clean = no errors against the published config schema;
cx_knowledge value warnings are recorded (value_warnings) but not scored.
"""
import math

W_GRAPH, W_PAIRS, W_HEADERS = 0.35, 0.45, 0.20


def _norm_scalar(v):
    return v.strip() if isinstance(v, str) else v


def values_equal(a, b) -> bool:
    """Lenient equality: numeric tolerance, and [x] == x for single-item lists."""
    if isinstance(a, list) and len(a) == 1 and not isinstance(b, list):
        a = a[0]
    if isinstance(b, list) and len(b) == 1 and not isinstance(a, list):
        b = b[0]
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(values_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(values_equal(a[k], b[k]) for k in a)
    return _norm_scalar(a) == _norm_scalar(b)


def pair_scores(expected: dict, got: dict) -> dict:
    """Precision/recall/F1 over (key, value) pairs, graphType excluded."""
    exp = {k: v for k, v in expected.items() if k != "graphType"}
    out = {k: v for k, v in got.items() if k != "graphType"}
    if not exp and not out:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "key_recall": 1.0,
                "missing": [], "extra": [], "wrong": []}
    matched = [k for k in exp if k in out and values_equal(exp[k], out[k])]
    precision = len(matched) / len(out) if out else 0.0
    recall = len(matched) / len(exp) if exp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "key_recall": round(sum(1 for k in exp if k in out) / len(exp), 4) if exp else 1.0,
        "missing": sorted(k for k in exp if k not in out),
        "extra": sorted(k for k in out if k not in exp),
        "wrong": sorted(k for k in exp if k in out and k not in matched),
    }


def schema_clean(lint: dict) -> bool:
    """No errors against the published config JSON Schema."""
    return not lint.get("schema_errors")


def value_warnings(lint: dict) -> list:
    return sorted(((lint.get("param_warnings") or {}).get("invalid_values") or {}))


def grade(task: dict, result: dict, lint: dict, intent_graph_type) -> dict:
    config = result.get("config") or {}
    parsed = bool(config)
    header_valid = parsed and not result.get("invalid_refs")
    clean = schema_clean(lint) if parsed else False
    got_type = str(config.get("graphType", "")).lower()

    checks = {"parsed": parsed, "header_valid": header_valid, "schema_clean": clean,
              "value_warnings": value_warnings(lint) if parsed else [],
              "graph_type": config.get("graphType")}

    if task.get("expected") is not None:
        expected = task["expected"]
        graph_ok = parsed and got_type == str(expected.get("graphType", "")).lower()
        pairs = pair_scores(expected, config) if parsed else pair_scores(expected, {})
        checks.update({"graph_type_ok": graph_ok, "pairs": pairs})
        reward = 0.0 if not parsed else (
            W_GRAPH * graph_ok + W_PAIRS * pairs["f1"] + W_HEADERS * header_valid)
    else:
        parts = [header_valid, clean]
        intent_ok = None
        if intent_graph_type:
            intent_ok = parsed and got_type == intent_graph_type.lower()
            parts.append(intent_ok)
        checks.update({"intent_graph_type": intent_graph_type, "intent_ok": intent_ok})
        reward = 0.0 if not parsed else sum(parts) / len(parts)

    checks["reward"] = round(float(reward), 4)
    return checks
