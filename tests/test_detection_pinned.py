"""The domain-aware detection is pinned.

Recognising a survival cohort or a per-sample sequencing-QC table from its column names is what makes
a CanvasXpress dashboard understand the data instead of just plotting columns. These tests fail if the
vocabulary, the thresholds or the outcome for the reference tables change by accident. Change them only
on purpose, and then update the JavaScript reference in canvasxpress-dashboards and its parity fixture.

The reference tables reproduce the column names and types of the 14 files in the CanvasXpress MCP app
test set (cohort, QC, generic, edge cases), with synthetic values.
"""

import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cx_dashboard_suggest as sg  # noqa: E402
import cx_datasets  # noqa: E402

PINNED_PATTERNS = {
    '_SURVIVAL_TIME_RE': ('^(time|t|surv(ival)?|os|pfs|dfs|efs|tte|duration|follow[\\s._-]?up|fu|days?|months?|weeks?|years?)$|^(os|pfs|dfs|efs|rfs|dss|ttp)[\\s._-]?(time|months?|days?|weeks?|years?|duration)$|^(overall|progression[\\s._-]?free|disease[\\s._-]?free|event[\\s._-]?free|relapse[\\s._-]?free)[\\s._-]?survival[\\s._-]?(time|months?|days?|weeks?|years?)$', 34),
    '_SURVIVAL_EVENT_RE': ('(event|status|dead|death|censor|vital|outcome|relapse|progress|recur)', 34),
    '_EVENT_LEVEL_RE': ('^(dead|alive|death|censor(ed)?|event|no[\\s._-]?event|yes|no|deceased|living|1|0|relapse|progression|cr|pd)$', 34),
    '_BATCH_RE': ('(lane|flow[\\s._-]?cell|batch|run|plate|library|prep|instrument|machine|seq)', 34),
    '_COHORT_GROUP_RE': ('(arm|group|treat|cohort|stage|condition|sex|subtype|genotype)', 34),
}
PINNED_QC_METRICS = {
    'reads': ('(^|[\\s._-])(total[\\s._-]?)?reads?($|[\\s._-])|read[\\s._-]?(pairs?|count)|^n[\\s._-]?reads$|fragments?', 34),
    'mapping': ('(mapped|mapping|aligned|alignment)', 34),
    'quality': ('(q30|q20|phred|base[\\s._-]?qual|mean[\\s._-]?qual)', 34),
    'duplication': ('(dup|duplicat)', 34),
    'gc': ('(^|[\\s._-])gc($|[\\s._-])|gc[\\s._-]?(content|pct|percent)', 34),
    'insert': ('insert[\\s._-]?size', 34),
    'coverage': ('(coverage|depth|cov[\\s._-]?x)', 34),
    'adapter': ('adapter', 34),
    'contamination': ('(contamin|freemix)', 34),
}
PINNED_CONSTANTS = {'_MIN_PANELS': 3, '_MAX_FILTER_FIELDS': 6, '_MIN_FACTOR_LEVELS': 2, '_MAX_FACTOR_LEVELS': 12}


@pytest.mark.parametrize("name", sorted(PINNED_PATTERNS))
def test_vocabulary_pattern_is_unchanged(name):
    pattern, flags = PINNED_PATTERNS[name]
    actual = getattr(sg, name)
    assert (actual.pattern, actual.flags) == (pattern, flags)


def test_qc_metric_families_are_unchanged():
    assert list(sg._QC_METRICS) == list(PINNED_QC_METRICS)
    for family, (pattern, flags) in PINNED_QC_METRICS.items():
        assert (sg._QC_METRICS[family].pattern, sg._QC_METRICS[family].flags) == (pattern, flags), family


def test_thresholds_are_unchanged():
    assert {k: getattr(sg, k) for k in PINNED_CONSTANTS} == PINNED_CONSTANTS


# ---------------------------------------------------------------------------
# Reference tables: (column name, kind) with the kinds of the real test files.
# ---------------------------------------------------------------------------
def _table(columns, rows=80, seed=1):
    rng = random.Random(seed)
    out = [[name for name, _ in columns]]
    for i in range(rows):
        row = []
        for name, kind in columns:
            if kind == "id":
                row.append("%s%03d" % (name[:1].upper(), i + 1))
            elif kind == "event":
                row.append(rng.choice([0, 1]))
            elif kind == "text-event":
                row.append(rng.choice(["Dead", "Alive"]))
            elif kind == "date":
                row.append("2025-%02d-%02d" % (1 + i // 28 % 12, 1 + i % 28))
            elif kind == "levels60":
                row.append("city%02d" % rng.randrange(60))
            elif isinstance(kind, tuple) and kind[0] == "cat":
                row.append(rng.choice(kind[1]))
            elif isinstance(kind, tuple) and kind[0] == "num":
                row.append(round(rng.uniform(kind[1], kind[2]), 2))
            else:
                raise AssertionError(kind)
        out.append(row)
    return out


def _cat(*levels):
    return ("cat", list(levels))


def _num(low, high):
    return ("num", low, high)


SURVIVAL = [("patient", "id"), ("OS_months", _num(1, 80)), ("OS_event", "event"), ("arm", _cat("A", "B")),
            ("sex", _cat("F", "M")), ("stage", _cat("I", "II", "III")), ("age", _num(30, 85)), ("bmi", _num(18, 40))]
QC = [("sample", "id"), ("lane", _cat("L1", "L2", "L3", "L4")), ("batch", _cat("RunA", "RunB", "RunC")),
      ("library", _cat("Exome", "PolyA")), ("total_reads", _num(3e7, 7e7)), ("pct_mapped", _num(80, 99)),
      ("pct_duplication", _num(5, 30)), ("pct_q30", _num(85, 98)), ("gc_content", _num(38, 58)),
      ("insert_size", _num(180, 380)), ("contamination_freemix", _num(0, 0.04))]

# file number, name, columns, rows, expected template (None = declined)
REFERENCE = [
    (1, "survival-cohort-small-60", SURVIVAL, 60, "cohort"),
    (3, "survival-cohort-300", SURVIVAL + [("site", _cat("s1", "s2", "s3")), ("biomarker", _num(0, 9))], 300, "cohort"),
    (4, "survival-TEXT-event-Dead-Alive (a text event: no curve, generic explorer)",
     [c if c[0] != "OS_event" else ("vital_status", "text-event") for c in SURVIVAL], 150, "explore"),
    (5, "ngs-qc-small-24", QC, 24, "ngs-qc"),
    (6, "ngs-qc-96", QC, 96, "ngs-qc"),
    (8, "sales-generic-300", [("order", "id"), ("date", "date"), ("region", _cat("N", "S", "E", "W")),
                              ("product", _cat("p1", "p2", "p3")), ("channel", _cat("web", "store")),
                              ("units", _num(1, 50)), ("unit_price", _num(5, 90)), ("discount", _num(0, 0.4)),
                              ("revenue", _num(10, 900)), ("rating", _num(1, 5))], 300, "explore"),
    (10, "expression-wide-80x40", [("sample", "id"), ("tissue", _cat("liver", "lung")), ("condition", _cat("ctl", "trt"))]
         + [("GENE%d" % i, _num(0, 12)) for i in range(1, 41)], 80, "explore"),
    (11, "timeseries-daily-3-sites", [("date", "date"), ("site", _cat("a", "b", "c")), ("temp_c", _num(-5, 30)),
                                      ("humidity_pct", _num(20, 95)), ("wind_kph", _num(0, 60))], 1095, "explore"),
    (12, "tiny-12-rows", [("id", "id"), ("group", _cat("g1", "g2", "g3")), ("x", _num(0, 9)), ("y", _num(0, 9))], 12, "explore"),
    (13, "high-cardinality-and-missing-400", [("customer", "id"), ("city_60_levels", "levels60"),
                                              ("tier", _cat("gold", "silver", "bronze")), ("spend", _num(5, 900)),
                                              ("visits", _num(1, 40))], 400, "explore"),
    (14, "numeric-only-no-factor-EXPECT-DECLINE", [("a", _num(0, 9)), ("b", _num(0, 9)), ("c", _num(0, 9))], 50, None),
]


@pytest.mark.parametrize("number,name,columns,rows,expected", REFERENCE, ids=[r[1].split(" ")[0] for r in REFERENCE])
def test_reference_table_gets_the_same_template(number, name, columns, rows, expected):
    result = sg.suggest(_table(columns, rows))
    if expected is None:
        assert result["suggested"] is False
        return
    assert result["suggested"] and result["template"] == expected, (number, name, result.get("reason"))
    assert result["valid"], result["errors"]


def test_survival_and_qc_templates_keep_their_own_panels():
    cohort = sg.suggest(_table(SURVIVAL, 60))
    assert [p["graphType"] for p in cohort["panels"] if not p["id"].startswith("kpi-")] == [
        "filters", "KaplanMeier", "Boxplot", "Scatter2D", "Bar"]
    qc = sg.suggest(_table(QC, 24))
    assert [p["id"] for p in qc["panels"] if not p["id"].startswith("kpi-")] == [
        "filters", "depth", "quality-scatter", "quality-spread", "case"]


def test_the_template_chooser_knows_nothing_about_kpi_cards():
    """suggest_dashboard (the part mirrored by the JavaScript reference) never returns a KPI panel."""
    for columns, rows in ((SURVIVAL, 60), (QC, 24)):
        data = cx_datasets.table_to_data(_table(columns, rows), id_column=columns[0][0])
        panels = sg.suggest_dashboard(data)["panels"]
        assert not any(p["id"].startswith("kpi-") for p in panels)
