"""scripts/generate_few_shots.py: id renumbering (offline) and, when CX_TEST_ENGINE points at a
canvasXpress.min.js (URL or path), a real seeded run in headless Chromium."""

import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
SCRIPT = os.path.join(ROOT, "scripts", "generate_few_shots.py")
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import generate_few_shots  # noqa: E402


def test_renumber_gives_consecutive_ids():
    records = [{"id": 1}, {"id": 1}, {"id": 2}]
    assert [r["id"] for r in generate_few_shots.renumber(records)] == [1, 2, 3]


def _run(engine, seed, out, graphs="Bar graph,Boxplot"):
    return subprocess.run(
        [sys.executable, SCRIPT, "--engine", engine, "--graphs", graphs,
         "--per-graph", "3", "--seed", str(seed), "--output", str(out)],
        capture_output=True, text=True)


@pytest.mark.skipif(not os.environ.get("CX_TEST_ENGINE"), reason="set CX_TEST_ENGINE to run")
def test_seeded_run_is_reproducible_and_has_the_corpus_shape(tmp_path):
    pytest.importorskip("playwright")
    engine = os.environ["CX_TEST_ENGINE"]
    a, b, c = tmp_path / "a.json", tmp_path / "b.json", tmp_path / "c.json"
    for seed, out in ((42, a), (42, b), (7, c)):
        result = _run(engine, seed, out)
        assert result.returncode == 0, result.stderr or result.stdout
    assert a.read_bytes() == b.read_bytes()
    assert a.read_bytes() != c.read_bytes()
    records = json.loads(a.read_text())
    assert len(records) == 6
    assert sorted(records[0]) == ["config", "data", "description", "header", "id"]
    assert [r["id"] for r in records] == [1, 2, 3, 4, 5, 6]


@pytest.mark.skipif(not os.environ.get("CX_TEST_ENGINE"), reason="set CX_TEST_ENGINE to run")
def test_unknown_graph_name_is_a_clear_error(tmp_path):
    pytest.importorskip("playwright")
    result = _run(os.environ["CX_TEST_ENGINE"], 1, tmp_path / "x.json", graphs="Bar")
    assert result.returncode != 0 and "Unknown graph name" in (result.stderr + result.stdout)


# --- curated supplement -------------------------------------------------------------

def test_append_supplement_adds_after_generated_with_defaults():
    generated = [{"id": 1, "data": "data", "header": ["Col1"], "description": "g", "config": {}}]
    merged = generate_few_shots.append_supplement(generated, [{"description": "s", "config": {"graphType": "Bar"}}])
    assert [r["description"] for r in merged] == ["g", "s"]
    assert merged[1]["data"] == "data" and merged[1]["header"] == []


def test_load_supplement_validates_records(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"description": "ok", "config": {}}, {"description": "no config"}]))
    with pytest.raises(SystemExit, match="record 1"):
        generate_few_shots.load_supplement(bad)
    good = tmp_path / "good.json"
    good.write_text(json.dumps([{"description": "ok", "config": {}}]))
    assert len(generate_few_shots.load_supplement(good)) == 1


def test_shipped_supplement_is_valid():
    path = os.path.join(ROOT, "data", "few_shot_supplement.json")
    assert len(generate_few_shots.load_supplement(path)) == 9


@pytest.mark.skipif(not os.environ.get("CX_TEST_ENGINE"), reason="set CX_TEST_ENGINE to run")
def test_all_graphs_run_appends_supplement_but_a_subset_does_not(tmp_path):
    pytest.importorskip("playwright")
    engine = os.environ["CX_TEST_ENGINE"]
    full, subset, skipped = tmp_path / "full.json", tmp_path / "subset.json", tmp_path / "skip.json"
    base = [sys.executable, SCRIPT, "--engine", engine, "--per-graph", "1", "--seed", "3"]
    assert subprocess.run(base + ["--output", str(full)], capture_output=True).returncode == 0
    assert subprocess.run(base + ["--graphs", "Bar graph", "--output", str(subset)],
                          capture_output=True).returncode == 0
    assert subprocess.run(base + ["--no-supplement", "--output", str(skipped)],
                          capture_output=True).returncode == 0
    full_records = json.loads(full.read_text())
    assert len(full_records) == len(json.loads(skipped.read_text())) + 9
    assert len(json.loads(subset.read_text())) == 1
    shipped = json.loads(open(os.path.join(ROOT, "data", "few_shot_supplement.json")).read())
    assert [r["description"] for r in full_records[-9:]] == [r["description"] for r in shipped]
    assert [r["id"] for r in full_records] == list(range(1, len(full_records) + 1))
