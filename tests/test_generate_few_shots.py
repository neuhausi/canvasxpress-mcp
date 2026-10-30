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
