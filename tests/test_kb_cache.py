"""Knowledge-base cache: state-dir location with legacy fallback, write tolerance,
and the CX_KB_BASE_URL override. Offline."""

import os
import subprocess
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

import cx_knowledge  # noqa: E402


def _env_probe(env_extra, expr):
    env = {k: v for k, v in os.environ.items() if k not in ("CX_STATE_DIR", "CX_KB_BASE_URL")}
    env.update(env_extra)
    out = subprocess.run([sys.executable, "-c", f"import sys; sys.path.insert(0, {SRC!r}); "
                          f"import cx_knowledge as k; print({expr})"],
                         capture_output=True, text=True, env=env, check=True)
    return out.stdout.strip().splitlines()[-1]


def test_defaults_are_unchanged():
    assert _env_probe({}, "k._CACHE_DIR == k._LEGACY_CACHE_DIR") == "True"
    assert _env_probe({}, "k._CACHE_DIR.parts[-2:]") == "('data', 'kb_cache')"
    assert _env_probe({}, "k.GITHUB_BASE").endswith("canvasxpress-LLM/main")


def test_state_dir_and_base_url_overrides():
    assert _env_probe({"CX_STATE_DIR": "/srv/state"}, "k._CACHE_DIR") == "/srv/state/kb_cache"
    assert _env_probe({"CX_KB_BASE_URL": "https://mirror.example/kb/"}, "k.GITHUB_BASE") \
        == "https://mirror.example/kb"


@pytest.fixture
def dirs(monkeypatch, tmp_path):
    state, legacy = tmp_path / "state", tmp_path / "legacy"
    monkeypatch.setattr(cx_knowledge, "_CACHE_DIR", state)
    monkeypatch.setattr(cx_knowledge, "_LEGACY_CACHE_DIR", legacy)
    return state, legacy


def test_cached_file_prefers_state_then_legacy(dirs):
    state, legacy = dirs
    legacy.mkdir()
    (legacy / "RULES.md").write_text("old")
    assert cx_knowledge._cached_file("RULES.md") == legacy / "RULES.md"
    state.mkdir()
    (state / "RULES.md").write_text("new")
    assert cx_knowledge._cached_file("RULES.md") == state / "RULES.md"
    assert cx_knowledge._cached_file("MISSING.md") is None


def test_offline_fetch_uses_legacy_cache(dirs, monkeypatch):
    _, legacy = dirs
    legacy.mkdir()
    for name in cx_knowledge._FETCH_FILES:
        (legacy / name).write_text("legacy " + name)
    monkeypatch.setattr(cx_knowledge, "_fetch_url", lambda url, *a, **k: None)
    files = cx_knowledge._fetch_and_cache_files()
    assert files["RULES.md"] == "legacy RULES.md"


def test_unwritable_cache_does_not_break_fetch(dirs, monkeypatch, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")  # a file where a directory is needed -> mkdir fails
    monkeypatch.setattr(cx_knowledge, "_CACHE_DIR", blocker / "kb_cache")
    monkeypatch.setattr(cx_knowledge, "_fetch_url", lambda url, *a, **k: "fresh")
    files = cx_knowledge._fetch_and_cache_files()
    assert files["SCHEMA.md"] == "fresh"
