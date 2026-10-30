"""Hosting options: Bedrock temperature retry, /health, retrieval readiness,
CX_DISABLED_TOOLS and the build_index.py metadata. All offline."""

import asyncio
import json
import os
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

import llm_providers  # noqa: E402
import server  # noqa: E402


# --- Bedrock temperature -------------------------------------------------------

class _FakeBedrock:
    def __init__(self, reject_temperature):
        self.reject = reject_temperature
        self.calls = []

    def converse(self, modelId, **body):
        cfg = body["inferenceConfig"]
        self.calls.append(dict(cfg))
        if self.reject and "temperature" in cfg:
            raise RuntimeError("ValidationException: temperature is deprecated for this model")
        return {
            "output": {"message": {"content": [{"text": "ok"}]}},
            "usage": {"inputTokens": 1, "outputTokens": 1},
            "stopReason": "end_turn",
        }


@pytest.fixture(autouse=True)
def clean_no_temperature():
    llm_providers._BEDROCK_NO_TEMPERATURE.clear()
    yield
    llm_providers._BEDROCK_NO_TEMPERATURE.clear()


def test_bedrock_retries_without_temperature_and_remembers(monkeypatch):
    fake = _FakeBedrock(reject_temperature=True)
    monkeypatch.setattr(llm_providers, "_get_bedrock", lambda: fake)
    text, _ = llm_providers._complete_bedrock("s", "u", "m5", 0.0, 100)
    assert text == "ok"
    assert "temperature" in fake.calls[0] and "temperature" not in fake.calls[1]
    llm_providers._complete_bedrock("s", "u", "m5", 0.0, 100)
    assert "temperature" not in fake.calls[2]  # no second rejected attempt


def test_bedrock_keeps_temperature_when_accepted(monkeypatch):
    fake = _FakeBedrock(reject_temperature=False)
    monkeypatch.setattr(llm_providers, "_get_bedrock", lambda: fake)
    llm_providers._complete_bedrock("s", "u", "m4", 0.3, 100)
    assert fake.calls == [{"maxTokens": 100, "temperature": 0.3}]


def test_bedrock_unrelated_errors_still_raise(monkeypatch):
    class Boom(_FakeBedrock):
        def converse(self, modelId, **body):
            raise RuntimeError("AccessDeniedException")
    monkeypatch.setattr(llm_providers, "_get_bedrock", lambda: Boom(False))
    with pytest.raises(RuntimeError, match="AccessDenied"):
        llm_providers._complete_bedrock("s", "u", "m", 0.0, 100)


# --- /health and retrieval readiness --------------------------------------------

def test_health_reports_retrieval_state(monkeypatch):
    monkeypatch.setattr(server, "_use_vector_index", False)
    monkeypatch.setattr(server, "_retrieval_reason", "index not found")
    body = json.loads(asyncio.run(server.rest_health(None)).body)
    assert body["status"] == "ok" and body["retrieval"] == "fallback"
    assert body["retrieval_reason"] == "index not found"
    monkeypatch.setattr(server, "_use_vector_index", True)
    monkeypatch.setattr(server, "_retrieval_reason", "")
    body = json.loads(asyncio.run(server.rest_health(None)).body)
    assert body["retrieval"] == "vector" and "retrieval_reason" not in body


def test_missing_index_falls_back_with_reason(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "DB_FILE", tmp_path / "nope.db")
    assert server._load_vector_index() is False
    assert "not found" in server._retrieval_reason


def test_empty_probe_falls_back(monkeypatch, tmp_path):
    db = tmp_path / "e.db"
    db.write_bytes(b"")
    monkeypatch.setattr(server, "DB_FILE", db)
    monkeypatch.setattr(server, "SentenceTransformer", lambda name: object())
    monkeypatch.setattr(server, "_vector_retrieve", lambda q, k: [])
    assert server._load_vector_index() is False
    assert server._use_vector_index is False
    assert "no rows" in server._retrieval_reason


# --- CX_DISABLED_TOOLS -------------------------------------------------------------

def _has_tool(name):
    return name in [t.name for t in asyncio.run(server.mcp.list_tools())]


def test_disabled_tools_removes_named_tool(monkeypatch):
    @server.mcp.tool(name="zz_tmp_tool")
    def zz_tmp_tool() -> str:
        return "x"

    assert _has_tool("zz_tmp_tool")
    monkeypatch.setenv("CX_DISABLED_TOOLS", "zz_tmp_tool")
    server._apply_disabled_tools()
    assert not _has_tool("zz_tmp_tool")


def test_unknown_disabled_tool_warns_or_fails_when_strict(monkeypatch):
    monkeypatch.setenv("CX_DISABLED_TOOLS", "no_such_tool")
    server._apply_disabled_tools()  # warns only
    monkeypatch.setenv("CX_STRICT_TOOLS", "1")
    with pytest.raises(RuntimeError, match="no_such_tool"):
        server._apply_disabled_tools()


# --- build_index.py ------------------------------------------------------------------

def test_build_index_records_meta_and_honors_flags(monkeypatch, tmp_path):
    pytest.importorskip("sqlite_vec")
    import build_index

    class FakeModel:
        def __init__(self, name): pass
        def get_sentence_embedding_dimension(self): return 4
        def encode(self, texts, **kw):
            import numpy as np
            return np.ones((len(texts), 4), dtype="float32")

    monkeypatch.setattr(build_index, "SentenceTransformer", FakeModel)
    examples = tmp_path / "ex.json"
    examples.write_text(json.dumps([{"type": "Bar", "description": "a bar chart",
                                     "config": {"graphType": "Bar"}}]))
    out = tmp_path / "out.db"
    monkeypatch.setattr(sys, "argv", ["build_index.py", "--examples", str(examples),
                                      "--output", str(out), "--model", "fake-model"])
    build_index.main()
    db = build_index._connect_with_sqlite_vec(str(out))
    meta = dict(db.execute("SELECT key, value FROM meta").fetchall())
    db.close()
    assert meta["model"] == "fake-model" and meta["dim"] == "4" and meta["count"] == "1"
    assert len(meta["corpus_sha256"]) == 64
