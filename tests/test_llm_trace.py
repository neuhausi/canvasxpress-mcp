"""
Tests for the per-request LLM call trace (llm_providers.begin/end_llm_trace and
server._attach_call_trace).

Offline: the provider dispatch is replaced with a fake, so no API is called.
"""

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import llm_providers as lp  # noqa: E402


_USAGE = {
    "input_tokens": 120, "output_tokens": 30, "stop_reason": "end_turn",
    "cache_read_input_tokens": 4000, "cache_creation_input_tokens": 0,
}


def _fake_ok(**kwargs):
    return "MODEL OUTPUT", dict(_USAGE)


def _fake_boom(**kwargs):
    raise RuntimeError("provider down")


@pytest.fixture
def fake_provider(monkeypatch):
    monkeypatch.setitem(lp._DISPATCH, lp.PROVIDER, _fake_ok)
    yield


def test_no_trace_outside_request(fake_provider):
    # Without begin_llm_trace() nothing is recorded and complete() is unchanged.
    text, usage = lp.complete("SYS", "USER")
    assert text == "MODEL OUTPUT" and usage["input_tokens"] == 120
    assert lp.end_llm_trace() == []


def test_trace_records_verbatim_prompt_timing_and_tokens(fake_provider):
    lp.begin_llm_trace()
    system = "STABLE SYSTEM PROMPT " * 50
    user = 'Similar examples...\n---\nGenerate the CanvasXpress config for:\n"a bar chart"'
    text, usage = lp.complete(system, user, temperature=0.2, max_tokens=99,
                              system_suffix="graph-type snippet")
    trace = lp.end_llm_trace()

    assert text == "MODEL OUTPUT"
    assert len(trace) == 1
    rec = trace[0]
    # The variable text is stored verbatim; the stable system prompt by size+hash.
    assert rec["user_prompt"] == user
    assert rec["system_suffix"] == "graph-type snippet"
    assert rec["system_chars"] == len(system)
    assert len(rec["system_sha1"]) == 12 and "system" not in rec
    # Call metadata + tokens.
    assert rec["ok"] is True and rec["ms"] >= 0
    assert rec["provider"] == lp.PROVIDER and rec["model"] == lp.MODEL
    assert rec["temperature"] == 0.2 and rec["max_tokens"] == 99
    assert rec["input_tokens"] == 120 and rec["output_tokens"] == 30
    assert rec["cache_read_input_tokens"] == 4000 and rec["stop_reason"] == "end_turn"
    # The trace is cleared after end.
    assert lp.end_llm_trace() == []


def test_trace_records_two_calls_in_order(fake_provider):
    lp.begin_llm_trace()
    lp.complete("S", "first")
    lp.complete("S", "second")
    prompts = [r["user_prompt"] for r in lp.end_llm_trace()]
    assert prompts == ["first", "second"]


def test_failed_call_is_recorded_and_reraised(monkeypatch):
    monkeypatch.setitem(lp._DISPATCH, lp.PROVIDER, _fake_boom)
    lp.begin_llm_trace()
    with pytest.raises(RuntimeError, match="provider down"):
        lp.complete("S", "U")
    trace = lp.end_llm_trace()
    assert len(trace) == 1
    assert trace[0]["ok"] is False
    assert "RuntimeError: provider down" in trace[0]["error"]
    assert trace[0]["user_prompt"] == "U"


def test_attach_call_trace_adds_timing_and_cost(fake_provider):
    # server import is expensive; keep it to the one test that needs it.
    import server  # noqa: WPS433

    lp.begin_llm_trace()
    lp.complete("S", "U")
    started = time.perf_counter() - 0.25  # pretend the request took ~250 ms
    resp = server._attach_call_trace({"success": True}, started, lp.end_llm_trace())

    assert resp["success"] is True
    assert resp["timing"]["duration_ms"] >= 250
    assert len(resp["llm_calls"]) == 1
    call = resp["llm_calls"][0]
    assert call["user_prompt"] == "U"
    assert isinstance(call["cost_usd"], float) and call["cost_usd"] >= 0.0
    # Cost follows the configured rates: with a priced model it must be > 0.
    if server._rates() != (0.0, 0.0):
        assert call["cost_usd"] > 0.0


def test_attach_call_trace_without_llm_calls():
    import server

    resp = server._attach_call_trace({}, time.perf_counter(), [])
    assert resp["llm_calls"] == []
    assert "duration_ms" in resp["timing"]
