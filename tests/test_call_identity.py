"""
Tests for the caller identity recorded with each tool call: the real client IP
(first hop of X-Forwarded-For behind the Apache proxy, else the socket peer) and
the hashed X-API-Key id (never the raw key). Pure functions, no server needed.
"""

import hashlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import server  # noqa: E402


def _scope(headers=None, client=("127.0.0.1", 5555)):
    return {
        "type": "http",
        "headers": [(k.encode(), v.encode()) for k, v in (headers or {}).items()],
        "client": client,
    }


def test_client_ip_prefers_first_forwarded_hop():
    # Apache appends the caller; upstream proxies may already have added hops.
    s = _scope({"X-Forwarded-For": "203.0.113.9, 10.0.0.2"})
    assert server._client_ip(s) == "203.0.113.9"


def test_client_ip_header_is_case_insensitive():
    s = _scope({"x-forwarded-for": "198.51.100.7"})
    assert server._client_ip(s) == "198.51.100.7"


def test_client_ip_falls_back_to_real_ip_then_socket_peer():
    assert server._client_ip(_scope({"X-Real-IP": "192.0.2.4"})) == "192.0.2.4"
    assert server._client_ip(_scope()) == "127.0.0.1"          # direct local call
    assert server._client_ip(_scope(client=None)) is None


def test_api_key_id_is_a_hash_prefix_never_the_key():
    s = _scope({"X-API-Key": "super-secret-key"})
    kid = server._api_key_id(s)
    assert kid == hashlib.sha256(b"super-secret-key").hexdigest()[:12]
    assert "super-secret-key" not in kid and len(kid) == 12


def test_api_key_id_none_when_absent():
    assert server._api_key_id(_scope()) is None
    assert server._api_key_id(_scope({"X-API-Key": "   "})) is None


def test_call_log_records_ip_and_key_id(tmp_path):
    log = server._CallLog(str(tmp_path / "calls.db"))
    log.log("id1", "generate", "/generate", {"description": "x"}, {"ok": True}, 200,
            ip="203.0.113.9", api_key_id="abc123def456")
    log.log("id2", "generate", "/generate", {}, {}, 200)   # identity optional
    import sqlite3
    con = sqlite3.connect(str(tmp_path / "calls.db"))
    rows = dict(con.execute("SELECT id, ip || '|' || COALESCE(api_key_id,'') FROM tool_calls").fetchall())
    assert rows["id1"] == "203.0.113.9|abc123def456"
    assert rows["id2"] is None or rows["id2"] == "|"  # NULL ip -> NULL concat
