# SPDX-License-Identifier: AGPL-3.0-only
"""HTTP tests for the server-side /sessions endpoints (REQ-0016).

These use a stub Kindi, so they need no GPU, model pack or tokenizer. The
sessions routes are host-only: they read state_dir/sessions and never touch
the engine.
"""
import json

import pytest
from fastapi.testclient import TestClient

from kindi import server
from kindi.sessions import SessionStore


class _FakeEngine(object):
    ctx = 4096


class _FakeKindi(object):
    cp = None
    supports_logprobs = False

    def __init__(self):
        self.eng = _FakeEngine()
        self.busy = {"active": False, "phase": "", "prompt_tokens": 0, "cached": 0,
                     "read": 0, "generated": 0, "started": 0.0}
        self.stats = {"requests": 0, "prompt_tokens": 0, "cached_tokens": 0,
                      "completion_tokens": 0}
        self.t0 = 0.0

    def head_active(self):
        return "exact"

    def after_turn(self, locked=False):
        return None

    def drafting(self):
        return {"enabled": False, "method": None, "drafter": None, "last": None}

    def progress(self):
        return None


@pytest.fixture
def client(monkeypatch, tmp_path):
    state = tmp_path / "state"
    K = _FakeKindi()
    monkeypatch.setattr(server, "_make_engine", lambda cfg, log: K)
    app = server.create_app({
        "model_name": "bonsai2-2b",
        "api_key": "",
        "pack": str(tmp_path),
        "state_dir": str(state),
    }, log=lambda *a: None)
    with TestClient(app) as c:
        yield c, str(state), SessionStore(str(state / "sessions"))


@pytest.fixture
def auth_client(monkeypatch, tmp_path):
    K = _FakeKindi()
    monkeypatch.setattr(server, "_make_engine", lambda cfg, log: K)
    app = server.create_app({
        "model_name": "bonsai2-2b",
        "api_key": "test-key",
        "state_dir": str(tmp_path / "state"),
    }, log=lambda *a: None)
    with TestClient(app) as c:
        yield c


def test_sessions_empty_list(client):
    c, _, _ = client
    r = c.get("/sessions")
    assert r.status_code == 200, r.text
    assert r.json() == {"sessions": []}


def test_sessions_list_get_filter_and_export(client):
    c, _, store = client
    rec = store.create(title="Demo chat",
                       messages=[("user", "hello"), ("assistant", "hi there")],
                       tags=["agent"])
    listing = c.get("/sessions").json()["sessions"]
    assert [s["id"] for s in listing] == [rec["id"]]
    assert listing[0]["title"] == "Demo chat"
    assert listing[0]["message_count"] == 2

    full = c.get("/sessions/%s" % rec["id"])
    assert full.status_code == 200, full.text
    assert [m["content"] for m in full.json()["messages"]] == ["hello", "hi there"]

    assert c.get("/sessions", params={"status": "open"}).json()["sessions"]
    assert c.get("/sessions", params={"status": "resolved"}).json() == {"sessions": []}
    assert c.get("/sessions", params={"tag": "agent"}).json()["sessions"]
    assert c.get("/sessions", params={"tag": "missing"}).json() == {"sessions": []}

    export = c.get("/sessions/%s/export" % rec["id"])
    assert export.status_code == 200, export.text
    assert export.headers["content-type"].startswith("text/markdown")
    disp = export.headers.get("content-disposition", "")
    assert disp.startswith("attachment;") and "filename=\"" in disp and disp.endswith(".md\"")
    assert "hello" in export.text and "hi there" in export.text
    assert export.text.startswith("# Demo chat")


def test_sessions_create_via_http(client):
    c, _, store = client
    r = c.post("/sessions", json={
        "title": "From API",
        "messages": [{"role": "user", "content": "hello"},
                     {"role": "assistant", "content": "hi there"}],
        "tags": ["api", "Agent"],
        "status": "needs-review",
        "protected": True,
    })
    assert r.status_code == 201, r.text
    rec = r.json()
    assert len(rec["id"]) == 12 and all(ch in "0123456789abcdef" for ch in rec["id"])
    assert rec["title"] == "From API"
    assert rec["status"] == "needs-review"
    assert rec["protected"] is True
    assert rec["unread"] is True
    assert [m["content"] for m in rec["messages"]] == ["hello", "hi there"]
    assert rec["tags"] == ["agent", "api"]

    assert store.get(rec["id"])["title"] == "From API"
    listing = c.get("/sessions").json()["sessions"]
    assert [row["id"] for row in listing] == [rec["id"]]
    assert listing[0]["message_count"] == 2


def test_sessions_create_rejects_invalid_fields_without_writing(client):
    c, _, store = client
    payloads = [
        {"title": 3},
        {"tags": "api"},
        {"status": "closed"},
        {"protected": "yes"},
        {"messages": "hello"},
        {"messages": [{"role": "user", "content": 3}]},
        {"messages": [{"role": "system"}]},
        {"messages": [{"role": "unknown", "content": "x"}]},
        {"messages": ["not an object"]},
    ]
    for payload in payloads:
        r = c.post("/sessions", json=payload)
        assert r.status_code == 400, (payload, r.text)
        assert r.json()["error"]["code"] == "invalid_request_error"
    assert store.list() == []


def test_sessions_status_and_tag_round_trip(client):
    c, _, store = client
    rec = store.create(title="triage me")
    sid = rec["id"]

    bad = c.post("/sessions/%s/status" % sid, json={"status": "closed"})
    assert bad.status_code == 400, bad.text
    assert bad.json()["error"]["code"] == "invalid_request_error"

    ok = c.post("/sessions/%s/status" % sid, json={"status": "needs-review"})
    assert ok.status_code == 200, ok.text
    assert ok.json()["status"] == "needs-review"

    tagged = c.post("/sessions/%s/tags" % sid, json={"tag": "Needs Review!"})
    assert tagged.status_code == 200, tagged.text
    assert tagged.json()["tags"] == ["needs-review"]

    empty = c.post("/sessions/%s/tags" % sid, json={"tag": "  "})
    assert empty.status_code == 400, empty.text

    assert store.get(sid)["status"] == "needs-review"


def test_sessions_unread_and_protected_via_routes(client):
    c, _, store = client
    rec = store.create(title="flagged", messages=[("user", "hello")])
    sid = rec["id"]
    listing = c.get("/sessions").json()["sessions"]
    assert listing[0]["unread"] is True and listing[0]["protected"] is False

    # POST /read clears the flag; no body fields are needed
    cleared = c.post("/sessions/%s/read" % sid)
    assert cleared.status_code == 200, cleared.text
    assert cleared.json()["unread"] is False
    assert store.get(sid)["unread"] is False

    # protection survives the summary route
    store.set_protected(sid, True)
    listing = c.get("/sessions").json()["sessions"]
    assert listing[0]["protected"] is True

    missing = c.post("/sessions/0123456789ab/read")
    assert missing.status_code == 404, missing.text


def test_sessions_missing_and_traversal_ids(client):
    c, _, _ = client
    missing = c.get("/sessions/0123456789ab")
    assert missing.status_code == 404, missing.text
    assert missing.json()["error"]["code"] == "not_found"

    # Ids are the store's 12-hex ids: anything else (including path-shaped input)
    # is refused and never reaches a file read. Traversal attempts are
    # normalised by the client/router into a plain 404 instead.
    for sid in ("not-an-id", "ABCDEF012345"):
        r = c.get("/sessions/%s" % sid)
        assert r.status_code == 400, (sid, r.text)
    for sid in ("../escape", "..%2Fescape"):
        r = c.get("/sessions/%s" % sid)
        assert r.status_code in (400, 404), (sid, r.text)


def test_sessions_concurrent_mutations_do_not_lose_updates(client):
    """Each request builds its own store, so without the shared lock two
    concurrent posts would race their read-modify-write and one update would
    silently win. Both must survive."""
    import threading

    c, _, store = client
    sid = store.create(title="race")["id"]
    errors = []

    def post(path, payload):
        try:
            for _ in range(5):
                r = c.post(path, json=payload)
                assert r.status_code == 200, r.text
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    t1 = threading.Thread(target=post, args=("/sessions/%s/tags" % sid, {"tag": "concurrent"}))
    t2 = threading.Thread(target=post, args=("/sessions/%s/status" % sid, {"status": "resolved"}))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert not errors
    rec = store.get(sid)
    assert rec["status"] == "resolved"
    assert "concurrent" in rec["tags"]


def test_sessions_require_the_api_key(auth_client):
    c = auth_client
    assert c.get("/sessions").status_code == 401
    assert c.get("/sessions/0123456789ab").status_code == 401
    assert c.post("/sessions", json={"title": "blocked"}).status_code == 401
    r = c.get("/sessions", headers={"Authorization": "Bearer test-key"})
    assert r.status_code == 200, r.text
    assert r.json() == {"sessions": []}
