# SPDX-License-Identifier: AGPL-3.0-only
"""Chat session store: record, triage, tags, raw-markdown export/import."""
import pytest

from kindi.sessions import SessionStore, session_id_message_hash, slug


def test_record_list_and_get(tmp_path):
    st = SessionStore(str(tmp_path / "sessions"))
    rec = st.create(title="Hello world", messages=[("user", "hi"), ("assistant", "hello")])
    assert st.get(rec["id"])["messages"][0]["content"] == "hi"
    assert [r["id"] for r in st.list()] == [rec["id"]]


def test_triage_status_transitions(tmp_path):
    st = SessionStore(str(tmp_path))
    a = st.create(title="a")
    b = st.create(title="b")
    st.set_status(b, "needs-review")
    assert {r["id"] for r in st.triage()} == {a["id"], b["id"]}
    st.set_status(a, "resolved")
    assert [r["id"] for r in st.triage()] == [b["id"]]
    with pytest.raises(ValueError, match="status must be"):
        st.set_status(a, "archived")


def test_tags_slugified_deduped_and_filterable(tmp_path):
    st = SessionStore(str(tmp_path))
    rec = st.create(title="My Chat!", tags=["Work Notes", "work-notes", ""])
    assert rec["tags"] == ["work-notes"]
    st.add_tag(rec, "Bug Fix")
    assert st.get(rec["id"])["tags"] == ["bug-fix", "work-notes"]
    st.remove_tag(rec, "work notes")
    assert st.get(rec["id"])["tags"] == ["bug-fix"]
    assert [r["id"] for r in st.list(tag="Bug Fix")] == [rec["id"]]


def test_export_is_raw_markdown_and_import_round_trips(tmp_path):
    st = SessionStore(str(tmp_path))
    raw_user = "line one\n  indented line\n\nparagraph two"
    rec = st.create(title="My chat", messages=[("user", raw_user), ("assistant", "answer")])
    st.add_tag(rec, "docs")
    text = st.export_markdown(rec)
    assert "# My chat" in text and raw_user in text  # verbatim, never reflowed
    path = st.export_file(rec, str(tmp_path / "out"))
    assert path.endswith("my-chat.md") or "my-chat" in path
    back = st.import_markdown(text)
    assert back["title"] == "My chat"
    assert [m["content"] for m in back["messages"]] == [raw_user, "answer"]
    assert back["tags"] == ["docs"]


def test_two_same_day_sessions_with_one_title_do_not_share_an_export(tmp_path):
    """The export name is date + slug, so a second session titled the same on the same day used to
    overwrite the first one's file; the id now disambiguates it."""
    st = SessionStore(str(tmp_path))
    first = st.create(title="notes", messages=[("user", "first")])
    second = st.create(title="notes", messages=[("user", "second")])
    out = str(tmp_path / "exports")
    pa = st.export_file(first, out)
    pb = st.export_file(second, out)
    assert pa != pb
    assert "first" in open(pa, encoding="utf-8").read()
    assert "second" in open(pb, encoding="utf-8").read()
    assert st.export_file(first, out) == pa          # a re-export still rewrites its own file


def test_sessions_cli_list_export_round_trip(tmp_path, capsys):
    from kindi.__main__ import main
    root = str(tmp_path)
    assert main(["--state-dir", root, "sessions", "list"]) == 0
    assert "no sessions" in capsys.readouterr().out
    import os
    st = SessionStore(os.path.join(root, "sessions"))
    rec = st.create(title="CLI demo", messages=[("user", "q")])
    assert main(["--state-dir", root, "sessions", "list"]) == 0
    assert rec["id"] in capsys.readouterr().out
    assert main(["--state-dir", root, "sessions", "export", "--out", str(tmp_path / "out"), rec["id"]]) == 0
    assert list((tmp_path / "out").glob("*.md"))


def test_append_messages_reuses_the_session(tmp_path):
    st = SessionStore(str(tmp_path))
    rec = st.create(title="Chat", messages=[("user", "first")])
    before = st.get(rec["id"])                                                # noqa: E501
    st.append_messages(rec["id"], [("assistant", "answer"), ("user", "follow-up")])
    after = st.get(rec["id"])                                                # noqa: E501
    assert [m["content"] for m in after["messages"]] == ["first", "answer", "follow-up"]
    assert after["updated"] >= before["updated"]
    with pytest.raises(ValueError, match="unknown role"):
        st.append_messages(rec["id"], [("narrator", "boo")])


def test_chat_resume_seed_and_append_round_trip(tmp_path, monkeypatch):
    from kindi import __main__ as cli
    cfg = {"state_dir": str(tmp_path / "state")}
    st = SessionStore(str(tmp_path / "state" / "sessions"))
    rec = st.create(title="Chat", messages=[("user", "first question"), ("assistant", "old answer")])
    seeded_rec, msgs = cli._resume_chat(cfg, rec["id"])
    assert seeded_rec["id"] == rec["id"]
    assert msgs == [{"role": "user", "content": "first question"},
                    {"role": "assistant", "content": "old answer"}]
    msgs += [{"role": "user", "content": "next question"}, {"role": "assistant", "content": "new answer"}]
    cli._append_chat(cfg, rec["id"], msgs[2:])
    contents = [m["content"] for m in st.get(rec["id"])["messages"]]          # noqa: E501
    assert contents == ["first question", "old answer", "next question", "new answer"]
    # a malformed or unknown id refuses loudly instead of seeding an empty chat
    with pytest.raises(SystemExit, match="12 hex characters"):
        cli._resume_chat(cfg, "../etc/passwd")
    with pytest.raises(SystemExit, match="no session"):
        cli._resume_chat(cfg, "0" * 12)


def _backdate(st, sid, days):
    import datetime
    import json as _json
    rec_path = st.path(sid)
    data = _json.loads(open(rec_path, encoding="utf-8").read())
    when = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)
    data["updated"] = when.isoformat(timespec="seconds")
    open(rec_path, "w", encoding="utf-8").write(_json.dumps(data, indent=2, sort_keys=True) + "\n")


def test_prune_only_removes_old_resolved_by_default(tmp_path):
    st = SessionStore(str(tmp_path))
    old_resolved = st.create(title="old resolved")
    st.set_status(old_resolved["id"], "resolved")
    _backdate(st, old_resolved["id"], 60)
    old_open = st.create(title="old open")
    _backdate(st, old_open["id"], 60)
    fresh = st.create(title="fresh")

    would = st.prune(older_than_days=30, dry_run=True)
    assert [r["id"] for r in would] == [old_resolved["id"]]
    assert st.get(old_resolved["id"]) is not None         # dry run deletes nothing
    gone = st.prune(older_than_days=30)
    assert [r["id"] for r in gone] == [old_resolved["id"]]
    assert sorted(r["id"] for r in st.list()) == sorted([old_open["id"], fresh["id"]])
    # --all widens the scope to every status once its file is old enough
    gone_all = st.prune(older_than_days=30, statuses=None)
    assert [r["id"] for r in gone_all] == [old_open["id"]]
    with pytest.raises(ValueError, match="non-negative"):
        st.prune(older_than_days=-1)


def test_sessions_prune_cli(tmp_path, capsys):
    from kindi.__main__ import main
    root = str(tmp_path)
    st = SessionStore(root + "/sessions")
    rec = st.create(title="old resolved")
    st.set_status(rec["id"], "resolved")
    _backdate(st, rec["id"], 60)
    assert main(["--state-dir", root, "sessions", "prune", "--days", "30", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "would prune 1 session(s)" in out and rec["id"] in out
    assert main(["--state-dir", root, "sessions", "prune", "--days", "30"]) == 0
    assert "pruned 1 session(s)" in capsys.readouterr().out
    assert st.list() == []


def test_store_stats(tmp_path):
    st = SessionStore(str(tmp_path))
    empty = st.stats()
    assert empty["sessions"] == 0 and empty["messages"] == 0 and empty["bytes"] == 0
    assert empty["oldest_updated"] is None and empty["newest_updated"] is None
    a = st.create(title="a", messages=[("user", "x"), ("assistant", "y")])
    b = st.create(title="b")
    st.set_status(b["id"], "resolved")
    got = st.stats()
    assert got["sessions"] == 2
    assert got["by_status"] == {"open": 1, "needs-review": 0, "resolved": 1}
    assert got["messages"] == 2
    assert got["bytes"] > 0
    assert got["oldest_updated"] <= got["newest_updated"]


def test_sessions_stats_cli_text_and_json(tmp_path, capsys):
    from kindi.__main__ import main
    import json as _json
    root = str(tmp_path)
    st = SessionStore(root + "/sessions")
    st.create(title="one", messages=[("user", "hi")])
    st.create(title="two")
    assert main(["--state-dir", root, "sessions", "stats"]) == 0
    out = capsys.readouterr().out
    assert "sessions: 2 (open 2, needs-review 0, resolved 0, unread 2, protected 0)" in out
    assert "messages: 1" in out
    assert main(["--state-dir", root, "sessions", "stats", "--json"]) == 0
    payload = _json.loads(capsys.readouterr().out)
    assert payload["sessions"] == 2 and payload["messages"] == 1 and payload["bytes"] > 0


def test_protected_records_survive_prune_and_the_protect_cli_toggles(tmp_path, capsys):
    from kindi.__main__ import main
    root = str(tmp_path)
    st = SessionStore(root + "/sessions")
    rec = st.create(title="precious", status="resolved")
    _backdate(st, rec["id"], 60)
    # protect first, then the pruner must keep it (and say so in the dry run)
    assert main(["--state-dir", root, "sessions", "protect", rec["id"]]) == 0
    assert "protected" in capsys.readouterr().out
    assert st.get(rec["id"])["protected"] is True
    would = st.prune(older_than_days=30, dry_run=True)
    assert len(would) == 1 and would[0]["protected"] is True
    gone = st.prune(older_than_days=30)
    assert len(gone) == 1 and gone[0]["protected"] is True
    assert st.get(rec["id"]) is not None       # kept
    # --all is not a bypass: protection still wins
    st.prune(older_than_days=30, statuses=None)
    assert st.get(rec["id"]) is not None
    # unprotect turns the record back into ordinary training data
    assert main(["--state-dir", root, "sessions", "protect", rec["id"], "--off"]) == 0
    assert st.get(rec["id"])["protected"] is False
    st.prune(older_than_days=30)
    with pytest.raises(KeyError):
        st.get(rec["id"])


def test_unread_flag_raised_by_new_turns_and_cleared_by_show(tmp_path, capsys):
    from kindi.__main__ import main
    root = str(tmp_path)
    st = SessionStore(root + "/sessions")
    rec = st.create(title="fresh", messages=[("user", "hi")])
    assert rec["unread"] is True                       # new transcript: not yet read
    st.mark_read(rec["id"])
    assert st.get(rec["id"])["unread"] is False
    st.append_messages(rec["id"], [("assistant", "reply")])
    assert st.get(rec["id"])["unread"] is True        # new turns raise it again
    stats = st.stats()
    assert stats["unread"] == 1 and stats["protected"] == 0
    # the CLI list shows the marker
    assert main(["--state-dir", root, "sessions", "list"]) == 0
    assert "[unread]" in capsys.readouterr().out
    # show clears it
    assert main(["--state-dir", root, "sessions", "show", rec["id"]]) == 0
    assert st.get(rec["id"])["unread"] is False
    # protected shows its own marker
    st.set_protected(rec["id"], True)
    assert main(["--state-dir", root, "sessions", "list"]) == 0
    assert "[protected]" in capsys.readouterr().out


def test_slug_and_hash(tmp_path):
    assert slug("  Hello, World!  ") == "hello-world"
    assert slug("") == "session"
    st = SessionStore(str(tmp_path))
    rec = st.create(messages=[("user", "x")])
    assert session_id_message_hash(rec) == session_id_message_hash(st.get(rec["id"]))
