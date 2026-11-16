# Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only. See LICENSE and COMMERCIAL-LICENSE.md.
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import sqlite3
import time
import uuid


def state_dir() -> Path:
    path = Path(os.environ.get("BRIDGE_STATE", "~/.local/state/kindi-bridge")).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


class Store:
    def __init__(self, path=None):
        self.path = Path(path) if path else state_dir() / "bridge.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.db() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS agents(
              id TEXT PRIMARY KEY, name TEXT UNIQUE NOT NULL, cwd TEXT NOT NULL,
              command TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'offline',
              pid INTEGER, heartbeat REAL, pane TEXT, collection TEXT);
            CREATE TABLE IF NOT EXISTS jobs(
              id TEXT PRIMARY KEY, agent TEXT NOT NULL REFERENCES agents(id),
              prompt TEXT NOT NULL, due REAL NOT NULL, state TEXT NOT NULL DEFAULT 'scheduled',
              kind TEXT NOT NULL DEFAULT 'prompt', task TEXT, interval REAL NOT NULL DEFAULT 0,
              max_checks INTEGER NOT NULL DEFAULT 1, attempts INTEGER NOT NULL DEFAULT 0,
              deadline REAL, accepted REAL, finished REAL, response TEXT, error TEXT,
              created REAL NOT NULL, previous_state TEXT);
            CREATE INDEX IF NOT EXISTS jobs_due ON jobs(state,due);
            CREATE TABLE IF NOT EXISTS events(
              id INTEGER PRIMARY KEY, at REAL NOT NULL, agent TEXT, job TEXT,
              kind TEXT NOT NULL, detail TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS collections(
              id TEXT PRIMARY KEY, root TEXT NOT NULL UNIQUE, state TEXT NOT NULL,
              updated REAL, error TEXT, semantic INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS documents(
              id TEXT PRIMARY KEY, collection TEXT NOT NULL REFERENCES collections(id),
              path TEXT NOT NULL, hash TEXT, error TEXT, UNIQUE(collection,path));
            CREATE TABLE IF NOT EXISTS chunks(
              id INTEGER PRIMARY KEY, document TEXT NOT NULL REFERENCES documents(id),
              source TEXT NOT NULL, text TEXT NOT NULL, vector TEXT);
            CREATE VIRTUAL TABLE IF NOT EXISTS search USING fts5(text, content='chunks', content_rowid='id');
            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
              INSERT INTO search(rowid,text) VALUES(new.id,new.text); END;
            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
              INSERT INTO search(search,rowid,text) VALUES('delete',old.id,old.text); END;
            """)

    @contextlib.contextmanager
    def db(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA foreign_keys=ON")
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def rows(self, sql, args=()):
        with self.db() as db:
            return [dict(r) for r in db.execute(sql, args)]

    def execute(self, sql, args=()):
        with self.db() as db:
            return db.execute(sql, args).rowcount

    def event(self, kind, detail, agent=None, job=None):
        self.execute("INSERT INTO events(at,agent,job,kind,detail) VALUES(?,?,?,?,?)",
                     (time.time(), agent, job, kind, str(detail)[-20000:]))

    def setting(self, key, value=None):
        if value is not None:
            self.execute("INSERT INTO settings VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
        r = self.rows("SELECT value FROM settings WHERE key=?", (key,))
        return r[0]["value"] if r else None

    def agent(self, name=None):
        name = name or self.setting("default_agent")
        r = self.rows("SELECT * FROM agents WHERE id=? OR name=?", (name, name))
        if not r:
            raise ValueError("Choose an agent first (agent add NAME --dir PATH)")
        return r[0]

    def add_agent(self, name, cwd, command):
        path = Path(cwd).expanduser().resolve(strict=True)
        if not path.is_dir():
            raise ValueError("Agent working path must be a directory")
        ident = uuid.uuid4().hex
        self.execute("INSERT INTO agents(id,name,cwd,command) VALUES(?,?,?,?)", (ident, name, str(path), json.dumps(command)))
        if not self.setting("default_agent"):
            self.setting("default_agent", ident)
        return self.agent(ident)

    def schedule(self, prompt, due, agent=None, interval=0, limit=1, task=None, deadline=None, kind="prompt"):
        target = self.agent(agent)
        if not prompt.strip() or limit < 1 or interval < 0:
            raise ValueError("Nonempty prompt, positive limit and nonnegative interval required")
        if (kind == "check" or limit > 1) and interval <= 0:
            raise ValueError("Repeated jobs require a positive interval")
        if kind == "check" and not (task or "").strip():
            raise ValueError("Completion checks require a task objective/identity")
        ident = uuid.uuid4().hex
        self.execute("""INSERT INTO jobs(id,agent,prompt,due,kind,task,interval,max_checks,deadline,created)
                     VALUES(?,?,?,?,?,?,?,?,?,?)""", (ident, target["id"], prompt, due, kind, task, interval, limit, deadline, time.time()))
        self.event("scheduled", prompt, target["id"], ident)
        return ident

    def delete_job(self, ident):
        # The worker claims in one transaction; deletion cannot race an already claimed delivery.
        return self.execute("""UPDATE jobs SET previous_state=state,state='deleted'
                           WHERE id=? AND state IN ('scheduled','paused')""", (ident,))

    def undo_delete(self, ident):
        return self.execute("UPDATE jobs SET state=previous_state WHERE id=? AND state='deleted'", (ident,))
