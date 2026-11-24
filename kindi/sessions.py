# SPDX-License-Identifier: AGPL-3.0-only
"""Chat session history: a host-only store for `kindi chat` turns.

Sessions live as one JSON file each under `<state_dir>/sessions/`. A session
has a title, an ordered message list, tags, and a triage status
(`open` -> `needs-review` -> `resolved`). Export is raw markdown: the exact
message text, verbatim, with a small header — no rewriting, no reflowing —
and `import_markdown` reads that same format back.
"""
import datetime
import hashlib
import json
import os
import re
import uuid

STATUSES = ("open", "needs-review", "resolved")


def slug(title, fallback="session"):
    """A safe lowercase dash slug for filenames and titles."""
    s = re.sub(r"[^a-z0-9]+", "-", str(title or "").lower()).strip("-")
    return (s or fallback)[:60]


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _parse_when(value):
    """Parse the store's ISO timestamp; 3.6 has no datetime.fromisoformat."""
    text = str(value)
    try:
        return datetime.datetime.fromisoformat(text)
    except AttributeError:
        return datetime.datetime.strptime(text, "%Y-%m-%dT%H:%M:%S%z")


class SessionStore(object):
    """One JSON file per session in a directory; no GPU, no pack needed."""

    def __init__(self, root):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)

    def path(self, sid):
        return os.path.join(self.root, str(sid) + ".json")

    # ---- creation and messages ----
    def create(self, title="", messages=None, tags=None, status="open", protected=False):
        if status not in STATUSES:
            raise ValueError("status must be one of %s (got %r)" % (STATUSES, status))
        sid = uuid.uuid4().hex[:12]
        rec = {"id": sid, "title": str(title or "untitled"), "created": _now(),
               "updated": _now(), "messages": [], "tags": sorted({slug(t) for t in (tags or []) if slug(t) != "session"}),
               "status": status, "protected": bool(protected), "unread": True}
        for role, content in (messages or []):
            self.add_message(rec, role, content)
        self._save(rec)
        return rec

    def add_message(self, rec_or_id, role, content):
        rec = self._coerce(rec_or_id)
        if role not in ("system", "user", "assistant", "tool"):
            raise ValueError("unknown role %r" % role)
        rec["messages"].append({"role": role, "content": str(content), "ts": _now()})
        rec["unread"] = True
        rec["updated"] = _now()
        self._save(rec)
        return rec

    def mark_read(self, rec_or_id):
        """Clear the unread flag (a session that was just shown)."""
        rec = self._coerce(rec_or_id)
        if rec.get("unread"):
            rec["unread"] = False
            self._save(rec)
        return rec

    def set_protected(self, rec_or_id, on=True):
        """Protect a record from destructive ops (prune treats it like a delete-refusal)."""
        rec = self._coerce(rec_or_id)
        rec["protected"] = bool(on)
        self._save(rec)
        return rec

    def append_messages(self, rec_or_id, messages):
        """Append new turns to a stored session and save once (`kindi chat --resume` reuses the id)."""
        rec = self._coerce(rec_or_id)
        for role, content in (messages or []):
            if role not in ("system", "user", "assistant", "tool"):
                raise ValueError("unknown role %r" % role)
        for role, content in (messages or []):
            rec["messages"].append({"role": role, "content": str(content), "ts": _now()})
        rec["unread"] = True
        rec["updated"] = _now()
        self._save(rec)
        return rec

    def stats(self):
        """Store-level visibility for retention decisions: counts by status, message totals,
        bytes on disk, and the oldest/newest last-update (None when the store is empty)."""
        rows = self.list()
        by_status = {s: 0 for s in STATUSES}
        for r in rows:
            if r.get("status") in by_status:
                by_status[r["status"]] += 1
        try:
            total_bytes = sum(os.path.getsize(self.path(r["id"])) for r in rows)
        except OSError:
            total_bytes = 0
        updates = sorted(r.get("updated") for r in rows if r.get("updated"))
        return {"sessions": len(rows), "by_status": by_status,
                "messages": sum(len(r.get("messages") or []) for r in rows),
                "bytes": total_bytes,
                "unread": sum(1 for r in rows if r.get("unread")),
                "protected": sum(1 for r in rows if r.get("protected")),
                "oldest_updated": updates[0] if updates else None,
                "newest_updated": updates[-1] if updates else None}

    # ---- retention ----
    def prune(self, older_than_days=30, statuses=("resolved",), dry_run=False):
        """Delete (or list) sessions whose last update is older than the cutoff.

        By default only ``resolved`` sessions go; pass ``statuses`` to widen
        the scope (``None`` means any status). Returns the pruned client-side
        records (id, title, status, updated) either way.
        """
        if older_than_days is None or int(older_than_days) < 0:
            raise ValueError("older_than_days must be a non-negative number")
        cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=int(older_than_days))
        out = []
        for rec in self.list():
            if statuses is not None and rec.get("status") not in statuses:
                continue
            try:
                when = _parse_when(rec.get("updated") or "")
            except ValueError:
                continue          # an unparseable timestamp is not evidence it is old
            if when < cutoff:
                row = {"id": rec["id"], "title": rec.get("title", "untitled"),
                       "status": rec.get("status"), "updated": rec.get("updated")}
                if rec.get("protected"):
                    # protected records survive pruning; the row still reports them so the
                    # operator sees what was kept (and why)
                    row["protected"] = True
                    out.append(row)
                    continue
                out.append(row)
                if not dry_run:
                    os.remove(self.path(rec["id"]))
        return out

    # ---- triage and tags ----
    def set_status(self, rec_or_id, status):
        if status not in STATUSES:
            raise ValueError("status must be one of %s (got %r)" % (STATUSES, status))
        rec = self._coerce(rec_or_id)
        rec["status"] = status
        rec["updated"] = _now()
        self._save(rec)
        return rec

    def add_tag(self, rec_or_id, tag):
        rec = self._coerce(rec_or_id)
        t = slug(tag) if str(tag or "").strip() else ""
        if t and t not in rec["tags"]:
            rec["tags"].append(t)
            rec["tags"].sort()
            rec["updated"] = _now()
            self._save(rec)
        return rec

    def remove_tag(self, rec_or_id, tag):
        rec = self._coerce(rec_or_id)
        t = slug(tag) if str(tag or "").strip() else ""
        if t in rec["tags"]:
            rec["tags"].remove(t)
            rec["updated"] = _now()
            self._save(rec)
        return rec

    # ---- lookup ----
    def get(self, sid):
        p = self.path(sid)
        if not os.path.isfile(p):
            raise KeyError("no session %r in %s" % (sid, self.root))
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    def list(self, status=None, tag=None):
        out = []
        if not os.path.isdir(self.root):
            return out
        for name in sorted(os.listdir(self.root)):
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(self.root, name), encoding="utf-8") as f:
                    rec = json.load(f)
            except (OSError, ValueError):
                continue
            if status is not None and rec.get("status") != status:
                continue
            if tag is not None and slug(tag) not in (rec.get("tags") or []):
                continue
            out.append(rec)
        out.sort(key=lambda r: (r.get("updated") or "", r.get("id") or ""), reverse=True)
        return out

    def triage(self):
        """Sessions still needing attention: open or needs-review, newest first."""
        return [r for r in self.list() if r.get("status") in ("open", "needs-review")]

    # ---- raw-markdown export / import ----
    def export_markdown(self, rec_or_id):
        rec = self._coerce(rec_or_id)
        lines = ["# %s" % rec["title"], "",
                 "- id: %s" % rec["id"],
                 "- created: %s" % rec["created"],
                 "- status: %s" % rec["status"],
                 "- tags: %s" % (", ".join(rec.get("tags") or []) or "-"), ""]
        for m in rec.get("messages") or []:
            lines.append("## %s" % m["role"].capitalize())
            lines.append("")
            lines.append(m["content"])
            lines.append("")
        return "\n".join(lines)

    def export_file(self, rec_or_id, directory):
        rec = self._coerce(rec_or_id)
        os.makedirs(directory, exist_ok=True)
        stem = "%s-%s" % (rec["created"][:10], slug(rec["title"]) or rec["id"])
        path = os.path.join(directory, stem + ".md")
        if os.path.isfile(path) and self._exported_id(path) != rec["id"]:
            # two sessions with the same title on the same day used to overwrite each other's
            # export; the id disambiguates the second one (re-exporting a session still
            # rewrites its own file)
            path = os.path.join(directory, "%s-%s.md" % (stem, rec["id"]))
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.export_markdown(rec))
        return path

    @staticmethod
    def _exported_id(path):
        """The `- id:` line of an exported file, or None: tells a name collision from a re-export."""
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.startswith("- id: "):
                        return line[6:].strip()
        except OSError:
            return None
        return None

    def import_markdown(self, text):
        """Parse the format `export_markdown` writes."""
        title, tags, status, messages, cur = "imported", [], "open", [], None
        for line in str(text).splitlines():
            if line.startswith("# "):
                title = line[2:].strip() or title
            elif line.startswith("- tags:"):
                tags = [t for t in re.split(r"[,\s]+", line.split(":", 1)[1].strip()) if t and t != "-"]
            elif line.startswith("- status:"):
                status = line.split(":", 1)[1].strip() or status
            elif line.startswith("## "):
                cur = {"role": line[3:].strip().lower(), "content": ""}
                messages.append(cur)
            elif cur is not None:
                cur["content"] += line + "\n"
        if status not in STATUSES:
            status = "open"
        msgs = [(m["role"], m["content"].strip("\n")) for m in messages]
        return self.create(title=title, messages=msgs, tags=tags, status=status)

    # ---- internals ----
    def _save(self, rec):
        # a unique tmp per write: two writers must never share one path, or one
        # truncating it mid-write leaves a torn file behind that reads as nothing
        tmp = "%s.%s.tmp" % (self.path(rec["id"]), uuid.uuid4().hex[:8])
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, self.path(rec["id"]))

    def _coerce(self, rec_or_id):
        if isinstance(rec_or_id, dict):
            return rec_or_id
        return self.get(str(rec_or_id))


def session_id_message_hash(rec):
    """A stable short hash of a session's transcript (for dedupe/logging)."""
    blob = "\n".join("%s:%s" % (m["role"], m["content"]) for m in rec.get("messages") or [])
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
