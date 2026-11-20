"""Pack and cache JSON must be read through `kindi.jsonio.load_json`.

`json.load(open(path))` leaves the handle to the garbage collector. On Windows that can
hold a pack file open long enough to block a concurrent convert or cache rewrite, so the
reader closes before returning and this test caps any new naked sites.
"""
import os
from pathlib import Path

from kindi.jsonio import load_json

ROOT = Path(__file__).resolve().parents[1]
MAX_NAKED_JSON_LOADS = 36


def test_load_json_reads_utf8_and_releases_the_handle(tmp_path):
    path = tmp_path / "meta.json"
    path.write_text('{"name": "caf\u00e9"}', encoding="utf-8")
    assert load_json(path) == {"name": "caf\u00e9"}
    os.remove(path)  # Windows refuses the delete while a handle is still open


def test_naked_json_loads_do_not_grow():
    offenders = []
    for path in sorted((ROOT / "kindi").rglob("*.py")):
        if path.name == "jsonio.py":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), 1):
            if "json.load(open(" in line:
                offenders.append("%s:%d" % (path.relative_to(ROOT), lineno))
    assert len(offenders) <= MAX_NAKED_JSON_LOADS, (
        "new naked json.load(open(...)) sites; use kindi.jsonio.load_json:\n"
        + "\n".join(offenders))
