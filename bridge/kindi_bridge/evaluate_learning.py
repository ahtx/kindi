# Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only. See LICENSE and COMMERCIAL-LICENSE.md.
"""Offline replay evaluation of the shipped correction library on real pi transcripts."""
import collections
import glob
import json
import os
import tempfile
from pathlib import Path

from .learn import load_corrections, nudge_for
from .store import Store

PATTERNS = {
    "python-traceback": "Traceback \\(most recent call last\\)",
    "parse-errors": "SyntaxError|Unexpected token|JSONDecodeError|invalid json",
    "missing-files": "No such file or directory",
    "usage-errors": "Usage:|invalid option|ambiguous redirect",
    "context-pressure": "compaction",
}
import re


def result_texts(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            d = json.loads(line)
        except ValueError:
            continue
        result = d.get("result")
        if result is None:
            continue
        out.append(result if isinstance(result, str) else json.dumps(result)[:2000])
    return out


def expected(texts):
    texts = texts[-12:]
    counts = {cid: sum(len(re.findall(pattern, text, re.I)) for text in texts) for cid, pattern in PATTERNS.items()}
    cid, count = max(counts.items(), key=lambda x: x[1])
    return cid if count >= 2 else None


def replay(texts, agent="eval"):
    with tempfile.TemporaryDirectory() as tmp:
        store = Store(Path(tmp) / "state.sqlite3")
        agent = store.add_agent(agent, tmp, ["fake"])
        for i, text in enumerate(texts):
            store.execute("INSERT INTO jobs(id,agent,prompt,due,finished,response,error,created) VALUES(?,?,?,?,?,?,?,?)",
                          (str(i), agent["id"], "replay", i, i, None, text, i))
        return nudge_for(store, agent["id"])


def correction_ids():
    return {c["then"]: c["id"] for c in load_corrections()}


def evaluate(paths, heldout=lambda i: True):
    by_id = correction_ids()
    rows = []
    for i, path in enumerate(paths):
        if not heldout(i):
            continue
        texts = result_texts(path)
        want, nudge = expected(texts), replay(texts)
        got = by_id.get(nudge)
        rows.append({"run": path, "expected": want, "actual": got, "correct": want == got})
    usable = [r for r in rows if r["expected"]]
    correct = sum(r["correct"] for r in usable)
    return {"runs": len(rows), "evaluated": len(usable), "correct": correct,
            "top1_accuracy": round(correct / max(1, len(usable)), 4), "rows": rows,
            "clean_false_alarm": replay(["completed successfully", "all tests passed"]) is not None}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("var", nargs="?", default=os.path.expanduser("~/conductor/var"))
    parser.add_argument("--out", help="write the complete report as JSON")
    args = parser.parse_args()
    paths = sorted(glob.glob(os.path.join(args.var, "jobs/*/a*/work/.agent/pi-r*.jsonl")))
    heldout = evaluate(paths, lambda i: i % 4 == 3)
    full = evaluate(paths)
    report = {"var": args.var, "method": "every fourth run held out; shipped library, no fitting",
              "heldout": heldout, "all_runs": full}
    text = json.dumps(report, indent=1)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print("heldout: {}/{} top-1 = {:.1%}; all: {}/{} = {:.1%}; clean false alarm={}".format(
        heldout["correct"], heldout["evaluated"], heldout["top1_accuracy"], full["correct"], full["evaluated"],
        full["top1_accuracy"], full["clean_false_alarm"]))


if __name__ == "__main__":
    main()
