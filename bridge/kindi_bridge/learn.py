# Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only. See LICENSE and COMMERCIAL-LICENSE.md.
"""Learning-layer runtime: corrections mined from agent transcripts, injected as nudges at the prompt choke point.

The correction LIBRARY is data (JSON): a list of {id, match, after, then}. `match` is a
case-insensitive regex searched over the agent's recent settled job responses and errors;
when a pattern has matched at least `after` times inside the recent window, its `then` text
is prepended to the next prompt as a one-line nudge. The nudge keeps firing while the pattern
keeps appearing in the window, and stops by itself when the failures age out — no extra state.

Files, first wins:
  ~/.local/state/kindi-bridge/corrections.json   learned library (the miner grows this)
  <package>/assets/corrections.json              shipped seed from the first mining pass
                                                 (conductor docs/learning-mining.md)
"""
import json
import re
from pathlib import Path

WINDOW = 12                       # settled jobs that count as "recent" for an agent


def _library_path():
    state = Path.home() / ".local" / "state" / "kindi-bridge" / "corrections.json"
    return state if state.exists() else Path(__file__).parent / "assets" / "corrections.json"


def load_corrections():
    try:
        data = json.loads(_library_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [c for c in data if isinstance(c, dict) and c.get("match") and c.get("then")]


def recent_texts(store, agent, limit=WINDOW):
    rows = store.rows(
        "SELECT response, error FROM jobs WHERE agent=? AND finished IS NOT NULL "
        "ORDER BY finished DESC LIMIT ?", (agent, limit))
    return [t for r in rows for t in (r['response'], r['error']) if t]


def nudge_for(store, agent):
    """The strongest correction (most matches) whose threshold is met in the agent's recent window."""
    texts = recent_texts(store, agent)
    if len(texts) < 2:
        return None
    blob = "\n".join(texts)
    best = None
    for c in load_corrections():
        n = len(re.findall(c["match"], blob, re.I))
        if n >= int(c.get("after", 2)) and (best is None or n > best[0]):
            best = (n, c["then"])
    return best[1] if best else None
