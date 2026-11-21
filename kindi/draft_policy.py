# SPDX-License-Identifier: AGPL-3.0-only
"""Adaptive draft-length control for speculative decoding (roadmap suggestion 4).

`kindi/lookup.py` already adapts the n-gram draft length from measured
acceptance. This module is the same idea for a model drafter, shared by any
draft family (2B pack, MTP head, DSpine): it keeps decayed acceptance counts
per drafter, and picks the draft length with the most expected tokens per
unit of verify-pass time. A verify pass of k+1 tokens costs more than one
step; `costs[k]` is its price in steps (measured per card, like
`lookup.CHECK_COST`). Without a measured cost the default is the lookup
table, so the policy is usable before a card is timed.

`kindi/spec.py` adopts it behind `KINDI_DRAFT_POLICY=1`: it clamps the
engine's base k and records every verify pass. The flag stays off by default
because making it the default needs a GPU gate (exactness 4/4 with the policy
driving k, and no tok/s regression against the fixed-k rows in `docs/MTP.md`).

`confidence` is the drafter's top-1 probability for its next position
(0..1). When given, it scales the measured acceptance for this step only:
a confident first token is more likely to be accepted than the historical
average, an unconfident one less. It never changes the output tokens, only
how many are drafted.
"""
# verify-pass price in plain steps, keyed by drafted tokens k (pass of k + 1).
# Same measured table as kindi/lookup.py (Nanbeige 33B, 2x RTX 3080).
from .lookup import CHECK_COST

DECAY = 0.8


class DraftPolicy(object):
    """One adaptive controller per (target, drafter) pair."""

    def __init__(self, name, kmax=3, costs=None, decay=DECAY):
        if not 0 <= kmax <= 7:
            raise ValueError("kmax must be 0..7 (the TMAX verify limit)")
        self.name = name
        self.kmax = int(kmax)
        self.costs = dict(costs or CHECK_COST)
        self.decay = float(decay)
        self.acc, self.tries = 3.0, 5.0  # kept / checked, decayed (prior 60%)
        self.conf = None                 # EWMA of drafter top-1 confidence
        self.steps = self.drafted = self.kept_total = 0

    @property
    def acceptance(self):
        return self.acc / self.tries if self.tries else 0.0

    def cost(self, k):
        return float(self.costs.get(k, 99.0))

    def limit_k(self, base_k):
        """Clamp an engine-chosen base draft length to what this controller proposes now.

        The engine keeps its own hand-tuned base (per card/mode); the controller may only shorten
        it, never lengthen it. Returns min(base_k, propose()), floored at 0. A controller with no
        observations at all has no evidence, so the base passes through unchanged.
        """
        if not self.tries:
            return int(base_k)
        return max(0, min(int(base_k), int(self.propose())))

    def observe(self, kept, drafted, confidence=None):
        """A verify pass kept `kept` of `drafted` drafted tokens."""
        self.acc = self.decay * self.acc + kept
        self.tries = self.decay * self.tries + kept + (1 if kept < drafted else 0)
        self.steps += 1
        self.drafted += drafted
        self.kept_total += kept
        if confidence is not None:
            c = min(1.0, max(0.0, float(confidence)))
            self.conf = c if self.conf is None else self.decay * self.conf + (1.0 - self.decay) * c

    def _rate(self, k, a):
        if k <= 0:
            return 1.0
        got = 1.0 + sum(a ** j for j in range(1, k + 1))
        return got / self.cost(k)

    def propose(self, confidence=None):
        """The k in 0..kmax with the most expected tokens per unit time, or 0.

        0 means: drafting is expected to cost more than it earns at the
        measured acceptance (and step confidence, when given).
        """
        if self.kmax <= 0:
            return 0
        a = self.acceptance
        if confidence is not None:
            a *= min(1.0, max(0.0, float(confidence)))
        best_k, best_rate = 0, 1.0
        for k in range(1, self.kmax + 1):
            r = self._rate(k, a)
            if r > best_rate + 1e-12:
                best_k, best_rate = k, r
        return best_k

    def stats(self):
        return {"name": self.name, "acceptance": round(self.acceptance, 4),
                "confidence": None if self.conf is None else round(self.conf, 4),
                "steps": self.steps, "drafted": self.drafted, "kept": self.kept_total}
