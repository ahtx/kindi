"""Shared mixture-of-experts policy for every Kindi MoE engine (Lightning/Nemotron, DeepSeek V4.1 Flash, Flash-Next).

The kernels stay with each engine, because the weight formats differ (Q2_0 ternary, MXFP4, Q4_K). This module is the
part that is the same for all of them:

- MoESpec: the shape of the experts (layers, experts per layer, experts used per token, bytes per expert).
- Usage: how often each expert of each layer is chosen, measured on text, saved as JSON next to the engine.
- plan(): which experts of which layer go to each memory tier (card, host RAM, disk), for a byte budget per tier.
  The card budget goes to the experts with the most measured use over ALL layers, not the same count in every
  layer, so a layer whose picks are spread out gets more card slots than a layer with a few hot experts.
- Placement: the result, with the expected share of routed picks that each tier serves.

An engine calls plan() once at load time and then uploads each layer's experts to the tier the placement names.
Nothing here touches the card, so the module is testable on the CPU (tools/test_moe.py).
"""
import json
import os

TIERS = ("card", "host", "disk")


class MoESpec(object):
    """layers: the MoE layer indexes of the model; n_exp experts per layer, n_used chosen per token;
    expert_bytes: the bytes of one expert (all its matrices) in the engine's own format, either one number for all
    layers, or a dict {layer: bytes} / a list indexed by layer when the quant types differ by layer (Flash-Next)"""

    def __init__(self, layers, n_exp, n_used, expert_bytes, shared=False):
        self.layers = [int(x) for x in layers]
        self.n_exp = int(n_exp)
        self.n_used = int(n_used)
        if isinstance(expert_bytes, dict):
            self._eb = {int(k): int(v) for k, v in expert_bytes.items()}
        elif isinstance(expert_bytes, (list, tuple)):
            self._eb = {il: int(expert_bytes[il]) for il in self.layers}
        else:
            self._eb = {il: int(expert_bytes) for il in self.layers}
        missing = [il for il in self.layers if il not in self._eb or self._eb[il] <= 0]
        self.shared = bool(shared)
        if not self.layers or self.n_used < 1 or self.n_used > self.n_exp or missing:
            raise ValueError("bad MoE spec: %d layers, %d of %d experts, no expert size for layers %s"
                             % (len(self.layers), self.n_used, self.n_exp, missing))
        self.expert_bytes = max(self._eb.values())     # (the largest; eb() gives each layer's own)

    def eb(self, il):
        """the bytes of one expert of layer il"""
        return self._eb[il]


class Usage(object):
    """per layer: a count for each expert (how often the router chose it). An older file with only an order
    (most used first) is read too; it then gives a uniform split, as before."""

    def __init__(self, spec, counts=None, order=None, meta=None):
        self.spec = spec
        self.counts = {int(k): [float(c) for c in v] for k, v in (counts or {}).items()}
        self._order = {int(k): [int(e) for e in v] for k, v in (order or {}).items()}
        self.meta = dict(meta or {})

    @classmethod
    def load(cls, spec, path):
        if not path or not os.path.exists(path):
            return cls(spec)
        d = json.load(open(path))
        return cls(spec, d.get("counts"), d.get("order"),
                   {k: v for k, v in d.items() if k not in ("counts", "order")})

    def save(self, path):
        d = dict(self.meta)
        d["order"] = {str(il): self.order(il) for il in self.spec.layers}
        if self.counts:
            d["counts"] = {str(il): [int(c) for c in self.counts[il]] for il in self.spec.layers if il in self.counts}
        with open(path, "w") as f:
            json.dump(d, f, separators=(",", ":"))

    def add(self, il, ids):
        """count the experts that layer il chose (any iterable of expert indexes)"""
        c = self.counts.setdefault(int(il), [0.0] * self.spec.n_exp)
        for e in ids:
            c[int(e)] += 1

    def order(self, il):
        """the experts of layer il, most used first (stable: ties keep index order)"""
        if il in self.counts:
            c = self.counts[il]
            return sorted(range(self.spec.n_exp), key=lambda e: (-c[e], e))
        if il in self._order:
            o = self._order[il]
            return o + [e for e in range(self.spec.n_exp) if e not in set(o)]
        return list(range(self.spec.n_exp))

    def has_counts(self):
        return all(il in self.counts and sum(self.counts[il]) > 0 for il in self.spec.layers)

    def share(self, il, experts):
        """the share of layer il's picks that the listed experts take (None without counts)"""
        if il not in self.counts:
            return None
        c = self.counts[il]
        t = sum(c)
        return sum(c[e] for e in experts) / t if t else None


class Placement(object):
    """tier[il][e] = "card" / "host" / "disk"; on(il, tier) lists the experts of a tier, most used first"""

    def __init__(self, spec, usage, tier):
        self.spec, self.usage, self.tier = spec, usage, tier

    def on(self, il, tier):
        return [e for e in self.usage.order(il) if self.tier[il][e] == tier]

    def count(self, il, tier):
        return sum(1 for t in self.tier[il] if t == tier)

    def counts(self, tier):
        return {il: self.count(il, tier) for il in self.spec.layers}

    def served(self, tier):
        """expected share of all routed picks that this tier serves (None without counts)"""
        if not self.usage.has_counts():
            return None
        num = den = 0.0
        for il in self.spec.layers:
            c = self.usage.counts[il]
            num += sum(c[e] for e in range(self.spec.n_exp) if self.tier[il][e] == tier)
            den += sum(c)
        return num / den if den else None

    def describe(self):
        cc = self.counts("card")
        lo, hi = min(cc.values()), max(cc.values())
        s = "%d to %d of %d experts per layer on the card" % (lo, hi, self.spec.n_exp) if lo != hi else \
            "%d of %d experts per layer on the card" % (lo, self.spec.n_exp)
        f = self.served("card")
        if f is not None and lo < self.spec.n_exp:
            s += " (%.1f%% of the measured picks)" % (100 * f)
        return s


def plan(spec, usage, card_bytes, host_bytes=None, min_card=None, weighted=None):
    """place the experts: card_bytes / host_bytes are the bytes that the experts may use in each tier (host_bytes
    None = unlimited; the rest goes to "disk", read from the mapped file). min_card: the least experts per layer on
    the card (default n_used, so one token's picks always fit). weighted: give the card slots by measured use over
    all layers (default: when the usage has counts, and KINDI_MOE_WEIGHTED is not 0); otherwise the same count in
    each layer, the most used experts of that layer first (the 0.5 behaviour)."""
    L, n = spec.layers, spec.n_exp
    eb = spec.eb
    min_card = spec.n_used if min_card is None else int(min_card)
    if weighted is None:
        weighted = usage.has_counts() and os.environ.get("KINDI_MOE_WEIGHTED", "1") != "0"
    card_bytes = int(card_bytes)
    need = sum(min_card * eb(il) for il in L)
    if card_bytes < need:
        raise MemoryError("the card has %.2f GB for experts; %d layers need %d experts each (%.2f GB)"
                          % (card_bytes / 1e9, len(L), min_card, need / 1e9))
    tier = {il: ["host"] * n for il in L}
    orders = {il: usage.order(il) for il in L}
    if card_bytes >= sum(n * eb(il) for il in L):
        for il in L:
            tier[il] = ["card"] * n
    elif not weighted:
        # the same count in each layer: the largest k whose experts fit
        per = sum(eb(il) for il in L)
        k = min(n, card_bytes // per)
        for il in L:
            for e in orders[il][:k]:
                tier[il][e] = "card"
    else:
        take = {il: min_card for il in L}
        left = card_bytes - need
        # greedy: the next card slot goes to the layer whose next expert has the largest share of that layer's
        # picks per byte (each token passes every MoE layer once, so each layer's picks count the same)
        tot = {il: sum(usage.counts[il]) or 1.0 for il in L}
        while True:
            best, gain = None, -1.0
            for il in L:
                if take[il] < n and eb(il) <= left:
                    g = usage.counts[il][orders[il][take[il]]] / tot[il] / eb(il)
                    if g > gain:
                        best, gain = il, g
            if best is None:
                break
            take[best] += 1
            left -= eb(best)
        for il in L:
            for e in orders[il][:take[il]]:
                tier[il][e] = "card"
    if host_bytes is not None:
        room = int(host_bytes)
        # the most used non-card experts (by share of their layer's picks) stay in host memory; the rest go to disk
        rest = sorted((((usage.counts[il][e] / (sum(usage.counts[il]) or 1.0)) if usage.has_counts() else 0.0),
                       -il, -e) for il in L for e in range(n) if tier[il][e] == "host")
        for share, nil, ne in reversed(rest):
            il, e = -nil, -ne
            if eb(il) <= room:
                room -= eb(il)
            else:
                tier[il][e] = "disk"
    return Placement(spec, usage, tier)
