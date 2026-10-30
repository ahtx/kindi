"""Prompt cache for Kindi's Gemma 4 engine (Nemotron).

The global attention layers keep K/V for every position, so any prefix that the cached tokens share with a new request
is valid there. A sliding-window layer keeps only its last `ring_slots` positions, and a query at position L needs the
keys from L - 511. So a request can start at L only when the ring still holds those keys, or from a copy of the rings
taken at a position P <= L. Three mechanisms:
  * resume in place: the common prefix with the conversation on the card, while its rings still hold the window;
  * checkpoints: the rings are copied to host memory at the end of every prompt batch (512 tokens) and at the end of
    every prompt; a request resumes from the last checkpoint at or before the point where it differs;
  * parked conversations: when a request shares less than half of the conversation on the card, that conversation
    (its tokens, the global K/V up to its length and its checkpoints) moves to host memory; its next turn moves it
    back. Agents that alternate between prompts keep their caches.
All of it is exact: the K/V and the rings are copied bit for bit.

Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only. See LICENSE and COMMERCIAL-LICENSE.md.
"""
import os

import numpy as np


def _lcp(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class G4Cache(object):
    def __init__(self, eng, log=print):
        self.eng, self.log = eng, log
        self.on = os.environ.get("KINDI_G4_PROMPT_CACHE", "1") != "0"
        self.ckpt_max = int(os.environ.get("KINDI_G4_CKPT_MAX", "16"))       # ring copies kept per conversation
        self.park_max = int(os.environ.get("KINDI_G4_PARK_MAX", "4"))         # conversations kept in host memory
        self.tokens = []                  # the tokens whose K/V the card holds, in position order
        self.ckpts = []                   # [(position, host copy of the rings)], ascending
        self.parked = []                  # [{"tokens", "globals", "ckpts"}], most recent last
        self.slots = getattr(eng, "ring_slots", 1024)

    # ---------------------------------------------------------------- engine state
    def _cards(self):
        """(device id, engine) for each card: one, or two in tensor parallel"""
        return [(c, s) for c, s in enumerate(self.eng.cards)] if hasattr(self.eng, "cards") else [(None, self.eng)]

    def _each(self, fn):
        import cupy as cp
        out = []
        for c, s in self._cards():
            if c is None:
                out.append(fn(s))
            else:
                with cp.cuda.Device(c):
                    out.append(fn(s))
        return out

    @staticmethod
    def _to_host(a):
        """an asynchronous copy of a device array into pinned host memory, in stream order (a later copy back on the
        same stream sees it complete), so the prompt does not wait for it"""
        import cupy as cp
        import cupyx
        buf = cupyx.empty_pinned(a.shape, a.dtype)
        a.get(stream=cp.cuda.get_current_stream(), out=buf)
        return buf

    def _rings_host(self):
        return self._each(lambda s: {i: (self._to_host(c["k"]), self._to_host(c["v"])) for i, c in s.cache.items() if i in s.local})

    def _rings_set(self, snap):
        import cupy as cp
        for (c, s), r in zip(self._cards(), snap):
            def put(s=s, r=r):
                for i, (k, v) in r.items():
                    s.cache[i]["k"].set(k); s.cache[i]["v"].set(v)
            if c is None:
                put()
            else:
                with cp.cuda.Device(c):
                    put()

    def _globals_host(self, n):
        return self._each(lambda s: {i: (self._to_host(c["k"][:n]), self._to_host(c["v"][:n])) for i, c in s.cache.items() if i not in s.local})

    def _globals_set(self, g, n):
        import cupy as cp
        for (c, s), r in zip(self._cards(), g):
            def put(s=s, r=r):
                for i, (k, v) in r.items():
                    s.cache[i]["k"][:n].set(k[:n]); s.cache[i]["v"][:n].set(v[:n])
            if c is None:
                put()
            else:
                with cp.cuda.Device(c):
                    put()

    # ---------------------------------------------------------------- requests
    def _resume_point(self, tokens, ckpts, ids, on_card=True):
        """the first position to compute for ids against a conversation (its tokens and checkpoints); on_card: its
        rings are the ones on the card (the active conversation), so a recent enough prefix needs no ring copy"""
        L = min(_lcp(tokens, ids), len(ids) - 1)
        if L <= 0:
            return 0, None
        if on_card and L - 511 >= len(tokens) - self.slots:
            return L, None                                    # the rings still hold the window
        best = None
        for p, snap in ckpts:
            if p <= L:
                best = (p, snap)
        return (best[0], best[1]) if best else (0, None)

    def prepare(self, ids):
        """set the card up for ids and return how many leading positions are already there"""
        if not self.on:
            self.tokens, self.ckpts = [], []
            return 0
        here, snap = self._resume_point(self.tokens, self.ckpts, ids)
        # a parked conversation that shares more of ids comes back to the card
        bi, bp = -1, here
        for i, P in enumerate(self.parked):
            p, _ = self._resume_point(P["tokens"], P["ckpts"], ids, on_card=False)
            if p > bp:
                bi, bp = i, p
        if bi >= 0:
            P = self.parked.pop(bi)           # take it out first: parking the active one may drop the oldest
            self._park_active(ids)
            p, snap = self._resume_point(P["tokens"], P["ckpts"], ids, on_card=False)
            self._globals_set(P["globals"], p)
            self.tokens, self.ckpts = list(P["tokens"][:p]), [c for c in P["ckpts"] if c[0] <= p]
            if snap is not None:
                self._rings_set(snap)
            self.log("prompt cache: conversation of %d tokens back from host memory, %d positions reused"
                     % (len(P["tokens"]), p))
            return p
        if here < 0.5 * len(self.tokens) and len(self.tokens) >= 1024:
            self._park_active(ids)
            here, snap = 0, None
        if snap is not None:
            self._rings_set(snap)
        self.tokens = list(self.tokens[:here])
        self.ckpts = [c for c in self.ckpts if c[0] <= here]
        return here

    def _park_active(self, ids):
        """move the conversation on the card to host memory (when it is long enough to be worth keeping)"""
        n = len(self.tokens)
        if n < 1024 or self.park_max <= 0:
            return
        ck = self.ckpts + ([(n, self._rings_host())] if not self.ckpts or self.ckpts[-1][0] != n else [])
        self.parked.append({"tokens": list(self.tokens), "globals": self._globals_host(n), "ckpts": ck[-self.ckpt_max:]})
        del self.parked[:-self.park_max]
        self.log("prompt cache: conversation of %d tokens parked in host memory (%d parked)" % (n, len(self.parked)))
        self.tokens, self.ckpts = [], []

    def advanced(self, tokens, checkpoint=False):
        """the card now holds tokens (call after each prompt batch, at the prompt's end and after generated tokens)"""
        self.tokens = list(tokens)
        if checkpoint and self.on and (not self.ckpts or self.ckpts[-1][0] < len(tokens)):
            self.ckpts.append((len(tokens), self._rings_host()))
            del self.ckpts[:-self.ckpt_max]

    def append(self, token):
        self.tokens.append(token)
