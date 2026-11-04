"""Prompt-lookup drafting: propose the next tokens by finding the last n tokens earlier in the text (the prompt and
the reply so far) and copying what followed them. Agent work copies file contents and tool output often, and there
a whole run of tokens is right at once. The proposals cost no model work; the model checks them in one pass.

A check pass of n tokens costs more than one step, so the draft length follows the measured acceptance: the length
with the most tokens per unit of time, or no draft when a draft is expected to lose time."""

# the time of a check pass of n tokens, in steps (Nanbeige 33B IQ3_XXS, two RTX 3080 at 190 W, 2026-10-30)
CHECK_COST = {1: 1.0, 2: 1.18, 3: 1.47, 4: 1.67, 5: 1.94, 6: 2.16, 7: 2.6, 8: 3.02}


class Lookup(object):
    def __init__(self, ids=(), nmin=2, nmax=4, cost=None):
        self.h, self.nmin, self.nmax = [], nmin, nmax
        self.cost = cost or CHECK_COST
        self.acc, self.tries = 3.0, 5.0                 # kept / checked drafted tokens, decayed (prior: 60%)
        self.last = [dict() for _ in range(nmax + 1)]   # n-gram -> the index of its last token, latest occurrence
        self.prev = [None] * (nmax + 1)                 # the earlier occurrence of the n-gram that ends the text
        for t in ids:
            self.add(t)

    def add(self, t):
        h = self.h
        h.append(t)
        i = len(h) - 1
        for n in range(self.nmin, self.nmax + 1):
            if i + 1 < n:
                self.prev[n] = None
                continue
            key = tuple(h[i - n + 1:])
            self.prev[n] = self.last[n].get(key)
            self.last[n][key] = i

    def length(self, kmax):
        """the draft length with the most expected tokens per step's time, given acceptance a per token: a pass
        with k drafted tokens gives 1 + a + ... + a^k tokens. 0: a draft is expected to cost time."""
        a = self.acc / self.tries
        best, kb, got = 1.0, 0, 1.0
        for k in range(1, kmax + 1):
            got += a ** k
            v = got / self.cost.get(k + 1, 99.0)
            if v > best:
                best, kb = v, k
        return kb

    def draft(self, kmax):
        """up to kmax tokens that followed the longest earlier match of the text's end; [] when there is none or a
        draft is expected to lose time. A 4-token match gives a 1-token draft even then, so the acceptance is
        measured again when copying starts."""
        h = self.h
        k = self.length(kmax)
        for n in range(self.nmax, self.nmin - 1, -1):
            j = self.prev[n]
            if j is not None:
                if k == 0 and n == self.nmax:
                    k = 1
                return h[j + 1:j + 1 + k]
        return []

    def kept(self, j, n):
        """a check pass kept j of n drafted tokens (the first rejected one, if any, counts as a try)"""
        self.acc = 0.8 * self.acc + j
        self.tries = 0.8 * self.tries + j + (j < n)
