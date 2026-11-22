"""Decode slots (F7, docs/dev/F7-SLOTS.md): N conversations in one plan engine, one batched pass per written token.

KINDI_SLOTS=N (N >= 2) on one card with the tensor-core plans (KINDI_TC=1): the engine keeps N state sets (GDN state,
conv state, K/V cache) and builds every single-sequence plan once per slot; activate(k) points the shard and the
engine's state views at slot k. The batched pass (spec.build_plan with slots=...) carries one token per slot: the
matrix products are the same tensor-core products as a one-token pass (each token's result does not depend on the
others), and the per-token parts (RoPE, the K/V store, the attention, the GDN steps) are the decode launches on that
slot's own state. So a slot writes exactly the tokens it writes alone.

A slot that is not writing in a pass is an idle row: its GDN write-back flag is 0, its K/V row goes to a scratch
position (ctx, one row past the context) and its attention reads one position.

Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only. See LICENSE and COMMERCIAL-LICENSE.md.
"""
import os
import threading
import time

import numpy as np
import cupy as cp

STATE = ("M", "conv", "Kq", "Kd", "Vq", "Vd")
PLAN_MB = 72                                          # card memory of one slot's plans (58 MiB on the 27B)
PLANS = ("tc_plans", "tc_commit", "plan_chunk")


def wanted():
    """the slot count asked for (KINDI_SLOTS), 1 = off"""
    try:
        return max(1, int(os.environ.get("KINDI_SLOTS", "1") or 1))
    except ValueError:
        return 1


def state_bytes(s):
    """bytes of one slot's GDN and conv state on shard s"""
    return sum(int(a.nbytes) for a in s.M) + sum(int(a.nbytes) for a in s.conv)


class SlotSet(object):
    """N slots of one engine on one card. Slot 0 is the engine's own state and plans."""

    def __init__(self, eng, n, log=print):
        from . import spec as _spec
        from . import chunk as _chunk
        from . import tp as _tp
        if eng.W != 1 or not getattr(eng, "tc", False):
            raise SystemExit("kindi: decode slots need one card with the tensor-core plans (KINDI_TC=1)")
        if n > _spec.TMAX:
            raise SystemExit("kindi: at most %d slots" % _spec.TMAX)
        self.eng, self.n, self.log = eng, n, log
        self.lock = threading.RLock()                 # one engine pass at a time (a prompt chunk or a batched step)
        s = self.s = eng.cards[0]
        t0 = time.time()
        with cp.cuda.Device(s.card):
            self.state = [{k: getattr(s, k) for k in STATE}]
            for _ in range(1, n):
                self.state.append({k: [cp.zeros_like(a) for a in getattr(s, k)] for k in STATE})
            self.plans = [{k: getattr(s, k, None) for k in PLANS}]
            for k in range(1, n):
                self._bind_state(k)
                # a slot reads its prompt with the T-token passes (no head) and the chunk plan; its decode steps go
                # through the batched plan. (No verify plans: speculative decoding is off with slots.)
                f0 = cp.cuda.runtime.memGetInfo()[0]
                pl = {"tc_plans": {}, "tc_commit": None}
                for T in range(1, _spec.TMAX + 1):
                    pl["tc_plans"][(T, False)] = _spec.Graph(_spec.build_plan(s, T, False, 1))
                pl["tc_plans"][(1, True)] = _spec.Graph(_spec.build_plan(s, 1, True, 1))
                pl["plan_chunk"] = _chunk._build(s) if getattr(eng, "chunk_ok", False) else None
                self.plans.append(pl)
                if k == 1:
                    log("decode slots: one slot's plans take %d MiB" % ((f0 - cp.cuda.runtime.memGetInfo()[0]) // 2 ** 20))
            self._bind_state(0)
            # a batched plan for each row count m = 1 .. n (rows = slots 0 .. m - 1): a pass takes the smallest one
            # that holds every writing slot
            self.batch = {}
            for m in range(1, n + 1):
                P = _spec.build_plan(s, m, True, 1, slots=self.state[:m])
                names = set(["tok%d" % k for k in range(m)] + ["%s%d" % (v, k) for v in "pqw" for k in range(m)])
                bad = set(d.var for _, d in P.dyn) - names
                if bad:
                    raise RuntimeError("kindi: slot plan variables %s are not set per pass" % sorted(bad))
                self.batch[m] = _spec.Graph(P)
        self.views = []
        for k in range(n):
            st = self.state[k]
            self.views.append({
                "M": [_tp.Dist([(s.card, a)]) for a in st["M"]],
                "conv": [_tp.Dist([(s.card, a)]) for a in st["conv"]],
                "Kc": [_tp.Dist([(s.card, st["Kq"][i]), (s.card, st["Kd"][i])], rows=True) for i in range(s.NATT)],
                "Vc": [_tp.Dist([(s.card, st["Vq"][i]), (s.card, st["Vd"][i])], rows=True) for i in range(s.NATT)]})
        self.active = 0
        self.warm()
        free = cp.cuda.runtime.memGetInfo()[0] // 2 ** 20
        log("decode slots: %d, ctx %d each (%.1fs; %d MiB free on the card)" % (n, eng.ctx, time.time() - t0, free))

    def warm(self):
        """run every plan once (a graph's first launch uploads it, and a slot's first request paid for that), then
        clear every slot's state"""
        eng = self.eng
        for k in range(self.n):
            self.activate(k)
            for key, g in self.plans[k]["tc_plans"].items():
                g.launch([0] * 8, 0)
            if self.plans[k]["plan_chunk"] is not None:
                eng.feed_batch([0] * self.s.chunk["T"], 0)
        for m in self.batch:
            self.step({j: (0, 0) for j in range(m)})
        cp.cuda.Device(self.s.card).synchronize()
        for k in range(self.n):
            self.activate(k)
            eng.reset()
        self.activate(0)

    def _bind_state(self, k):
        for key in STATE:
            setattr(self.s, key, self.state[k][key])

    def activate(self, k):
        """point the shard (state and plans) and the engine's state views at slot k"""
        self._bind_state(k)
        for key in PLANS:
            setattr(self.s, key, self.plans[k][key])
        v = self.views[k]
        e = self.eng
        e.M, e.conv, e.Kc, e.Vc = v["M"], v["conv"], v["Kc"], v["Vc"]
        e._spq, e._sppend = [], False                  # (speculative decoding is off with slots)
        self.active = k

    def step(self, rows, keep=None):
        """one batched pass: rows = {slot: (token, position)} for the slots that write; returns {slot: argmax}.
        The other slots are idle rows (their state does not change). keep = (ylogs, aouts): each writing slot's
        logits row and argmax are copied there before the next pass can run."""
        from . import spec as _spec
        eng = self.eng
        m = max(rows) + 1
        vals = {}
        for k in range(m):
            if k in rows:
                tok, pos = rows[k]
                if pos >= eng.ctx:
                    raise OverflowError("position %d is past the context (%d)" % (pos, eng.ctx))
                vals.update({"tok%d" % k: int(tok), "p%d" % k: int(pos), "q%d" % k: int(pos) + 1, "w%d" % k: 1})
            else:
                vals.update({"tok%d" % k: 0, "p%d" % k: eng.ctx, "q%d" % k: 1, "w%d" % k: 0})
        rv = [vals["p%d" % k] for k in range(m)] + [vals["q%d" % k] for k in range(m)] + [vals["w%d" % k] for k in range(m)]
        with self.lock:
            g = self.batch[m]
            g.plan.rv.set(np.array(rv, dtype=np.int32))     # the row kernels' per-row values (spec.cu *_R)
            g.launch(vals=vals)
            out = _spec.argmaxes(eng, m)
            if keep is not None:
                for k in rows:
                    keep[0][k][:] = self.logits(k)
                    keep[1][k][0] = out[k]
        return {k: out[k] for k in rows}

    def logits(self, k):
        """slot k's logits row of the last batched pass (on the card; valid until the next pass)"""
        s = self.s
        return s.tcb["YH"][k * s.R_head:(k + 1) * s.R_head]


class Batcher(object):
    """Collects the slots' decode steps into batched passes. A slot thread calls step(k, tok, pos) and waits; the
    first waiting thread runs a pass once every writing slot has handed in its token, or after wait_ms."""

    def __init__(self, ss, wait_ms=None):
        self.ss = ss
        self.wait = float(os.environ.get("KINDI_SLOTS_WAIT_MS", "2") if wait_ms is None else wait_ms) / 1000.0
        self.cv = threading.Condition()
        self.pending, self.results, self.writing = {}, {}, set()
        self.running = False
        self.passes = self.rows = 0
        self.debug = {} if os.environ.get("KINDI_SLOTS_DEBUG") == "1" else None
        self.log = getattr(ss, "log", print)
        s = ss.s
        with cp.cuda.Device(s.card):
            # each slot's logits row and best token, copied after every pass (the sampler reads them, see Slot.ylog)
            self.ylog = [cp.zeros(s.R_head, dtype=cp.float32) for _ in range(ss.n)]
            self.aout = [cp.zeros(1, dtype=cp.int32) for _ in range(ss.n)]

    def step(self, k, tok, pos):
        t_in = time.time()
        r = self._step(k, tok, pos)
        if self.debug is not None:
            d = self.debug.setdefault(k, [0, 0.0])
            d[0] += 1
            d[1] += time.time() - t_in
        return r

    def _step(self, k, tok, pos):
        with self.cv:
            self.pending[k] = (tok, pos)
            self.writing.add(k)
            self.cv.notify_all()
            t0 = time.time()
            while k not in self.results:
                ready = len(self.pending) >= len(self.writing) or time.time() - t0 >= self.wait
                if not self.running and self.pending and ready:
                    rows, self.pending = self.pending, {}
                    self.running = True
                    self.cv.release()
                    try:
                        res = self.ss.step(rows, keep=(self.ylog, self.aout))
                    except BaseException as e:  # noqa: BLE001 - every slot of the pass gets the error
                        res = {j: e for j in rows}
                    finally:
                        self.cv.acquire()
                        self.running = False
                    self.passes += 1
                    self.rows += len(rows)
                    self.results.update(res)
                    self.cv.notify_all()
                else:
                    self.cv.wait(self.wait if not self.running else 0.05)
            r = self.results.pop(k)
        if isinstance(r, BaseException):
            raise r
        return r

    def done(self, k):
        """slot k's request is over: the others no longer wait for it"""
        with self.cv:
            self.writing.discard(k)
            self.pending.pop(k, None)
            self.cv.notify_all()
            if self.debug is not None and k in self.debug:
                n, t = self.debug.pop(k)
                self.log("slot %d: %d steps, %.2f ms each; writing now %s; %d passes, %.2f rows per pass" % (
                    k, n, t / max(n, 1) * 1000, sorted(self.writing), self.passes, self.rows / max(self.passes, 1)))


class Slot(object):
    """The engine interface (cache.Runner, Sampler, the server) for slot k: prompt reading runs on slot k's own plans,
    one chunk per engine turn; a decode step goes through the Batcher."""

    _NOT = ("spec_begin", "spec_end", "spec_known", "spec_presampled", "drafter", "mtp", "dspine", "ckpt_slot")

    def __init__(self, ss, batcher, k):
        self._ss, self._b, self.slot = ss, batcher, k
        self._eng = ss.eng
        v = ss.views[k]
        self.M, self.conv, self.Kc, self.Vc = v["M"], v["conv"], v["Kc"], v["Vc"]
        s = ss.s
        self._T = s.chunk["T"] if getattr(ss.eng, "chunk_ok", False) else 8
        self._best = 0

    def __getattr__(self, name):
        if name in Slot._NOT:
            raise AttributeError(name)
        return getattr(self._eng, name)

    def _run(self, fn, *a):
        with self._ss.lock:
            self._ss.activate(self.slot)
            return fn(*a)

    def reset(self):
        self._run(self._eng.reset)

    def feed(self, tok, pos):
        self._run(self._eng.feed, tok, pos)

    def feed_batch(self, toks, pos):
        T = self._T
        for i in range(0, len(toks), T):          # one chunk per engine turn: the other slots' passes go in between
            self._run(self._eng.feed_batch, toks[i:i + T], pos + i)

    def step(self, tok, pos):
        t = self._b.step(self.slot, tok, pos)
        self._best = t
        return t

    @property
    def ylog(self):
        return self._b.ylog[self.slot]

    @property
    def aout(self):
        return self._b.aout[self.slot]

    def release(self):
        self._b.done(self.slot)
