"""Prompt cache for a recurrent-hybrid model.

Bonsai's Gated-DeltaNet layers carry a recurrent state that cannot be rewound, so prefix reuse works on checkpoints:
  * checkpoints: the GDN state is copied to host RAM every `checkpoint_every` tokens and at the end of each prompt;
    a request resumes from the last checkpoint at or before its first differing token (the attention K/V cache is
    position-indexed and stays valid for any prefix);
  * parked conversations: when a request is unrelated to the active conversation (shares < 50 % of it), the active
    one - tokens, checkpoints, end state and K/V - is moved to host RAM (`park_max` kept) and resumed when its next
    turn arrives (agents alternate between their main prompt and side prompts);
  * saved openings: when two requests share >= `prefix_min` tokens and then differ, the full state at that checkpoint
    is written to disk (`prefix_max` kept, least recently used dropped), so the next session that starts the same way
    (same system prompt and tools) loads it in a fraction of a second, also after a restart.
All of it is exact: resuming gives the same tokens as reading the prompt from scratch."""
import hashlib
import json
import os
import shutil
import time

import numpy as np
try:
    import cupy as cp
except ImportError:               # the CPU engine (no VRAM checkpoint slot)
    cp = None


def _host(a):
    """a host (NumPy) copy of an engine array: CuPy on the GPU engines, NumPy on the CPU engine"""
    return a.get() if hasattr(a, "get") else np.array(a)


# pinned checkpoint copies: on by default under WSL2 (/dev/dxg; the WDDM driver copies pageable memory slowly: first
# token 1.52 -> 0.98 s on a GTX 1070, issue #4); on Linux they gave no gain (RTX 3060, 19 MB: 6.2 vs 6.9 ms)
PINNED = os.environ.get("KINDI_CKPT_PINNED", "1" if os.path.exists("/dev/dxg") else "0") == "1"


def _host_ckpt(a):
    """a checkpoint's host copy of a GPU array, in pinned (page-locked) memory (KINDI_CKPT_PINNED=1), else a plain
    NumPy array"""
    if not PINNED or cp is None or not isinstance(a, cp.ndarray):
        return _host(a)
    try:
        mem = cp.cuda.alloc_pinned_memory(a.nbytes)
    except Exception:  # noqa: BLE001 - no pinned memory left: a plain copy
        return _host(a)
    h = np.frombuffer(mem, dtype=a.dtype, count=a.size).reshape(a.shape)   # (h keeps mem alive)
    a.get(out=h)
    return h


def _to_dev(dst, pos, src):
    """dst[:pos] = src[:pos] (a parked K/V block back into the engine), in place. A plain CuPy assignment from a
    host array first makes a temporary device copy, and on a full card there is no room for it (GTX 1080, 27B at
    65,536 context: out of memory while a 41,873-token conversation was brought back)."""
    s = np.ascontiguousarray(src[:pos])
    if hasattr(dst, "set") and not hasattr(dst, "parts") and not isinstance(dst, np.ndarray):
        dst[:pos].set(s)                     # CuPy
    else:
        dst[:pos] = s                        # NumPy (CPU engine) or tp.Dist (copies in place itself)


def _set_host(dst, h):
    """dst[...] = h in place: CuPy .set on the GPU engines, np.copyto on the CPU engine (whose arrays the C code
    holds by address, so they must not be replaced)"""
    if hasattr(dst, "set") and not isinstance(dst, np.ndarray):
        dst.set(h)
    else:
        np.copyto(dst, h)


def _free_device_pool():
    try:
        import cupy
        cupy.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass


class Cancelled(Exception):
    """The client disconnected; the engine state is consistent (every processed token is in Runner.tokens)."""


def _lcp(a, b):
    n = min(len(a), len(b))
    L = 0
    while L < n and a[L] == b[L]:
        L += 1
    return L


# the files whose code decides the arithmetic of the state (a change of any of them changes the saved states)
_ARITH = ("kindi.cu", "chunk.cu", "chunk1.cu", "tc.cu", "tcp.cu", "tcf.cu", "tca.cu", "spec.cu", "kcpu.c", "engine.py", "tp.py",
          "chunk.py", "chunk1.py", "tc.py", "spec.py", "cpu.py", "layout.py")


def engine_fingerprint(eng, cfg):
    """(short hash, description) of everything that changes the engine state for the same tokens: the engine and its
    weight format, the K/V type, the prompt path (chunks, int8 products, tile, attention), the tensor-core and
    speculative settings, the head mode, the adapter and the code of the arithmetic. Saved openings are kept per
    fingerprint: a state made by another engine is never resumed."""
    here = os.path.dirname(os.path.abspath(__file__))
    env = os.environ
    cards = getattr(eng, "cards", None)
    c0 = cards[0] if cards else eng
    chunk = getattr(c0, "chunk", None) or {}
    d = {
        "engine": type(eng).__module__ + "." + type(eng).__name__,
        "cards": len(cards) if cards else 1,
        "weights": "5-trit lane" if getattr(c0, "g5l", False) else ("5-trit" if getattr(c0, "g5", False) else "2-bit"),
        "kv": "q8_0" if cards else "f16",
        "chunk": int(getattr(eng, "chunk_ok", False)) and int(chunk.get("T", 0)),
        "chunk_q8": bool(chunk.get("q8")),
        "chunk_tc": bool(chunk.get("tc")),
        "chunk_att": env.get("KINDI_CHUNK_ATT", "flash"),
        "chunk_tile": env.get("KINDI_CHUNK_TILE", ""),
        "batch": env.get("KINDI_BATCH", "4"),
        "tc": env.get("KINDI_TC", "0"),
        "tc_tmax": env.get("KINDI_TC_TMAX", ""),
        "fd": env.get("KINDI_FD", "0"),               # split flash-decode attention (tcf.cu)
        "d1": env.get("KINDI_D1", "0"),               # the one-pass RMS fold
        "fd_ch": env.get("KINDI_FD_CH", "512"),       # positions per flash-decode block
        "kexpf": env.get("KINDI_KEXPF", "1"),         # kindi.cu's kexpf in the tc kernels (0: NVRTC's expf)
        "att_tc": env.get("KINDI_ATT_TC", "0"),       # tensor-core prompt attention (tca.cu)
        "spec": env.get("KINDI_SPEC", "0"),
        "head": cfg.get("head", "exact"),
        "code": {},
    }
    for name in _ARITH:
        p = os.path.join(here, name)
        if os.path.exists(p):
            d["code"][name] = hashlib.sha256(open(p, "rb").read()).hexdigest()[:12]
    h = hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]
    return h, d


class Openings(object):
    """Saved prompt openings on disk: tokens[:p] and the engine state after them (GDN state + K/V below p)."""

    def __init__(self, eng, root, keep, log, fingerprint=None):
        self.eng, self.root, self.keep, self.log = eng, root, keep, log
        self.fp = fingerprint                     # (hash, description) of the engine; entries of another are ignored
        self.entries = {}
        try:
            os.makedirs(root, exist_ok=True)
            for key in os.listdir(root):
                d = os.path.join(root, key)
                try:
                    meta = json.load(open(os.path.join(d, "meta.json")))
                    if self.fp is not None and meta.get("engine") != self.fp[0]:
                        continue                  # made by another engine configuration
                    if meta["p"] < eng.ctx:
                        self.entries[key] = {"p": meta["p"], "used": meta.get("used", 0),
                                             "tokens": np.load(os.path.join(d, "tokens.npy"))}
                except Exception:  # noqa: BLE001 - a damaged entry is skipped, not fatal
                    continue
        except OSError as e:
            log("saved openings disabled: %s" % e)
        log("saved openings: %d in %s" % (len(self.entries), root))

    @staticmethod
    def key_of(tokens):
        return hashlib.sha1(np.asarray(tokens, dtype=np.int32).tobytes()).hexdigest()[:20]

    def has(self, tokens):
        return self.key_of(tokens) in self.entries

    def best(self, T, lo, hi):
        """The longest saved opening with lo < p <= hi that T starts with: (key, p) or None."""
        if not self.entries:
            return None
        Ta = np.asarray(T[:hi], dtype=np.int32)
        best = None
        for key, e in self.entries.items():
            p = e["p"]
            if lo < p <= hi and (best is None or p > best[1]) and np.array_equal(e["tokens"], Ta[:p]):
                best = (key, p)
        return best

    def save(self, tokens, Ms, Cs):
        p, key = len(tokens), self.key_of(tokens)
        if key in self.entries or self.keep <= 0:
            return
        t0 = time.time()
        d = os.path.join(self.root, key)
        try:
            os.makedirs(d, exist_ok=True)
            arrs = {}
            for i, m in enumerate(Ms):
                arrs["M%d" % i] = m
            for i, c in enumerate(Cs):
                arrs["C%d" % i] = c
            for i in range(len(self.eng.Kc)):
                arrs["K%d" % i] = _host(self.eng.Kc[i][:p])
                arrs["V%d" % i] = _host(self.eng.Vc[i][:p])
            np.savez(os.path.join(d, "state.npz"), **arrs)
            toks = np.asarray(tokens, dtype=np.int32)
            np.save(os.path.join(d, "tokens.npy"), toks)
            now = time.time()
            meta = {"p": p, "used": now, "created": now}
            if self.fp is not None:
                meta["engine"], meta["engine_config"] = self.fp
            json.dump(meta, open(os.path.join(d, "meta.json"), "w"))
            self.entries[key] = {"p": p, "used": now, "tokens": toks}
            self.log("saved an opening of %d tokens (%.1fs)" % (p, time.time() - t0))
        except Exception as e:  # noqa: BLE001 - a full disk must not break serving
            self.log("could not save an opening: %s" % e)
            shutil.rmtree(d, ignore_errors=True)
            return
        while len(self.entries) > self.keep:
            old = min(self.entries, key=lambda k: self.entries[k]["used"])
            self.entries.pop(old, None)
            shutil.rmtree(os.path.join(self.root, old), ignore_errors=True)

    def load(self, key):
        t0 = time.time()
        eng = self.eng
        d = os.path.join(self.root, key)
        z = np.load(os.path.join(d, "state.npz"))
        p = self.entries[key]["p"]
        Ms = [z["M%d" % i] for i in range(len(eng.M))]
        Cs = [z["C%d" % i] for i in range(len(eng.conv))]
        for a, h in zip(eng.M, Ms):
            _set_host(a, h)
        for a, h in zip(eng.conv, Cs):
            _set_host(a, h)
        for i in range(len(eng.Kc)):
            eng.Kc[i][:p] = z["K%d" % i]
            eng.Vc[i][:p] = z["V%d" % i]
        self.entries[key]["used"] = time.time()
        try:
            meta = json.load(open(os.path.join(d, "meta.json")))
            meta["used"] = time.time()
            json.dump(meta, open(os.path.join(d, "meta.json"), "w"))
        except Exception:  # noqa: BLE001
            pass
        self.log("loaded a saved opening of %d tokens (%.1fs)" % (p, time.time() - t0))
        return p, Ms, Cs


class _Slot(object):
    """KINDI_CKPT_VRAM=1 (one card): checkpoints without the wait. A checkpoint is a device copy of the GDN state into
    a VRAM slot (~1 ms). Its host copy (into the checkpoint's own pinned buffer, ~0.6 s for 152 MB over the cluster's
    PCIe x1 risers) is made when the server is idle - after a request, in 1 MB pieces on a background thread that
    pauses as soon as the next request comes in - never during a request's decode, whose per-token readbacks share
    the link. A checkpoint that is needed on the host before then (parked, saved as an opening, or one that may not
    be dropped when a newer one needs the slot) is copied right away. The newest checkpoint stays in the slot, so
    resuming from it is a device copy. Costs one state copy of card memory (taken before the context fit)."""

    PIECE = 1 << 20

    def __init__(self, devs, nM):
        import threading
        self.devs, self.nM = list(devs), nM           # the engine's GDN state arrays: M (nM of them), then conv
        self.stage = [cp.empty_like(a) for a in self.devs]
        self.total = sum(a.nbytes for a in self.devs)
        self.dev_id = cp.cuda.Device().id
        self.owner = None                 # the checkpoint whose state is in the slot (its host copy may be partial)
        self.spare = []                   # pinned buffers of checkpoints that are gone (2 kept for reuse)
        self.lock = threading.Lock()      # held by whoever copies pieces
        self.go = threading.Event()       # set: the background copier may run
        threading.Thread(target=self._worker, daemon=True, name="kindi-ckpt-copy").start()

    def _pieces(self, st, stream, stop):
        """copy the slot into st.host from where it stopped; True when complete"""
        while st.off < self.total:
            if stop is not None and not stop.is_set():
                return False
            i, o = st.at()
            b, h = self.stage[i], st.host[i]
            src, dst = b.reshape(-1).view(cp.uint8), h.reshape(-1).view(np.uint8)
            n = min(self.PIECE, src.size - o)
            src[o:o + n].get(out=dst[o:o + n], stream=stream)
            stream.synchronize()
            st.off += n
        st.done = True
        return True

    def _worker(self):
        with cp.cuda.Device(self.dev_id):
            stream = cp.cuda.Stream(non_blocking=True)
            while True:
                self.go.wait()
                with self.lock:
                    st = self.owner
                    if st is not None and not st.done and not st.lost and self.go.is_set():
                        self._pieces(st, stream, self.go)
                    if st is None or st.done or st.lost:
                        self.go.clear()

    def idle(self):
        """the server has nothing to do: finish the newest checkpoint's host copy in the background"""
        if self.owner is not None and not self.owner.done:
            self.go.set()

    def pause(self):
        """a request starts: the background copy stops at the next piece"""
        self.go.clear()
        with self.lock:
            pass

    def complete(self, st):
        """st's host copy, now (on the caller's path)"""
        if st.done or st.lost:
            return
        self.pause()
        with self.lock:
            if st is self.owner:
                self._pieces(st, cp.cuda.Stream(non_blocking=True), None)

    def pending(self):
        return self.owner is not None and not self.owner.done and not self.owner.lost

    def save(self, drop_ok):
        """a new checkpoint into the slot. The one there now: copied to the host first, or (drop_ok) dropped."""
        self.pause()
        prev = self.owner
        if prev is not None and not prev.done and not prev.lost:
            if drop_ok:
                prev.lost = True
            else:
                self.complete(prev)
        for a, b in zip(self.devs, self.stage):
            cp.copyto(b, a)
        mem = self.spare.pop() if self.spare else cp.cuda.alloc_pinned_memory(self.total)
        host, o = [], 0
        for a in self.devs:
            host.append(np.frombuffer(mem, dtype=a.dtype, count=a.size, offset=o).reshape(a.shape))
            o += a.nbytes
        cp.cuda.Device(self.dev_id).synchronize()        # (the slot is written before any piece is read)
        st = _State(self, host, mem)
        import weakref
        weakref.finalize(st, self._recycle, mem)       # (mem only: the finalizer must not keep st alive)
        self.owner = st
        return st

    def _recycle(self, mem):
        """a checkpoint is gone: its pinned buffer is reused (2 spares), or given back to the OS"""
        if len(self.spare) < 2:
            self.spare.append(mem)
        else:
            del mem
            cp.get_default_pinned_memory_pool().free_all_blocks()

    def restore(self, st):
        """the state of checkpoint st back in the engine: from the slot if it is still there"""
        if st is self.owner:
            for a, b in zip(self.devs, self.stage):
                cp.copyto(a, b)
            return True
        return False


class _State(object):
    """a checkpoint made through the VRAM slot; iterates as (Ms, Cs) host arrays like the plain ones"""

    def __init__(self, slot, host, mem):
        self.slot, self.host, self.mem = slot, host, mem
        self.lost, self.done, self.off = False, False, 0
        self.sizes = [h.nbytes for h in host]

    def at(self):
        """(array index, byte offset in it) of the copy position"""
        o = self.off
        for i, n in enumerate(self.sizes):
            if o < n:
                return i, o
            o -= n
        return len(self.sizes) - 1, self.sizes[-1]

    def __iter__(self):
        if self.lost:
            raise KeyError("a dropped checkpoint")
        if not self.done:
            self.slot.complete(self)
        n = self.slot.nM
        return iter((self.host[:n], self.host[n:]))


def _ram_bytes():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 8 * 2 ** 30


# branch parking (and resuming inside a parked conversation without taking it apart) is on by default on ROCm: the
# 780M shares system RAM, so parked conversations are cheap to keep and to restore
try:
    import cupy as _cp
    _PARK_DEFAULT = "1" if getattr(_cp.cuda.runtime, "is_hip", False) else "0"
except ImportError:
    _PARK_DEFAULT = "0"


class Runner(object):
    """The engine plus the cache. self.tokens = the tokens whose effect is in the engine state."""

    def __init__(self, eng, cfg, pack_id, log, progress=None):
        self.eng, self.log = eng, log
        self.every = max(16, int(cfg["checkpoint_every"]))
        self.park_max = int(cfg["park_max"])
        # host-memory budget: a checkpoint is a copy of the recurrent state (19 MB on the 2B, 50 MB on the 9B), and a
        # parked conversation also keeps its K/V cache; space the checkpoints so that every conversation this cache
        # may hold fits (default: 20% of the host RAM)
        state = sum(m.nbytes for m in eng.M) + sum(c.nbytes for c in eng.conv)
        kv_full = sum(k.nbytes for k in eng.Kc) + sum(v.nbytes for v in eng.Vc)
        budget = int(cfg.get("checkpoint_budget_mb") or 0) * 2 ** 20 or _ram_bytes() // 5
        avail = budget - self.park_max * kv_full
        need = (self.park_max + 1) * state * eng.ctx
        every = -(-need // max(avail, 1) // 256) * 256 if avail > 0 else eng.ctx // 4
        if every > self.every:
            self.every = min(every, max(256, eng.ctx // 4))
        log("prompt cache: a checkpoint every %d tokens (%.0f MB each), %d parked conversations, budget %d MB"
            % (self.every, state / 2 ** 20, self.park_max, budget // 2 ** 20))
        # dense agent-opening checkpoints below dense_limit, every KINDI_DENSE_SPACING tokens (0: none). Each is a full
        # state copy (152 MB on the 27B); 256 was the old spacing (16 copies below 4096)
        self.dense, self.dense_limit = int(os.environ.get("KINDI_DENSE_SPACING", "256")), min(4096, eng.ctx // 2)
        # everything the cache keeps in host RAM - checkpoints (active and parked), parked end states and K/V - stays
        # under the budget: the oldest goes first (see _enforce)
        self.state_bytes, self.budget = state, budget
        self.kv_tok = kv_full // eng.ctx                    # K/V bytes per token
        self.keep_pos = []                                  # opening_end checkpoints (last 2), never dropped as extras
        self.prefix_min = int(cfg["prefix_min"])
        self.progress = progress or {}
        self.tokens, self.ckpt, self.parked = [], {}, []
        self.fingerprint = engine_fingerprint(eng, cfg)
        log("engine fingerprint %s: %s" % (self.fingerprint[0], json.dumps({k: v for k, v in self.fingerprint[1].items()
                                                                           if k != "code"})))
        self.openings = Openings(eng, os.path.join(cfg["state_dir"], "openings", pack_id + "-e" + self.fingerprint[0]),
                                 int(cfg["prefix_max"]), log, fingerprint=self.fingerprint)
        self.slot = getattr(eng, "ckpt_slot", None)          # KINDI_CKPT_VRAM (made by the engine before its fit)

    # ---- state copies
    def _dense(self, p):
        """Agents start every session with the same long opening (system prompt + tools) that differs only near its
        end (the working folder, the date): checkpoints every dense tokens below dense_limit let a new session
        resume a few tokens before its first difference instead of at 0."""
        return self.dense > 0 and p < self.dense_limit and p % self.dense == 0

    def _save_ckpt(self, pos):
        t0 = time.time()
        self._save_ckpt1(pos)
        self._tsave = getattr(self, "_tsave", 0.0) + time.time() - t0

    def _save_ckpt1(self, pos):
        if pos not in self.ckpt:
            if self.slot is not None:
                sl = self.slot
                # the checkpoint in the slot whose host copy is not made yet: dropped if it is one of a request's own
                # (its end or generation-prompt boundary), copied first if it is an opening, dense or evenly spaced one
                key = next((q for q, v in self.ckpt.items() if v is sl.owner), None) if sl.pending() else None
                drop = key is not None and key not in self.keep_pos and not self._dense(key) and key % self.every
                if drop:
                    del self.ckpt[key]
                self.ckpt[pos] = sl.save(drop or key is None)
            else:
                self.ckpt[pos] = ([_host_ckpt(m) for m in self.eng.M], [_host_ckpt(c) for c in self.eng.conv])
            # besides the evenly spaced checkpoints, every request adds two (its generation prompt and its end): keep
            # only the two latest of those, or a long agent session collects one state copy per turn (a leak of
            # ~100 MB per turn on the 9B)
            extra = sorted(p for p in self.ckpt if p % self.every and not self._dense(p) and p not in self.keep_pos)
            for p in extra[:-2]:
                del self.ckpt[p]
            self._enforce()

    def _held(self):
        """host bytes held: distinct checkpoint copies (a parked conversation shares some with the active one), plus
        each parked conversation's end state and K/V"""
        seen, tot = set(), 0
        for d in [self.ckpt] + [P["ckpt"] for P in self.parked]:
            for v in d.values():
                if id(v) not in seen:
                    seen.add(id(v))
                    tot += self.state_bytes
        for P in self.parked:
            tot += self.state_bytes + P["kv_bytes"]
        return tot

    def _enforce(self):
        """Drop the oldest until the cache fits the budget: the oldest parked conversation's checkpoints, then that
        conversation, then the active conversation's oldest checkpoints (the two newest and opening_end are kept)."""
        dropped = 0
        while self._held() > self.budget:
            if self.parked:
                P = self.parked[-1]
                if P["ckpt"]:
                    del P["ckpt"][next(iter(P["ckpt"]))]
                else:
                    self.parked.pop()
            else:
                cand = [q for q in self.ckpt if q not in self.keep_pos][:-2]
                if not cand:
                    break
                del self.ckpt[cand[0]]
            dropped += 1
        if dropped:
            self.log("prompt cache: %d copies dropped to stay within %d MB (%d MB held)"
                     % (dropped, self.budget // 2 ** 20, self._held() // 2 ** 20))


    def _restore(self, st):
        if self.slot is not None and self.slot.restore(st):
            return
        self._set_state(*st)

    def _set_state(self, Ms, Cs):
        for a, h in zip(self.eng.M, Ms):
            _set_host(a, h)
        for a, h in zip(self.eng.conv, Cs):
            _set_host(a, h)

    def _park(self):
        n = len(self.tokens)
        if n < self.prefix_min or self.park_max <= 0:
            return
        # a parked conversation needs its end state and K/V in host RAM; one that cannot fit next to the active
        # conversation's minimum (3 checkpoints) is not parked (its next turn reads from the last shared checkpoint)
        kvb = n * self.kv_tok
        if kvb + self.state_bytes > self.budget - 3 * self.state_bytes:
            self.log("prompt cache: a %d-token conversation (%d MB of K/V) does not fit the budget - not parked"
                     % (n, kvb // 2 ** 20))
            return
        toks = list(self.tokens)
        if self.slot is not None and self.slot.pending():      # the parked copy keeps it: its host copy first
            self.slot.complete(self.slot.owner)
        self.parked = [P for P in self.parked if _lcp(P["tokens"], toks) < 0.9 * len(P["tokens"])]
        del self.parked[self.park_max - 1:]                 # make room first: never two parked K/V copies at once
        self.parked.insert(0, {
            "tokens": toks, "ckpt": dict(self.ckpt), "kv_bytes": kvb,
            "end": ([_host(m) for m in self.eng.M], [_host(c) for c in self.eng.conv]),
            "K": [_host(k[:n]) for k in self.eng.Kc], "V": [_host(v[:n]) for v in self.eng.Vc]})
        del self.parked[self.park_max:]
        self._enforce()

    def _resume_in(self, P, T):
        L = _lcp(P["tokens"], T)
        if L == len(P["tokens"]) and L < len(T):
            return L
        c = [p for p in P["ckpt"] if p <= min(L, len(T) - 1)]
        return max(c) if c else 0

    def _unpark(self, i, pos):
        """resume parked conversation i at pos. Resuming at its end takes it out of the parked list; resuming
        inside it (another conversation that shares its first pos tokens: the same system prompt and tools, a
        common history) copies the state out and leaves it parked, so its own next turn still resumes at its end
        (KINDI_PARK_SHARE=1, with branch parking; the radix-tree behaviour for shared prefixes)"""
        share = (pos < len(self.parked[i]["tokens"]) and os.environ.get("KINDI_PARK_BRANCH", _PARK_DEFAULT) == "1"
                 and os.environ.get("KINDI_PARK_SHARE", "1") == "1")
        P = self.parked[i] if share else self.parked.pop(i)
        self._restore(P["end"] if pos == len(P["tokens"]) else P["ckpt"][pos])
        for l in range(len(self.eng.Kc)):
            _to_dev(self.eng.Kc[l], pos, P["K"][l])
            _to_dev(self.eng.Vc[l], pos, P["V"][l])
        self.tokens = P["tokens"][:pos]
        self.ckpt = {p: v for p, v in P["ckpt"].items() if p <= pos}

    # ---- the request path
    def prepare(self, T, boundary, opening_end=None, cancel=None):
        """Make the engine state = T processed. Returns (greedy next token, tokens reused).
        boundary: where the generation prompt starts (checkpointed too).
        opening_end: where T's first message (system prompt + tools) ends. An opening is saved only when the two
        requests part before that point (+ a few tokens): a new session, or an agent that edits its system prompt -
        not the next turn of the same conversation, which always parts near its own end."""
        tp0 = time.time()
        self._tsave = 0.0
        if self.slot is not None:
            self.slot.pause()
        if opening_end is None:
            opening_end = len(T)
        if 0 < opening_end < len(T) - 1 and opening_end not in self.keep_pos:
            self.keep_pos = (self.keep_pos + [opening_end])[-2:]      # checkpointed below; kept across turns
        off = os.environ.get("KINDI_PROMPT_CACHE", "1") == "0"     # read every prompt from the start (for comparisons)
        if off:
            self.eng.reset(); self.tokens = []; self.ckpt.clear(); self.parked = []
        S = self.tokens
        nS = len(S)
        L = _lcp(S, T)
        if L == nS and L < len(T):
            a_start = L
        else:
            c = [p for p in self.ckpt if p <= min(L, len(T) - 1)]
            a_start = max(c) if c else 0
        best_i, best_p = None, a_start
        for i, P in enumerate(self.parked):
            p = self._resume_in(P, T)
            if p > best_p:
                best_i, best_p = i, p
        branch = os.environ.get("KINDI_PARK_BRANCH", _PARK_DEFAULT) == "1" and nS - L >= self.prefix_min
        if L < nS and (L < 0.5 * nS or branch):                     # an unrelated prompt, or a branch: park it
            chosen = self.parked.pop(best_i) if best_i is not None else None   # out first: parking cannot drop it
            self._park()
            if chosen is not None:
                self.parked.insert(0, chosen)
                best_i = 0
        if best_i is not None:
            try:
                self._unpark(best_i, best_p)
            except Exception as e:
                if "OutOfMemory" not in type(e).__name__:
                    raise
                # the engine state is half restored: drop it and read this prompt from the start
                print("[kindi] could not bring back a parked conversation (%s); reading the prompt from the start"
                      % e, flush=True)
                self.eng.reset(); self.tokens = []; self.ckpt.clear()
                _free_device_pool()
            nS = len(self.tokens)
            L = _lcp(self.tokens, T)
        if L == len(self.tokens) and L < len(T):
            start = L
        else:
            c = [p for p in self.ckpt if p <= min(L, len(T) - 1)]
            if c:
                start = max(c)
                self._restore(self.ckpt[start])
                del self.tokens[start:]
                if (start >= self.prefix_min and L < nS and L <= opening_end + 16
                        and not self.openings.has(self.tokens[:start])):
                    self.openings.save(self.tokens[:start], *self.ckpt[start])
            else:
                self.eng.reset()
                self.tokens = []
                start = 0
        for p in [p for p in self.ckpt if p > start]:
            del self.ckpt[p]
        hit = None if off else self.openings.best(T, start, len(T) - 1)
        if hit is not None:
            try:
                p, Ms, Cs = self.openings.load(hit[0])
            except Exception as e:  # noqa: BLE001 - a missing or damaged opening: read the prompt instead
                self.log("saved opening %s is not usable (%s): reading the prompt" % (hit[0], e))
                self.openings.entries.pop(hit[0], None)
                shutil.rmtree(os.path.join(self.openings.root, hit[0]), ignore_errors=True)
                self.eng.reset()                  # the load may have set part of the state
                self.tokens, self.ckpt, start = [], {}, 0
            else:
                self.tokens = list(T[:p])
                self.ckpt = {p: (Ms, Cs)}
                start = p
        known = getattr(self.eng, "spec_known", None)
        if known is not None and os.environ.get("KINDI_SPEC_KNOWN", "1") == "1":
            known(T[:start])                      # the drafters' history: the tokens of the state just restored
        tp1 = time.time()
        self.log("cache: %d prompt tokens, %d shared with the last request, resumed at %d" % (len(T), L, start))
        nxt = None
        self.progress.update(phase="reading prompt", cached=start, read=0)
        ck = lambda p: p > start and (p % self.every == 0 or p == boundary or p == opening_end  # noqa: E731
                                      or self._dense(p))
        batch = getattr(self.eng, "feed_batch", None)
        pos = start
        while pos < len(T):
            if cancel is not None and cancel.is_set():
                raise Cancelled("the client went away after %d of %d prompt tokens" % (pos - start, len(T) - start))
            if ck(pos):
                self._save_ckpt(pos)
            if pos == len(T) - 1:
                nxt = self.eng.step(T[pos], pos)
                end = pos + 1
            elif batch is not None:
                # prompt tokens in batches (the engine shares work between them), never across a checkpoint
                end = pos + 1
                while end < len(T) - 1 and not ck(end) and end - pos < getattr(self.eng, "batch_max", 16):
                    end += 1
                batch(T[pos:end], pos)
            else:
                self.eng.feed(T[pos], pos)
                end = pos + 1
            self.tokens.extend(T[pos:end])
            pos = end
            self.progress["read"] = pos - start
        tp2 = time.time()
        self._save_ckpt(len(T))
        if os.environ.get("KINDI_CACHE_TIMING") == "1":
            self.log("cache timing: match/restore %.3fs, read+step %.3fs (checkpoints inside %.3fs), end checkpoint %.3fs"
                     % (tp1 - tp0, tp2 - tp1, self._tsave - (time.time() - tp2), time.time() - tp2))
        return nxt, start

    def idle(self):
        """the server finished a request (KINDI_CKPT_VRAM: the newest checkpoint goes to host RAM now)"""
        if self.slot is not None:
            self.slot.idle()

    def step(self, tok):
        nxt = self.eng.step(tok, len(self.tokens))
        self.tokens.append(tok)
        return nxt
