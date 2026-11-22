"""The card's expert cache for the Flash-Next card engine (kindi.qwen4exp_gpu): slots grouped by expert size, filled
from the usage plan, and (KINDI_QXG_ADAPT=1) kept up to date with the experts the conversation asks for, through the
shared F9 policy (kindi/moeaccel.c ma_cache: aged counts, swaps at the end of a step, a new expert used only after
its copy has landed).

Experts of different layers have different sizes (1.5-2.7 MB: the gate/up and down formats differ by layer), so the
slots are grouped in classes of one size: one buffer per class on the card, shared by all the layers of that class;
a layer's slot table maps its experts to slots of its class's buffer. The card kernels read hot + slot * rowbytes,
so they need no change.

Which side computes a pick never changes its value (the card's and the CPU's expert kernels give the same bits), so
the cache changes the speed, not the output."""
import ctypes
import os

import numpy as np
import torch

from kindi import qwen4exp_quant as Q


# ---------------------------------------------------------------- the elastic part (KINDI_QXG_EXT)
class _Prop(ctypes.Structure):          # CUmemAllocationProp
    _fields_ = [("type", ctypes.c_int), ("htype", ctypes.c_int), ("loc_type", ctypes.c_int), ("loc_id", ctypes.c_int),
                ("win32", ctypes.c_void_p), ("compression", ctypes.c_ubyte), ("rdma", ctypes.c_ubyte),
                ("usage", ctypes.c_ushort), ("reserved", ctypes.c_ubyte * 4)]


class _Access(ctypes.Structure):        # CUmemAccessDesc
    _fields_ = [("loc_type", ctypes.c_int), ("loc_id", ctypes.c_int), ("flags", ctypes.c_int)]


class _CAI(object):                     # a raw card range as a tensor (no ownership)
    def __init__(self, ptr, shape):
        self.__cuda_array_interface__ = {"shape": shape, "typestr": "|u1", "data": (ptr, False), "version": 3}


class VBuf(object):
    """one class buffer in the card's virtual address space (CUDA virtual memory management): rows [0, base) always
    backed by card memory, rows [base, base + ext) backed only while the decode runs (map_ext / unmap_ext). The
    addresses never move, so the decode graphs and the kernels (hot + slot * rowbytes) stay valid while the tail
    comes and goes."""

    def __init__(self, cu, dev, rows_base, rows_ext, rb):
        self.cu, self.dev, self.rb = cu, dev, rb
        p = _Prop()
        p.type, p.loc_type, p.loc_id = 1, 1, dev            # pinned, on the device
        self.prop = p
        g = ctypes.c_size_t()
        self._chk(cu.cuMemGetAllocationGranularity(ctypes.byref(g), ctypes.byref(p), 0), "granularity")
        self.gran = g.value
        up = lambda n: (n + self.gran - 1) // self.gran * self.gran                # noqa: E731
        self.base_bytes = up(max(rows_base, 1) * rb)
        self.rows_ext_max = rows_ext
        self.ext_bytes_max = up(rows_ext * rb) if rows_ext else 0
        self.va = ctypes.c_uint64()
        self._chk(cu.cuMemAddressReserve(ctypes.byref(self.va), ctypes.c_size_t(self.base_bytes + self.ext_bytes_max),
                                         ctypes.c_size_t(0), ctypes.c_uint64(0), ctypes.c_ulonglong(0)), "reserve")
        self.h_base = self._map(self.va.value, self.base_bytes)
        self.h_ext, self.ext_bytes = None, 0
        self.rows_total = rows_base + rows_ext
        self.tensor = torch.as_tensor(_CAI(self.va.value, (self.rows_total, rb)), device="cuda:%d" % dev)

    def _chk(self, r, what):
        if r:
            raise RuntimeError("KINDI_QXG_EXT: %s failed (CUDA driver error %d)" % (what, r))

    def _map(self, ptr, n):
        h = ctypes.c_ulonglong()
        self._chk(self.cu.cuMemCreate(ctypes.byref(h), ctypes.c_size_t(n), ctypes.byref(self.prop),
                                      ctypes.c_ulonglong(0)), "create")
        self._chk(self.cu.cuMemMap(ctypes.c_uint64(ptr), ctypes.c_size_t(n), ctypes.c_size_t(0), h,
                                   ctypes.c_ulonglong(0)), "map")
        a = _Access(1, self.dev, 3)                          # read and write from this device
        self._chk(self.cu.cuMemSetAccess(ctypes.c_uint64(ptr), ctypes.c_size_t(n), ctypes.byref(a), ctypes.c_size_t(1)),
                  "access")
        return h

    def map_ext(self, rows):
        """back the first rows of the tail; returns the rows backed (granularity: whole multiples of gran)"""
        n = min(self.ext_bytes_max, (rows * self.rb + self.gran - 1) // self.gran * self.gran)
        if n <= 0 or self.h_ext is not None:
            return 0
        self.h_ext = self._map(self.va.value + self.base_bytes, n)
        self.ext_bytes = n
        return min(self.rows_ext_max, n // self.rb)

    def unmap_ext(self):
        if self.h_ext is None:
            return
        self._chk(self.cu.cuMemUnmap(ctypes.c_uint64(self.va.value + self.base_bytes), ctypes.c_size_t(self.ext_bytes)),
                  "unmap")
        self._chk(self.cu.cuMemRelease(self.h_ext), "release")
        self.h_ext, self.ext_bytes = None, 0


class CardCache(object):
    def __init__(self, gm, dv, layers, place, ext=None):
        self.g, self.dv = gm, dv
        m = gm.ref
        W = m.w
        self.ne = m.n_exp
        # the bytes of an expert's gate, up and down in the file, per layer: (host address, bytes per expert)
        self.src = {}
        for l in layers:
            parts = []
            for k in ("gate", "up", "down"):
                name = "blk.%d.ffn_%s_exps.weight" % (l, k)
                mm, base, ne, ty = W.t[name]
                n = Q.nbytes(ty, int(np.prod(ne))) // ne[2]
                parts.append((mm, base, n))
            self.src[l] = parts
        eb = {l: sum(p[2] for p in self.src[l]) for l in layers}
        self.classes = {}                              # rowbytes -> [layers]
        for l in layers:
            self.classes.setdefault(eb[l], []).append(l)
        # KINDI_QXG_FETCH > 0 (the PCIe share, item 7; needs KINDI_QXG_PIN=1 and the mailbox pass): each class buffer
        # gets that many spare rows after its slots, which the card fills with fetched experts layer by layer
        self.nspare = max(0, min(64, int(os.environ.get("KINDI_QXG_FETCH", "0") or 0)))
        self.spare0 = {}
        self.cls_of, self.buf, self.owner = {}, {}, {}
        self.slot_pin = {}
        self.pend = []
        # KINDI_QXG_EXT: ext {layer: [experts, priority order]}: the class buffers get a tail of slots after the
        # plan's (rows base_rows .. ) that has card memory only while the decode runs (ext_on / ext_off: CUDA virtual
        # memory, the addresses never move). The tail's slots are cache slots like the others (the adaptive policy
        # fills them); while the tail is off its experts are the CPU's, and no copy goes into it
        self.base_rows, self.vbuf, self.ext_cand = {}, {}, {}
        self.ext_live, self.ext_dead = False, False
        for ci, (rb, ls) in enumerate(sorted(self.classes.items())):
            cap = sum(len(place[l]) for l in ls)
            self.spare0[ci] = max(cap, 1)
            self.base_rows[ci] = max(cap, 1)
            cand = []
            if ext and not self.nspare:
                for l in ls:
                    cand += [(l, e) for e in ext.get(l, [])]
                rank = {(l, e): i for l in ls for i, e in enumerate(ext.get(l, []))}
                cand.sort(key=lambda le: rank[le])                 # the layers interleaved by rank
            buf, nx = None, 0
            if cand:
                try:
                    from kindi.qwen4exp_gpu import lib
                    vb = VBuf(lib().cu, dv.index if hasattr(dv, "index") else int(dv), max(cap, 1), len(cand), rb)
                    self.vbuf[ci], buf, nx = vb, vb.tensor, len(cand)
                    self.ext_cand[ci] = cand
                except Exception as ex:                      # no virtual memory management: the static cache only
                    gm.log("KINDI_QXG_EXT: off (%s)" % ex)
            if buf is None:
                buf = torch.empty((max(cap, 1) + self.nspare, rb), dtype=torch.uint8, device=dv)
            owner = np.full((max(cap, 1) + nx, 2), -1, np.int32)   # slot -> (layer, expert)
            s = 0
            for l in ls:
                self.cls_of[l] = ci
                slot = np.full(self.ne, -1, np.int32)
                for e in place[l]:
                    buf[s].copy_(torch.from_numpy(self._row(l, e)))
                    slot[e] = s
                    owner[s] = (l, e)
                    s += 1
                pin = torch.from_numpy(slot).pin_memory()       # pinned: one async copy per changed layer
                self.slot_pin[l] = pin
                gm.slot[l] = pin.numpy()
                gm.hot[l] = buf
                gm.hot_count[l] = len(place[l])
            self.buf[ci], self.owner[ci] = buf, owner
        self.rowbytes = {ci: rb for ci, (rb, ls) in enumerate(sorted(self.classes.items()))}
        self.adapt = None
        self.pending = []
        self.copies = 0
        self.steps = 0

    # ------------------------------------------------------------ the elastic tail (KINDI_QXG_EXT)
    def ext_on(self, dec=None):
        """the decode starts: the tails get card memory again; the experts their slots held (or, the first time,
        the plan's next experts) are copied back in and the slot tables point at them again"""
        if self.ext_live or self.ext_dead or not self.vbuf:
            return 0
        n_in = 0
        # rows to back per class: all of them, or (ext_keep, the default) as many as leave ext_keep bytes of the card
        # free, shared over the classes in proportion to their tails
        want = {ci: vb.rows_ext_max for ci, vb in self.vbuf.items()}
        keep = getattr(self, "ext_keep", None)
        if keep is not None:
            free = torch.cuda.mem_get_info(self.dv)[0]
            tot = sum(vb.ext_bytes_max for vb in self.vbuf.values()) or 1
            budget = max(0.0, free - keep)
            for ci, vb in self.vbuf.items():
                gb = budget * vb.ext_bytes_max / tot // vb.gran * vb.gran      # whole granules: no rounding up
                want[ci] = min(vb.rows_ext_max, int(gb // vb.rb))
        self.ext_rows = {}
        try:
            for ci, vb in self.vbuf.items():
                self.ext_rows[ci] = vb.map_ext(want[ci]) if want[ci] > 0 else 0
        except RuntimeError as ex:                         # no room: the tail stays off (its slots get no copies)
            for vb in self.vbuf.values():
                try:
                    vb.unmap_ext()
                except RuntimeError:
                    pass
            self.ext_dead = True
            self.g.log("elastic cache: the tail could not be mapped (%s); off from now on" % ex)
            return 0
        dirty = set()
        for ci, vb in self.vbuf.items():
            owner, b0 = self.owner[ci], self.base_rows[ci]
            if not getattr(self, "_ext_seeded", False):     # the first time: the plan's next experts
                for i, (l, e) in enumerate(self.ext_cand[ci]):
                    if owner[b0 + i, 0] < 0 and self.g.slot[l][e] < 0:
                        owner[b0 + i] = (l, e)
            live = b0 + self.ext_rows.get(ci, 0)
            if getattr(self, "adapt", None):
                self.C.ma_cache_set_live(self.adapt[ci], live)
            for sl in range(b0, live):
                l, e = owner[sl]
                if l < 0 or self.g.slot[l][e] >= 0:
                    continue
                self._copy_in(vb.tensor[sl].data_ptr(), l, e)
                self._set(dec, l, e, sl)
                dirty.add(l)
                n_in += 1
        self._ext_seeded = True
        torch.cuda.synchronize(self.dv)                     # the copies have landed before any step reads them
        if dec is not None:
            for l in dirty:
                dec.slotmap[l].copy_(self.slot_pin[l], non_blocking=True)
        self.ext_live = True
        return n_in

    def ext_off(self, dec=None):
        """a long prompt read needs the memory: the copies in flight finish, the tails' experts go back to the CPU
        (slot tables first; the slots keep their owners, for ext_on), then the memory goes back to the driver"""
        if not self.ext_live:
            return
        self._drain(dec)
        dirty = set()
        for ci in self.vbuf:
            owner, b0 = self.owner[ci], self.base_rows[ci]
            for sl in range(b0, len(owner)):
                l, e = owner[sl]
                if l >= 0 and self.g.slot[l][e] == sl:
                    self.g.slot[l][e] = -1
                    if dec is not None and getattr(dec, "ma_oncard", None) is not None:
                        dec.ma_oncard[l, e] = 0
                    dirty.add(l)
        if dec is not None:
            for l in dirty:
                dec.slotmap[l].copy_(self.slot_pin[l], non_blocking=True)
        torch.cuda.synchronize(self.dv)
        for ci, vb in self.vbuf.items():
            vb.unmap_ext()
            if getattr(self, "adapt", None):
                self.C.ma_cache_set_live(self.adapt[ci], self.base_rows[ci])
        self.ext_rows = {}
        self.ext_live = False

    def _drain(self, dec):
        """wait for the helper's copies, then admit them (as after_step does)"""
        if getattr(self, "helper", None) is None:
            return
        import threading
        ev = threading.Event()
        self.jobs.put(("sync", ev))
        ev.wait()
        torch.cuda.synchronize(self.dv)
        while not self.done.empty():
            self.pend.append(self.done.get())
        for ci, s, l, e, ev2 in self.pend:
            self.C.ma_cache_admit(self.adapt[ci], s)
            self.owner[ci][s] = (l, e)
            self._set(dec, l, e, s)
            if dec is not None:
                dec.slotmap[l].copy_(self.slot_pin[l], non_blocking=True)
        self.pend = []

    def _copy_in(self, dst, l, e):
        """expert e of layer l into the card row at dst: from the page-locked file pages (KINDI_QXG_PIN=1) by
        copy engine, else through host memory"""
        l, e = int(l), int(e)                             # numpy int32 from the owner table: e * n overflows
        if getattr(self, "pinned", False):
            from kindi.qwen4exp_gpu import lib
            cu = lib().cu
            st = ctypes.c_void_p(torch.cuda.current_stream(self.dv).cuda_stream)
            for mm, base, n in self.src[l]:
                a = np.frombuffer(mm, np.uint8).ctypes.data + base + e * n
                cu.cuMemcpyHtoDAsync_v2(ctypes.c_uint64(dst), ctypes.c_void_p(a), ctypes.c_size_t(n), st)
                dst += n
        else:
            row = torch.from_numpy(self._row(l, e))
            torch.as_tensor(_CAI(dst, (row.numel(),)), device=self.dv).copy_(row)

    def _row(self, l, e):
        return np.concatenate([np.frombuffer(mm, np.uint8, n, base + e * n) for mm, base, n in self.src[l]])

    # ------------------------------------------------------------ the adaptive part (KINDI_QXG_ADAPT=1)
    def start_adaptive(self, window=None, swap=None, decay=None):
        """the F9 policy per class; returns False when it is not available (no qx_cpu library, e.g. on Windows)"""
        try:
            from kindi import qx_cpu
            C = qx_cpu.lib()
        except Exception:
            return False
        I, V, Fl = ctypes.c_int, ctypes.c_void_p, ctypes.c_float
        C.ma_cache_new.argtypes = [I, I, I, I, I, Fl]
        C.ma_cache_new.restype = V
        C.ma_cache_exclude_layer.argtypes = [V, I]
        C.ma_cache_seed.argtypes = [V, V, V, I]
        C.ma_cache_seed.restype = I
        C.ma_cache_note.argtypes = [V, I, V, I]
        C.ma_cache_tick.argtypes = [V, V, I]
        C.ma_cache_tick.restype = I
        C.ma_cache_admit.argtypes = [V, I]
        C.ma_cache_set_live.argtypes = [V, I]
        C.ma_cache_stats.argtypes = [V, V]
        self.C = C
        window = window or int(os.environ.get("KINDI_QXG_ADAPT_WINDOW", "4"))
        swap = swap or int(os.environ.get("KINDI_QXG_ADAPT_SWAP", "64"))
        decay = decay or float(os.environ.get("KINDI_QXG_ADAPT_DECAY", "0.9"))
        self.adapt = {}
        nl = self.g.L
        for ci, owner in self.owner.items():
            c = C.ma_cache_new(nl, self.ne, len(owner), window, swap, decay)
            ls = set(l for l, ci_ in self.cls_of.items() if ci_ == ci)
            for l in range(nl):
                if l not in ls:
                    C.ma_cache_exclude_layer(c, l)
            fill = owner[owner[:, 0] >= 0]
            lay, exp = np.ascontiguousarray(fill[:, 0]), np.ascontiguousarray(fill[:, 1])
            C.ma_cache_seed(c, lay.ctypes.data, exp.ctypes.data, len(fill))
            if ci in self.vbuf:                           # the tail has memory only while it is mapped (ext_on)
                C.ma_cache_set_live(c, self.base_rows[ci] + (getattr(self, "ext_rows", {}).get(ci, 0)
                                                             if self.ext_live else 0))
            self.adapt[ci] = c
        self.swapbuf = np.zeros((256, 3), np.int32)
        self.stream = torch.cuda.Stream(self.dv)
        # KINDI_QXG_PIN=1: page-lock the experts in the mapped file (read-only), so a swap is one DMA copy from the
        # file's pages, with no CPU copy (the copies then take no memory bandwidth from the CPU experts beyond the
        # DMA itself); it locks the experts' RAM, so it is opt-in. Otherwise each swap is copied through pinned
        # staging (one CPU copy of the expert); the number of swaps per round (KINDI_QXG_ADAPT_SWAP) is the budget.
        self.pinned = os.environ.get("KINDI_QXG_PIN", "0") == "1" and self._pin()
        return True

    def _pin(self):
        from kindi.qwen4exp_gpu import lib
        cu = lib().cu
        page = 4096
        free0 = torch.cuda.mem_get_info(self.dv)[0]          # the driver's bookkeeping for the pinned pages takes card memory
        regions = {}
        for l, parts in self.src.items():
            for mm, base, n in parts:
                a = np.frombuffer(mm, np.uint8).ctypes.data
                regions.setdefault(a, []).append((base, base + n * self.ne))
        self._addr = {}
        for a, rs in regions.items():
            rs.sort()
            merged = []
            for lo, hi in rs:
                lo, hi = lo // page * page, (hi + page - 1) // page * page
                if merged and lo <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], hi)
                else:
                    merged.append([lo, hi])
            for lo, hi in merged:
                # read-only, and mapped into the card's address space (the PCIe share reads it from the card)
                r = cu.cuMemHostRegister_v2(ctypes.c_void_p(a + lo), ctypes.c_size_t(hi - lo), 0x08 | 0x02)
                if r:
                    self.g.log("KINDI_QXG_PIN=1: page-locking the experts failed (CUDA error %d); staging copies" % r)
                    return False
                d = ctypes.c_uint64()
                if cu.cuMemHostGetDevicePointer_v2(ctypes.byref(d), ctypes.c_void_p(a + lo), 0) == 0:
                    self._addr[a + lo] = (hi - lo, d.value)
        self.g.log("KINDI_QXG_PIN=1: card memory free %.3f -> %.3f GB around the page-locking" % (
            free0 / 1e9, torch.cuda.mem_get_info(self.dv)[0] / 1e9))
        self.g.log("KINDI_QXG_PIN=1: experts page-locked (%.1f GB)" % (
            sum(n * self.ne for parts in self.src.values() for _, _, n in parts) / 1e9))
        return True

    def device_src(self, l):
        """the card's addresses of expert 0's gate, up and down of layer l in the page-locked file pages, and their
        bytes per expert -> ([3 addresses], [3 sizes]); None when the pages are not mapped"""
        out, nb = [], []
        for mm, base, n in self.src[l]:
            h = np.frombuffer(mm, np.uint8).ctypes.data + base
            hit = [(lo, d) for lo, (sz, d) in getattr(self, "_addr", {}).items() if lo <= h < lo + sz]
            if not hit:
                return None
            out.append(hit[0][1] + (h - hit[0][0]))
            nb.append(n)
        return out, nb

    def note(self, l, ids):
        """the picks of layer l in this step (int32 array); the mailbox pass notes them in C instead"""
        if self.adapt is not None and l in self.cls_of:
            ids = np.ascontiguousarray(ids, np.int32)
            self.C.ma_cache_note(self.adapt[self.cls_of[l]], l, ids.ctypes.data, ids.size)

    def note_table(self, nl):
        """the per-layer cache pointers for the mailbox step (qx_ma_ctx_set_note), kept alive here"""
        arr = (ctypes.c_void_p * nl)()
        for l in range(nl):
            if l in self.cls_of and self.adapt is not None:
                arr[l] = self.adapt[self.cls_of[l]]
        self._note_arr = arr
        return arr

    def _start_helper(self):
        """the copy thread: takes (class, slot, layer, expert, event to wait for) jobs, copies the expert into its slot
        on its own stream (directly from page-locked file pages, or through a ring of pinned staging slots made once),
        records an event per job; the decode thread admits a job once its event has completed"""
        import queue
        import threading
        from kindi.qwen4exp_gpu import lib
        self.cu = lib().cu
        self.jobs = queue.Queue()
        self.done = queue.Queue()
        nring = int(os.environ.get("KINDI_QXG_ADAPT_RING", "32"))
        if not self.pinned:
            rb = max(self.rowbytes.values())
            self.ring = torch.empty((nring, rb), dtype=torch.uint8).pin_memory()
            self.ring_ev = [None] * nring
            self.ring_i = 0
        self.addr = {l: [(np.frombuffer(mm, np.uint8).ctypes.data + base, n) for mm, base, n in parts]
                     for l, parts in self.src.items()}

        main_cpus = set(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else set()

        def run():
            torch.cuda.set_device(self.dv)
            # off the decode thread's core: the decode thread is pinned to the pool's first core (KINDI_QX_HOSTPIN),
            # and a thread inherits that, so the copies (a staging memmove per expert without KINDI_QXG_PIN) ran on
            # the decode thread's core and took its time (rented RTX 3080 test machine: about 3.5 ms a step); KINDI_QXG_HELPER_CPUS sets
            # the cores (default: every core but the decode thread's)
            if hasattr(os, "sched_setaffinity"):
                try:
                    want = os.environ.get("KINDI_QXG_HELPER_CPUS")
                    if want:
                        from kindi.qx_cpu import cpus as _cpus
                        cs = set(_cpus(want))
                    else:
                        cs = set(range(os.cpu_count() or 1)) - main_cpus
                    if cs:
                        os.sched_setaffinity(0, cs)
                except (OSError, ValueError):
                    pass
            while True:
                job = self.jobs.get()
                if job is None:
                    return
                if job[0] == "sync":                      # ext_off: every earlier copy has been issued
                    self.stream.synchronize()
                    job[1].set()
                    continue
                ci, sl, l, e, after = job
                self.stream.wait_event(after)
                dst = self.buf[ci][sl].data_ptr()
                st = ctypes.c_void_p(self.stream.cuda_stream)
                if self.pinned:
                    for a, n in self.addr[l]:
                        self.cu.cuMemcpyHtoDAsync_v2(ctypes.c_uint64(dst), ctypes.c_void_p(a + e * n), ctypes.c_size_t(n), st)
                        dst += n
                else:
                    i = self.ring_i
                    self.ring_i = (i + 1) % len(self.ring_ev)
                    if self.ring_ev[i] is not None:
                        self.ring_ev[i].synchronize()            # that staging slot's last copy has finished
                    hp, off = self.ring[i].data_ptr(), 0
                    for a, n in self.addr[l]:
                        ctypes.memmove(hp + off, a + e * n, n)
                        off += n
                    self.cu.cuMemcpyHtoDAsync_v2(ctypes.c_uint64(dst), ctypes.c_void_p(hp), ctypes.c_size_t(off), st)
                ev = torch.cuda.Event()
                ev.record(self.stream)
                if not self.pinned:
                    self.ring_ev[i] = ev
                self.done.put((ci, sl, l, e, ev))
        self.helper = threading.Thread(target=run, name="kindi-swaps", daemon=True)
        self.helper.start()
        import atexit
        atexit.register(self.stop_helper)

    def stop_helper(self, timeout=10.0):
        """the copy thread ends after the copies it has (a None job), and is joined: at process exit a daemon thread
        still inside a CUDA call aborted the process ("terminate called without an active exception", B62)"""
        h = getattr(self, "helper", None)
        if h is None or not h.is_alive():
            return
        self.jobs.put(None)
        h.join(timeout)
        try:
            self.stream.synchronize()
        except Exception:
            pass

    def after_step(self, dec):
        """the end of a step (on the decode thread, cheap): admit the copies that have landed, start this round's
        swaps (evictions at once; the copies go to the helper thread), and write the changed layers' slot tables to
        the card with one copy each"""
        if self.adapt is None:
            return
        if getattr(self, "helper", None) is None:
            self._start_helper()
            import atexit
            atexit.register(self.save_hot)
        self.steps += 1
        if self.steps % int(os.environ.get("KINDI_QXG_HOTSET_EVERY", "512")) == 0:
            self.save_hot()
        dirty = set()
        left = []
        while not self.done.empty():
            self.pend.append(self.done.get())
        for p_ in getattr(self, "pend", []):
            ci, s, l, e, ev = p_
            if ev.query():
                self.C.ma_cache_admit(self.adapt[ci], s)
                self.owner[ci][s] = (l, e)
                self._set(dec, l, e, s)
                dirty.add(l)
            else:
                left.append(p_)
        self.pend = left
        after = None
        for ci, c in self.adapt.items():
            n = self.C.ma_cache_tick(c, self.swapbuf.ctypes.data, len(self.swapbuf))
            if n <= 0:
                continue
            if after is None:
                after = torch.cuda.Event()
                after.record(torch.cuda.current_stream(self.dv))     # the work queued so far may read the victims
            for s, l, e in self.swapbuf[:n].tolist():
                if s >= self.base_rows[ci] + (getattr(self, "ext_rows", {}).get(ci, 0) if self.ext_live else 0):
                    continue                              # the tail has no memory now: the expert stays the CPU's
                ol, oe = self.owner[ci][s]
                if ol >= 0:
                    self._set(dec, ol, oe, -1)            # evicted at once (the next step computes it on the CPU)
                    self.owner[ci][s] = (-1, -1)
                    dirty.add(ol)
                self.jobs.put((ci, s, l, e, after))
                self.copies += 1
        for l in dirty:                                    # one small copy per changed layer, before the next step
            dec.slotmap[l].copy_(self.slot_pin[l], non_blocking=True)

    def save_hot(self):
        """the current decode set of this card's layers into the session file (gm.hotset_path; other cards' layers
        kept), read by the next start's placement"""
        path = getattr(self.g, "hotset_path", None)
        if not path or self.adapt is None:
            return
        import json
        try:
            d = json.load(open(path)) if os.path.exists(path) else {}
        except (OSError, ValueError):
            d = {}
        hot = d.get("hot", {})
        for l in self.cls_of:
            hot[str(l)] = []
        for owner in self.owner.values():
            for l, e in owner.tolist():
                if l >= 0 and str(l) in hot:
                    hot[str(l)].append(int(e))
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path + ".tmp", "w") as f:
                json.dump({"hot": hot}, f, separators=(",", ":"))
            os.replace(path + ".tmp", path)
        except OSError:
            pass

    def _set(self, dec, l, e, s):
        self.g.slot[l][e] = s
        if getattr(dec, "ma_oncard", None) is not None:
            dec.ma_oncard[l, e] = 1 if s >= 0 else 0

    def stats(self):
        if self.adapt is None:
            return None
        tot = np.zeros(5, np.int64)
        for c in self.adapt.values():
            a = (ctypes.c_longlong * 5)()
            self.C.ma_cache_stats(c, a)
            tot += np.array(a[:], np.int64)
        return {"picks": int(tot[0]), "hits": int(tot[1]), "swaps": int(tot[2]), "admits": int(tot[3]),
                "windows": int(tot[4])}
