"""Launcher of the Flash-Next card kernels (kindi/qx_gpu.cu): compiled at run time with NVRTC to PTX, loaded with
the CUDA driver API into each card's primary context (the one PyTorch uses), and launched on PyTorch's stream.
No CUDA toolkit is needed: NVRTC comes with PyTorch (or set KINDI_NVRTC to the library), the driver API with the
driver. Works on Linux and on Windows.

Arguments are device pointers (int: tensor.data_ptr()), ints and floats; the caller owns the memory."""
import ctypes
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# the formats of k_fgu (gate/up) and k_fdn (down): values and bytes per weight block (qx_gpu.cu XG<TY>)
XG_PER = {22: 256, 18: 256, 21: 256, 23: 256, 20: 32, 42: 64}
XG_BYTES = {22: 82, 18: 98, 21: 110, 23: 136, 20: 18, 42: 18}
KERNELS = ("k_quant", "k_quant256", "k_gate_up", "k_down", "k_combine", "k_dense", "k_gdn", "k_gate_up_slots", "k_down_slots",
           "k_fgu", "k_fdn",
           "k_combine_sel", "k_dense1", "k_dense8", "k_rms_g", "k_silu_s", "k_mix_out", "k_write", "k_dot",
           "k_swiglu", "k_router", "k_conv1", "k_gdn1", "k_gnorm", "k_dense8m", "k_combine_hot", "k_deq",
           "k_ma_seq", "k_ma_post", "k_ma_wait", "k_dense8T", "k_dense8mT", "k_dense_tc", "k_dense_tcm",
           "k_rms_gT", "k_mix_outT", "k_writeT", "k_dotT", "k_routerT", "k_combine_selT", "k_conv1T", "k_gdn1T",
           "k_gnormT", "k_pick", "k_deq_row", "k_fetch_plan", "k_fetch_copy", "k_fetch_reset", "k_quant_ab", "k_quant_w",
           "k_deq16", "k_gdn4", "k_gdn_conv", "k_gdn_state", "k_gdn_norm", "k_xmm_tc", "k_silu_mul", "k_fetch_apply",
           "k_xtile_iq2s", "k_xtile_iq3xxs", "k_xtile_iq3s", "k_xtile_iq4xs", "k_xtile_iq4nl", "k_xtile_q20", "k_count",
           "k_qsa_prep", "k_qsa_bkey", "k_qsa_sel", "k_qsa_attn", "k_qsa_comb", "k_ts", "k_dense8T2", "k_dense8mT2",
           "k_dense8T2_1", "k_dense8T2_2", "k_dense8T2_3", "k_dense8mT2_1", "k_dense8mT2_2", "k_dense8mT2_3", "k_qx8")


def _find_nvrtc():
    if os.environ.get("KINDI_NVRTC"):
        return os.environ["KINDI_NVRTC"]
    import torch
    dirs = [os.path.join(os.path.dirname(torch.__file__), "lib")]
    for p in sys.path:
        dirs.append(os.path.join(p, "nvidia", "cuda_nvrtc", "lib"))
        dirs.append(os.path.join(p, "nvidia", "cu13", "lib"))
    pats = ["nvrtc64_*_0.dll"] if os.name == "nt" else ["libnvrtc.so*"]
    for d in dirs:
        for pat in pats:
            hits = sorted(h for h in glob.glob(os.path.join(d, pat)) if "builtins" not in h and ".alt." not in h)
            if hits:
                return hits[-1]
    raise OSError("NVRTC not found; set KINDI_NVRTC")


class Kernels(object):
    def __init__(self, arch=None):
        self.nvrtc = ctypes.CDLL(_find_nvrtc())
        self.cu = ctypes.CDLL("nvcuda.dll" if os.name == "nt" else "libcuda.so.1")
        self._chk(self.cu.cuInit(0), "cuInit")
        self.arch = arch
        self.ptx = {}
        self.mod = {}          # device -> {kernel name: CUfunction}

    def _chk(self, r, what):
        if r:
            raise RuntimeError("qx_gpu: %s failed (CUDA driver error %d)" % (what, r))

    def _compile(self, arch):
        if arch in self.ptx:
            return self.ptx[arch]
        src = open(os.path.join(HERE, "qx_gpu.cu"), encoding="utf-8").read().encode()
        hdr = open(os.path.join(HERE, "nbq_grids.h"), encoding="utf-8").read().encode()
        mah = open(os.path.join(HERE, "moeaccel.cuh"), encoding="utf-8").read().encode()
        N = self.nvrtc
        prog = ctypes.c_void_p()
        names = (ctypes.c_char_p * 2)(b"nbq_grids.h", b"moeaccel.cuh")
        srcs = (ctypes.c_char_p * 2)(hdr, mah)
        r = N.nvrtcCreateProgram(ctypes.byref(prog), src, b"qx_gpu.cu", 2, srcs, names)
        if r:
            raise RuntimeError("nvrtcCreateProgram failed (%d)" % r)
        opts = [("--gpu-architecture=" + arch).encode(), b"-default-device", b"--std=c++14",
                ("-DXG_RW=%d" % self.D8RW).encode(),
                ("-DXG_D8MINB=%d" % int(os.environ.get("KINDI_QXG_D8MINB", "3"))).encode(),
                ("-DXG_GL=%d" % self.FS_GL).encode(),
                ("-DXG_I2F=%d" % int(os.environ.get("KINDI_QXG_I2F", "1"))).encode(),
                ("-DXG_D8TMINB=%d" % int(os.environ.get("KINDI_QXG_D8TMINB", "3"))).encode(),
                ("-DXG_ROUTER2=%d" % int(os.environ.get("KINDI_QXG_ROUTER2", "1"))).encode(),
                ("-DXG_Q6T=%d" % int(os.environ.get("KINDI_QXG_Q6T", "1"))).encode(),
                ("-DXG_KQ5=%d" % int(os.environ.get("KINDI_QXG_KQ5", "3"))).encode(),
                ("-DXG_IDENSE=%d" % int(self.IDENSE)).encode()]
        if arch.startswith("sm_") and int(arch[3:]) < 70 and os.environ.get("KINDI_QXG_DLCM", "auto") != "0":
            opts.append(b"-Xptxas=-dlcm=ca")    # the PTX path's CU_JIT_CACHE_MODE, set at compile time for a cubin
        r = N.nvrtcCompileProgram(prog, len(opts), (ctypes.c_char_p * len(opts))(*opts))
        n = ctypes.c_size_t()
        N.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
        log = ctypes.create_string_buffer(n.value)
        N.nvrtcGetProgramLog(prog, log)
        if r:
            raise RuntimeError("NVRTC: qx_gpu.cu did not compile:\n" + log.value.decode(errors="replace"))
        if arch.startswith("sm_"):     # a cubin: loads on any driver of the same CUDA major, unlike PTX (needs a new enough JIT)
            N.nvrtcGetCUBINSize(prog, ctypes.byref(n))
            ptx = ctypes.create_string_buffer(n.value)
            if N.nvrtcGetCUBIN(prog, ptx):
                raise RuntimeError("nvrtcGetCUBIN failed")
        else:
            N.nvrtcGetPTXSize(prog, ctypes.byref(n))
            ptx = ctypes.create_string_buffer(n.value)
            N.nvrtcGetPTX(prog, ptx)
        N.nvrtcDestroyProgram(ctypes.byref(prog))
        self.ptx[arch] = ptx.raw
        return ptx.raw

    def _ctx(self, dev):
        """make card dev's primary context current in this thread; load the module there once"""
        cu = self.cu
        d = ctypes.c_int()
        self._chk(cu.cuDeviceGet(ctypes.byref(d), dev), "cuDeviceGet")
        ctx = ctypes.c_void_p()
        if dev not in self.mod:
            self._chk(cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), d), "cuDevicePrimaryCtxRetain")
            self._chk(cu.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")
            ma, mi = ctypes.c_int(), ctypes.c_int()
            cu.cuDeviceGetAttribute(ctypes.byref(ma), 75, d)          # compute capability major
            cu.cuDeviceGetAttribute(ctypes.byref(mi), 76, d)
            arch = self.arch or "compute_%d%d" % (ma.value, mi.value)
            mod = ctypes.c_void_p()
            ptx = None
            if not self.arch and os.environ.get("KINDI_QXG_CUBIN", "1") != "0":
                try:    # the card's own cubin first; any failure falls back to the PTX below
                    ptx = self._compile("sm_%d%d" % (ma.value, mi.value))
                    self._load(cu, mod, ptx, ma.value)
                except Exception:
                    ptx, mod = None, ctypes.c_void_p()
            if ptx is None:
                ptx = self._compile(arch)
                self._load(cu, mod, ptx, ma.value)
            fns = {}
            for k in KERNELS:
                f = ctypes.c_void_p()
                self._chk(cu.cuModuleGetFunction(ctypes.byref(f), mod, k.encode()), "cuModuleGetFunction " + k)
                fns[k] = f
            self.mod[dev] = (ctx, fns)
        else:
            self._chk(cu.cuCtxSetCurrent(self.mod[dev][0]), "cuCtxSetCurrent")
        return self.mod[dev][1]

    def _load(self, cu, mod, ptx, major):
        if True:
            # KINDI_QXG_DLCM (auto: on below sm_70): global loads cached in L1 (CU_JIT_CACHE_MODE = CA). Maxwell and
            # Pascal cache them only in L2 unless asked; the dense kernels' small per-lane scale reads then each go to
            # L2. Caching does not change any value, so the same bits.
            dl = os.environ.get("KINDI_QXG_DLCM", "auto")
            if dl == "1" or (dl == "auto" and major < 7):
                opt = (ctypes.c_int * 1)(14)                         # CU_JIT_CACHE_MODE
                val = (ctypes.c_void_p * 1)(2)                       # CU_JIT_CACHE_OPTION_CA
                self._chk(cu.cuModuleLoadDataEx(ctypes.byref(mod), ctypes.c_char_p(ptx), 1, opt, val),
                          "cuModuleLoadDataEx")
            else:
                self._chk(cu.cuModuleLoadData(ctypes.byref(mod), ctypes.c_char_p(ptx)), "cuModuleLoadData")

    def launch(self, dev, stream, name, grid, block, args):
        """args: a list of ("p", int) pointers, ("i", int), ("l", int) 64-bit, ("f", float)"""
        fns = self._ctx(dev)
        vals = []
        for kind, v in args:
            if kind == "p":
                vals.append(ctypes.c_void_p(int(v)))
            elif kind == "i":
                vals.append(ctypes.c_int(int(v)))
            elif kind == "l":
                vals.append(ctypes.c_longlong(int(v)))
            else:
                vals.append(ctypes.c_float(float(v)))
        params = (ctypes.c_void_p * len(vals))(*[ctypes.cast(ctypes.pointer(v), ctypes.c_void_p) for v in vals])
        g = tuple(grid) + (1,) * (3 - len(grid))
        r = self.cu.cuLaunchKernel(fns[name], g[0], g[1], g[2], block, 1, 1, 0, ctypes.c_void_p(stream), params, None)
        self._chk(r, "launch " + name)

    # ------------------------------------------------------------ the kernels (dev: card number; st: CUstream)
    def quant(self, dev, st, x, n, q, d):
        nb = n // 32
        self.launch(dev, st, "k_quant", ((nb + 127) // 128,), 128, [("p", x), ("l", nb), ("p", q), ("p", d)])

    # ------------------------------------------------------------ the mailbox (moeaccel.cuh / moeaccel.c)
    def host_mapped(self, dev, nbytes):
        """page-locked host memory mapped into card dev's address space -> (host address, device address)"""
        self._ctx(dev)
        h = ctypes.c_void_p()
        self._chk(self.cu.cuMemHostAlloc(ctypes.byref(h), ctypes.c_size_t(nbytes), 0x01 | 0x02), "cuMemHostAlloc")
        d = ctypes.c_uint64()
        self._chk(self.cu.cuMemHostGetDevicePointer_v2(ctypes.byref(d), h, 0), "cuMemHostGetDevicePointer")
        ctypes.memset(h, 0, nbytes)
        return h.value, d.value

    # ------------------------------------------------------------ the PCIe share of the misses (item 7)
    def fetch_plan(self, dev, st, ids, n, slotmap, nf, l, smax, spare0, lst):
        self.launch(dev, st, "k_fetch_plan", (1,), 32, [("p", ids), ("i", n), ("p", slotmap), ("p", nf), ("i", l),
                                                       ("i", smax), ("i", spare0), ("p", lst)])

    def fetch_copy(self, dev, st, lst, smax, src, nb, buf, rowbytes, spare0, chunk=1 << 16):
        """src: the device addresses of expert 0's gate, up and down; nb: their bytes per expert"""
        nx = (max(nb) + chunk - 1) // chunk
        self.launch(dev, st, "k_fetch_copy", (nx, smax, 3), 256,
                    [("p", lst), ("i", smax), ("p", src[0]), ("p", src[1]), ("p", src[2]), ("l", nb[0]),
                     ("l", nb[1]), ("l", nb[2]), ("p", buf), ("l", rowbytes), ("i", spare0), ("l", chunk)])

    def fetch_apply(self, dev, st, lst, smax, slotmap, spare0):
        self.launch(dev, st, "k_fetch_apply", (1,), max(32, smax), [("p", lst), ("i", smax), ("p", slotmap),
                                                                     ("i", spare0)])

    def fetch_reset(self, dev, st, lst, smax, slotmap):
        self.launch(dev, st, "k_fetch_reset", (1,), max(32, smax), [("p", lst), ("i", smax), ("p", slotmap)])

    # KINDI_QXG_QSAK=1: the decode's QSA attention (one-token steps and verify windows alike) on four kernels instead
    # of PyTorch's per-token ops; changes the decode's bits (the gate decides)
    QSAK = os.environ.get("KINDI_QXG_QSAK", "1") == "1"            # on by default since B32/B38 (gate 174/174)

    def qsa(self, dev, st, T, pos, qg, kx, vx, ik, iq, qn, kn, iqn, ikn, inv, half, eps, nh, nkv, nih, hd, di, ra, nk,
            nb, lmax, kc, vc, rk, bk, qo, qi, sc, sel, ok, part, out):
        """the decode attention of T tokens at positions pos [T] (int64) after their projections (k_qsa_*)"""
        self.launch(dev, st, "k_qsa_prep", (T, nh + nkv + nih + 1), hd,
                    [("p", pos), ("p", qg), ("p", kx), ("p", vx), ("p", ik), ("p", iq), ("p", qn), ("p", kn),
                     ("p", iqn), ("p", inv), ("i", half), ("f", eps), ("i", nh), ("i", nkv), ("i", nih), ("i", di),
                     ("p", qo), ("p", kc), ("p", vc), ("p", rk), ("p", qi)])
        self.launch(dev, st, "k_qsa_bkey", (T,), di, [("p", pos), ("p", rk), ("p", ikn), ("p", inv), ("i", half),
                                                      ("f", eps), ("i", ra), ("p", bk)])
        self.launch(dev, st, "k_qsa_sel", (T,), 1024, [("p", pos), ("p", qi), ("p", bk), ("i", nb), ("i", nih),
                                                       ("i", di), ("i", ra), ("i", nk), ("l", lmax), ("p", sc),
                                                       ("p", sel), ("p", ok)])
        S = min(nk, nb) * ra + ra - 1
        self.launch(dev, st, "k_qsa_attn", (T, nkv, 16), hd, [("p", sel), ("p", ok), ("i", S), ("p", qo),
                                                              ("p", kc), ("p", vc), ("i", nkv), ("i", nh // nkv),
                                                              ("p", part)])
        self.launch(dev, st, "k_qsa_comb", (T, nh), hd, [("p", part), ("p", qg), ("i", nh), ("p", out)])

    def count(self, dev, st, ids, n, counts):
        self.launch(dev, st, "k_count", ((n + 127) // 128,), 128, [("p", ids), ("i", n), ("p", counts)])

    def ts(self, dev, st, buf, i):
        self.launch(dev, st, "k_ts", (1,), 32, [("p", buf), ("i", i)])

    def ma_seq(self, dev, st, seq):
        self.launch(dev, st, "k_ma_seq", (1,), 32, [("p", seq)])

    def ma_post(self, dev, st, box, seq, ids, x, T, nsel, kin, xoff):
        self.launch(dev, st, "k_ma_post", (1,), 256, [("p", box), ("p", seq), ("p", ids), ("p", x), ("i", T),
                                                     ("i", nsel), ("i", kin), ("l", xoff)])

    def ma_wait(self, dev, st, box, seq, dst, ids, slotmap, T, nsel, K, rowoff):
        self.launch(dev, st, "k_ma_wait", (T * nsel,), 256, [("p", box), ("p", seq), ("p", dst), ("p", ids),
                                                            ("p", slotmap), ("i", nsel), ("i", K), ("l", rowoff)])

    # the down input's activation block: 32 (default) or 64 (KINDI_QX_DN=64, item G: integer sums per weight block,
    # the CPU's qx_cpu.c the same through its own copy of the switch)
    DN = int(os.environ.get("KINDI_QX_DN", "32"))

    def quant_dn(self, dev, st, x, n, q, d):
        """the down input h quantized for the down rows (blocks of DN values)"""
        if self.DN == 64:
            nb = n // 64
            self.launch(dev, st, "k_quant_w", ((nb + 7) // 8,), 256, [("p", x), ("l", nb), ("p", q), ("p", d),
                                                                     ("i", 64)])
        else:
            self.quant(dev, st, x, n, q, d)

    # ------------------------------------------------------------ experts on int8 tensor cores (item G)
    def xmm_tc(self, dev, st, ty, cols, rows, ab, nu, mats, part, pstart, ptok, pslot, nsel, in_pair, xq, xs, out, ldo):
        if nu:
            self.launch(dev, st, "k_xmm_tc", ((rows + 63) // 64, nu), 128,
                        [("i", ty), ("i", cols), ("i", rows), ("i", ab), ("p", mats), ("i", part), ("p", pstart),
                         ("p", ptok), ("p", pslot), ("i", nsel), ("i", in_pair), ("p", xq), ("p", xs), ("p", out),
                         ("i", ldo)])

    # the tiled expert GEMM (k_xtile_*, sm_80+; the same bits as k_xmm_tc): KINDI_QXG_XTILE=1 (default) or 0 (k_xmm_tc)
    XTILE = os.environ.get("KINDI_QXG_XTILE", "1") != "0"
    XT_KERNEL = {22: "k_xtile_iq2s", 18: "k_xtile_iq3xxs", 21: "k_xtile_iq3s", 23: "k_xtile_iq4xs", 20: "k_xtile_iq4nl",
                 42: "k_xtile_q20"}
    XT_PER = {22: 256, 18: 256, 21: 256, 23: 256, 20: 32, 42: 64}
    XT_BYTES = {22: 82, 18: 98, 21: 110, 23: 136, 20: 18, 42: 18}
    XT_BM, XT_BN = 128, 64

    def xtile_ok(self, ty, cols, rows, ntiles=0):
        """k_xtile_* takes this matrix (the caller has checked sm_80+)"""
        return (self.XTILE and ty in self.XT_KERNEL and rows % self.XT_BM == 0 and cols % 64 == 0
                and cols % self.XT_PER[ty] == 0 and ntiles <= 65535)

    def xmm_tile(self, dev, st, ty, cols, rows, ab, mats, part, ubase, tiles, ntiles, ptok, pslot, nsel, in_pair, xq,
                 xs, out, ldo):
        """k_xmm_tc's rows with the tiled GEMM; tiles: ntiles int4 (expert + ubase, first pair, end pair, 0) entries
        of at most XT_BN pairs (qwen4exp_gpu.xtiles)"""
        if ntiles:
            self.launch(dev, st, self.XT_KERNEL[ty], (rows // self.XT_BM, ntiles), 256,
                        [("i", cols), ("i", ab), ("p", mats), ("i", part), ("i", ubase), ("p", tiles), ("p", ptok),
                         ("p", pslot), ("i", nsel), ("i", in_pair), ("p", xq), ("p", xs), ("p", out), ("i", ldo)])

    def silu_mul(self, dev, st, g, v, h, n):
        self.launch(dev, st, "k_silu_mul", ((n + 255) // 256,), 256, [("p", g), ("p", v), ("p", h), ("l", n)])

    def quant256(self, dev, st, x, n, q, d):
        nb = n // 256                                   # k_quant256's bits, one warp a block (k_quant_w)
        self.launch(dev, st, "k_quant_w", ((nb + 7) // 8,), 256, [("p", x), ("l", nb), ("p", q), ("p", d), ("i", 256)])

    _shapes_ok = set()

    def check_shape(self, K, F):
        """the expert kernels give each output row a group of 8 (k_gate_up, k_down, the slot kernels, k_fgu) or 16
        lanes (k_fdn) and return the groups past the last row before full-warp shuffles. A warp then has lanes that
        never reach the shuffle, which hangs on sm_70+ (tools/check_warp_masks.py). Every row count must fill whole
        warps: F (the expert width) a multiple of 4 and K (the hidden size; k_down_slots does K / 2 rows) of 8."""
        if (K, F) in self._shapes_ok:
            return
        if F % 4 or K % 8:
            raise ValueError("Kindi qx: expert shape K=%d, F=%d is not supported: the expert kernels need F a multiple "
                             "of 4 and K a multiple of 8 (a partial warp of rows would hang on sm_70+ cards)" % (K, F))
        self._shapes_ok.add((K, F))

    def gate_up(self, dev, st, ty, K, F, nu, mats, pstart, ptok, xq, xd, h):
        self.check_shape(K, F)
        if nu:
            self.launch(dev, st, "k_gate_up", ((F + 15) // 16, nu), 128,
                        [("i", ty), ("i", K), ("i", F), ("p", mats), ("p", pstart), ("p", ptok), ("p", xq), ("p", xd),
                         ("p", h)])

    def down(self, dev, st, ty, K, F, nsel, nu, mats, pstart, ptok, pslot, hq, hd, parts):
        self.check_shape(K, F)
        if nu:
            self.launch(dev, st, "k_down", ((K + 15) // 16, nu), 128,
                        [("i", ty), ("i", K), ("i", F), ("i", nsel), ("p", mats), ("p", pstart), ("p", ptok),
                         ("p", pslot), ("p", hq), ("p", hd), ("p", parts), ("i", self.DN)])

    def combine(self, dev, st, T, nsel, K, w, parts, out):
        n = T * K
        self.launch(dev, st, "k_combine", ((n + 255) // 256,), 256,
                    [("i", T), ("i", nsel), ("i", K), ("p", w), ("p", parts), ("p", out)])

    # KINDI_QXG_FSLOTS=1: the decode slots on k_fgu / k_fdn (a whole 32-value sub-block a thread; same integers and
    # float steps as k_gate_up_slots / k_down_slots, so the same bits); formats outside XG_PER keep the old kernels
    FSLOTS = os.environ.get("KINDI_QXG_FSLOTS", "1") == "1"
    FS_GL = int(os.environ.get("KINDI_QXG_FSGL", "16"))      # k_fgu lanes per row: 8, 16 or 32

    def fslots(self, ty, n, maxblk, tys):
        return self.FSLOTS and ty in tys and n % XG_PER[ty] == 0 and n % 32 == 0 and n // XG_PER[ty] <= maxblk

    def gate_up_slots(self, dev, st, ty, K, F, nsel, idx, slotmap, hot, rowbytes, xq, xd, h, tok_nsel=None,
                      min_slot=0):
        """nsel slots; slot s reads input row s // tok_nsel (default: one token, all slots on row 0)"""
        self.check_shape(K, F)
        if self.fslots(ty, K, 40, (18, 21, 22, 23)):
            nw = 256 // self.FS_GL
            self.launch(dev, st, "k_fgu", ((F + nw - 1) // nw, nsel), 256,
                        [("i", ty), ("i", K), ("i", F), ("p", idx), ("p", slotmap), ("p", hot), ("l", rowbytes),
                         ("p", xq), ("p", xd), ("p", h), ("i", tok_nsel or nsel), ("i", min_slot)])
            return
        self.launch(dev, st, "k_gate_up_slots", ((F + 15) // 16, nsel), 128,
                    [("i", ty), ("i", K), ("i", F), ("p", idx), ("p", slotmap), ("p", hot), ("l", rowbytes),
                     ("p", xq), ("p", xd), ("p", h), ("i", tok_nsel or nsel), ("i", min_slot)])

    def down_slots(self, dev, st, tyg, ty, K, F, nsel, idx, slotmap, hot, rowbytes, hq, hd, parts, min_slot=0):
        self.check_shape(K, F)
        if (self.DN == 64 and K % 16 == 0 and self.fslots(ty, F, 24, (20, 42))
                and self.fslots(tyg, K, 40, (18, 21, 22, 23))):
            rbg = (K // XG_PER[tyg]) * XG_BYTES[tyg]
            self.launch(dev, st, "k_fdn", (K // 16, nsel), 256,
                        [("i", ty), ("i", K), ("i", F), ("p", idx), ("p", slotmap), ("p", hot), ("l", rowbytes),
                         ("l", 2 * F * rbg), ("p", hq), ("p", hd), ("p", parts), ("i", min_slot)])
            return
        self.launch(dev, st, "k_down_slots", ((K // 2 + 15) // 16, nsel), 128,
                    [("i", tyg), ("i", ty), ("i", K), ("i", F), ("p", idx), ("p", slotmap), ("p", hot),
                     ("l", rowbytes), ("p", hq), ("p", hd), ("p", parts), ("i", self.DN), ("i", min_slot)])

    def combine_sel(self, dev, st, nsel, K, w, idx, slotmap, hotp, coldp, out):
        self.launch(dev, st, "k_combine_sel", ((K + 255) // 256,), 256,
                    [("i", nsel), ("i", K), ("p", w), ("p", idx), ("p", slotmap), ("p", hotp), ("p", coldp),
                     ("p", out)])

    # ------------------------------------------------------------ fused one-token steps
    def rms_g(self, dev, st, h, w, out, G, n, eps):
        self.launch(dev, st, "k_rms_g", (G,), 256, [("p", h), ("p", w), ("p", out), ("i", n), ("f", eps)])

    def silu_s(self, dev, st, x, n, s):
        self.launch(dev, st, "k_silu_s", ((n + 255) // 256,), 256, [("p", x), ("i", n), ("f", s)])

    def mix_out(self, dev, st, u, hn, inj, x, wr, G, n, hc):
        self.launch(dev, st, "k_mix_out", ((n + 255) // 256,), 256,
                    [("p", u), ("p", hn), ("p", inj), ("p", x), ("p", wr), ("i", G), ("i", n), ("f", hc)])

    def write(self, dev, st, hin, y, y2, sdot, wr, out, G, n):
        self.launch(dev, st, "k_write", ((G * n + 255) // 256,), 256,
                    [("p", hin), ("p", y), ("p", y2), ("p", sdot), ("p", wr), ("p", out), ("i", G), ("i", n)])

    def dot(self, dev, st, x, v, n, out):
        self.launch(dev, st, "k_dot", (1,), 256, [("p", x), ("p", v), ("i", n), ("p", out)])

    def swiglu(self, dev, st, g, u, out, n):
        self.launch(dev, st, "k_swiglu", ((n + 255) // 256,), 256, [("p", g), ("p", u), ("p", out), ("i", n)])

    def router(self, dev, st, logit, n, k, idx, top):
        self.launch(dev, st, "k_router", (1,), (n + 31) // 32 * 32,
                    [("p", logit), ("i", n), ("i", k), ("p", idx), ("p", top)])

    # ------------------------------------------------------------ the window's T-token steps (k_*T)
    def launch3(self, dev, st, name, grid, block, args):
        self.launch(dev, st, name, grid, block, args)

    def rms_gT(self, dev, st, h, w, out, G, n, eps, T):
        self.launch(dev, st, "k_rms_gT", (G, 1, T), 256, [("p", h), ("p", w), ("p", out), ("i", n), ("f", eps),
                                                         ("l", G * n)])

    def mix_outT(self, dev, st, u, hn, inj, x, wr, G, n, hc, T, ldu, ldhn, ldinj, ldx, ldwr):
        self.launch(dev, st, "k_mix_outT", ((n + 255) // 256, 1, T), 256,
                    [("p", u), ("p", hn), ("p", inj), ("p", x), ("p", wr), ("i", G), ("i", n), ("f", hc),
                     ("l", ldu), ("l", ldhn), ("l", ldinj), ("l", ldx), ("l", ldwr)])

    def writeT(self, dev, st, hin, y, y2, sdot, wr, out, G, n, T, ldh, ldy, ldwr):
        self.launch(dev, st, "k_writeT", ((G * n + 255) // 256, 1, T), 256,
                    [("p", hin), ("p", y), ("p", y2), ("p", sdot), ("p", wr), ("p", out), ("i", G), ("i", n),
                     ("l", ldh), ("l", ldy), ("l", ldwr)])

    def dotT(self, dev, st, x, v, n, out, T, ldx):
        self.launch(dev, st, "k_dotT", (1, 1, T), 256, [("p", x), ("p", v), ("i", n), ("p", out), ("l", ldx)])

    def routerT(self, dev, st, logit, n, k, idx, top, T):
        self.launch(dev, st, "k_routerT", (1, 1, T), (n + 31) // 32 * 32,
                    [("p", logit), ("i", n), ("i", k), ("p", idx), ("p", top)])

    def combine_selT(self, dev, st, nsel, K, w, idx, slotmap, hotp, coldp, out, T):
        self.launch(dev, st, "k_combine_selT", ((K + 255) // 256, 1, T), 256,
                    [("i", nsel), ("i", K), ("p", w), ("p", idx), ("p", slotmap), ("p", hotp), ("p", coldp),
                     ("p", out)])

    def conv1T(self, dev, st, x, state, w, y, C, kw, T, ldx, hist):
        self.launch(dev, st, "k_conv1T", ((C + 255) // 256,), 256,
                    [("p", x), ("p", state), ("p", w), ("p", y), ("i", C), ("i", kw), ("i", T), ("l", ldx),
                     ("p", hist)])

    def gdn1T(self, dev, st, nk, nv, conv, b, a, ssm_a, dtb, S, out, T, ldconv, ldba, ldout, Shist):
        self.launch(dev, st, "k_gdn1T", (nv, 4), 128,
                    [("i", nk), ("p", conv), ("p", b), ("p", a), ("p", ssm_a), ("p", dtb), ("p", S), ("p", out),
                     ("i", T), ("l", ldconv), ("l", ldba), ("l", ldout), ("p", Shist)])

    def gnormT(self, dev, st, x, z, w, o, nh, eps, T, ldx, ldz):
        self.launch(dev, st, "k_gnormT", (nh, 1, T), 128,
                    [("p", x), ("p", z), ("p", w), ("p", o), ("f", eps), ("l", ldx), ("l", ldz)])

    def conv1(self, dev, st, x, state, w, y, C, kw):
        self.launch(dev, st, "k_conv1", ((C + 255) // 256,), 256,
                    [("p", x), ("p", state), ("p", w), ("p", y), ("i", C), ("i", kw)])

    def gdn1(self, dev, st, nk, nv, conv, b, a, ssm_a, dtb, S, out):
        self.launch(dev, st, "k_gdn1", (nv, 4), 128, [("i", nk), ("p", conv), ("p", b), ("p", a), ("p", ssm_a),
                                                      ("p", dtb), ("p", S), ("p", out)])

    def gnorm(self, dev, st, x, z, w, o, nh, eps):
        self.launch(dev, st, "k_gnorm", (nh,), 128, [("p", x), ("p", z), ("p", w), ("p", o), ("f", eps)])

    @staticmethod
    def dense8_plan(rows, cols):
        """(split, rows per block) of k_dense8 for a matrix: few rows with long rows get several warps per row"""
        steps = (cols // 32 + 7) // 8
        split = 8 if rows < 4096 and steps >= 64 else 4 if rows < 4096 and steps >= 16 else 1
        return split, 8 // split

    # KINDI_QXG_D8T2=1 (default): the T-token dense products with two rows a warp (k_dense8T2 / k_dense8mT2): the
    # same bits, the inputs loaded once for both rows
    D8T2 = os.environ.get("KINDI_QXG_D8T2", "1") == "1"
    D8RW = int(os.environ.get("KINDI_QXG_D8RW", "2"))       # rows a warp (2 or 4)
    T2FIX = os.environ.get("KINDI_QXG_T2FIX", "1") == "1"   # a kernel per window size (1-3), own register budget

    # KINDI_QXG_IDENSE=1: the decode's dense products (one token, every window size, the head) take their inputs as
    # 8-bit integers per 32 values (k_qx8) against the K-quant / IQ4_XS weights with dp4a; every path does the same
    # integer and float steps, so a row has the same bits in single steps and windows (default; B79 3060: +2%
    # writing, gate 174/174). 0: the float products
    IDENSE = os.environ.get("KINDI_QXG_IDENSE", "1") == "1"

    def qx8(self, dev, st, X, T, cols):
        """(the buffer, its address): X [T, cols] as int8 [T, cols] then the scales float [T, cols / 32]; (None, 0)
        without IDENSE. The buffer is a PyTorch allocation on the current stream (st): keep it until the launch"""
        if not self.IDENSE:
            return None, 0
        import torch
        n32 = T * cols // 32
        buf = torch.empty(T * cols + 4 * n32, dtype=torch.uint8, device=torch.device("cuda", dev))
        self.launch(dev, st, "k_qx8", ((n32 + 7) // 8,), 256,
                    [("p", X), ("i", n32), ("p", buf.data_ptr()), ("p", buf.data_ptr() + T * cols)])
        return buf, buf.data_ptr()

    def dense8T(self, dev, st, ty, W, rows, cols, rb, X, T, Y):
        """T <= 8 tokens through one matrix, each row with the bits of a one-token k_dense8 launch"""
        split, rpb = self.dense8_plan(rows, cols)
        if self.D8T2 and T <= 5:
            keep, xq = self.qx8(dev, st, X, T, cols)
            self.launch(dev, st, ("k_dense8T2_%d" % T) if (self.T2FIX and T <= 3) else "k_dense8T2", ((rows + self.D8RW * rpb - 1) // (self.D8RW * rpb),), 256,
                        [("i", ty), ("p", W), ("i", rows), ("i", cols), ("l", rb), ("p", X), ("p", Y), ("i", split),
                         ("i", T), ("p", xq)])
            return
        self.launch(dev, st, "k_dense8T", ((rows + rpb - 1) // rpb,), 256,
                    [("i", ty), ("p", W), ("i", rows), ("i", cols), ("l", rb), ("p", X), ("p", Y), ("i", split),
                     ("i", T)])

    def dense8mT(self, dev, st, desc, nm, nblocks, X, T, cols=0):
        """desc made with rpb2() rows per block when D8T2 (the caller plans with Kernels.rpbT); cols: the input's
        (every matrix's) columns, needed with IDENSE"""
        name = "k_dense8mT2" if (self.D8T2 and T <= 5) else "k_dense8mT"
        if name == "k_dense8mT2" and self.T2FIX and T <= 3:
            name = "k_dense8mT2_%d" % T
        args = [("p", desc), ("i", nm), ("p", X), ("i", T)]
        keep = None
        if name != "k_dense8mT":
            if self.IDENSE and not cols:
                raise ValueError("dense8mT: IDENSE needs the input's columns")
            keep, xq = self.qx8(dev, st, X, T, cols)
            args.append(("p", xq))
        self.launch(dev, st, name, (nblocks,), 256, args)

    @classmethod
    def planT(cls, rows, cols, T):
        """(split, rows per block) of the T-token kernels"""
        split, rpb = cls.dense8_plan(rows, cols)
        return split, (cls.D8RW * rpb if (cls.D8T2 and T <= 5) else rpb)

    def dense8m(self, dev, st, desc, nm, nblocks, X, cols=0):
        if self.IDENSE and not cols:
            raise ValueError("dense8m: IDENSE needs the input's columns")
        keep, xq = self.qx8(dev, st, X, 1, cols)
        self.launch(dev, st, "k_dense8m", (nblocks,), 256, [("p", desc), ("i", nm), ("p", X), ("p", xq)])

    def dense8_1(self, dev, st, ty, W, rows, cols, rb, X, Y, split):
        """one token through one matrix with k_dense8 (split warps a row)"""
        keep, xq = self.qx8(dev, st, X, 1, cols)
        rpb = 8 // split
        self.launch(dev, st, "k_dense8", ((rows + rpb - 1) // rpb,), 256,
                    [("i", ty), ("p", W), ("i", rows), ("i", cols), ("l", rb), ("p", X), ("p", Y), ("i", split),
                     ("p", xq)])

    def combine_hot(self, dev, st, nsel, K, w, idx, slotmap, hotp, coldsum, out):
        self.launch(dev, st, "k_combine_hot", ((K + 255) // 256,), 256,
                    [("i", nsel), ("i", K), ("p", w), ("p", idx), ("p", slotmap), ("p", hotp), ("p", coldsum),
                     ("p", out)])

    def pick(self, dev, st, x, n, map_, tok, sel, prob):
        self.launch(dev, st, "k_pick", (1,), 1024,
                    [("p", x), ("i", n), ("p", map_), ("p", tok), ("p", sel), ("p", prob)])

    def deq_row(self, dev, st, ty, W, cols, rb, sel, out):
        self.launch(dev, st, "k_deq_row", ((cols + 255) // 256,), 256,
                    [("i", ty), ("p", W), ("i", cols), ("l", rb), ("p", sel), ("p", out)])

    def deq16(self, dev, st, ty, W, rows, cols, rb, out):
        n = rows * cols
        self.launch(dev, st, "k_deq16", ((n + 255) // 256,), 256,
                    [("i", ty), ("p", W), ("i", rows), ("i", cols), ("l", rb), ("p", out)])

    def _cublas(self):
        """the cuBLAS library PyTorch loaded (its handle is PyTorch's)"""
        lib = getattr(self, "_cublas_lib", None)
        if lib is None:
            path = None
            if os.name != "nt":
                for line in open("/proc/self/maps"):
                    f = line.split()[-1]
                    if "libcublas.so" in os.path.basename(f) and "Lt" not in os.path.basename(f):
                        path = f
                        break
            if path is None:
                import torch
                hits = sorted(glob.glob(os.path.join(os.path.dirname(torch.__file__), "lib",
                                                     "cublas64_*.dll" if os.name == "nt" else "libcublas.so*")))
                hits = [h for h in hits if "Lt" not in os.path.basename(h)]
                if not hits:
                    raise OSError("cuBLAS not found")
                path = hits[-1]
            lib = self._cublas_lib = ctypes.CDLL(path)
        return lib

    def gemm16(self, st, x16, w16, y, T, rows, cols):
        """y [T, rows] fp32 = x16 [T, cols] fp16 . w16 [rows, cols] fp16 ^T, fp32 accumulation on tensor cores
        (cublasGemmEx on PyTorch's handle of the current card, on stream st)"""
        import torch
        cb = self._cublas()
        h = ctypes.c_void_p(torch.cuda.current_blas_handle())
        self._chk(cb.cublasSetStream_v2(h, ctypes.c_void_p(st)), "cublasSetStream")
        one, zero = ctypes.c_float(1.0), ctypes.c_float(0.0)
        r = cb.cublasGemmEx(h, 1, 0, rows, T, cols, ctypes.byref(one), ctypes.c_void_p(w16), 2, cols,
                            ctypes.c_void_p(x16), 2, cols, ctypes.byref(zero), ctypes.c_void_p(y), 0, rows, 68, 99)
        self._chk(r, "cublasGemmEx")

    def deq(self, dev, st, ty, W, rows, cols, rb, out):
        n = rows * cols
        self.launch(dev, st, "k_deq", ((n + 255) // 256,), 256,
                    [("i", ty), ("p", W), ("i", rows), ("i", cols), ("l", rb), ("p", out)])

    # ------------------------------------------------------------ tensor-core dense (1-8 tokens, sm_80+)
    tc_drafting = False        # set by the model when the MTP head is loaded (drafting may run)

    def tc_on(self, dev, T=1):
        """the decode's dense products on tensor cores (cards of compute capability 8.0 and newer)? KINDI_QXG_TC:
        auto (default): on for every decode product when the MTP head is loaded (drafting: a token's numbers must
        not depend on how many drafts were accepted, so single steps and windows use the same kernel), off
        without the head (one token is faster on k_dense8); 1: always; 0: never"""
        mode = os.environ.get("KINDI_QXG_TC", "auto")
        # auto: off since the two-row T-token kernels (D8T2): B30 (3060), the whole model's dense products in a graph:
        # k_dense8 24.0 ms one token, k_dense8T2 26.6 / 27.4 ms at T = 2 / 3, against the tensor-core path's 35.5 /
        # 40.8 (single steps and windows keep each other's bits either way)
        if mode == "0" or (mode == "auto" and (self.D8T2 or not self.tc_drafting)):
            return False
        cache = self.__dict__.setdefault("_tc", {})
        if dev not in cache:
            import torch
            cache[dev] = torch.cuda.get_device_capability(dev)[0] >= 8
        return cache[dev]

    def dense_tc(self, dev, st, ty, W, rows, cols, rb, X, T, Y):
        self.launch(dev, st, "k_dense_tc", ((rows + 15) // 16,), 256,
                    [("i", ty), ("p", W), ("i", rows), ("i", cols), ("l", rb), ("p", X), ("p", Y), ("i", T)])

    def dense_tcm(self, dev, st, desc, nm, nblocks, X, T):
        self.launch(dev, st, "k_dense_tcm", (nblocks,), 256, [("p", desc), ("i", nm), ("p", X), ("i", T)])

    def t1_on(self, dev):
        """one token through the T-token kernels (k_dense8T2_1, k_dense8mT2_1: each row with k_dense8's bits)?
        KINDI_QXG_D8T1: auto (default) = below sm_70, where they beat k_dense8 (M4000, the whole model's dense
        products: 52 against 87 ms); 1: always; 0: never"""
        cache = self.__dict__.setdefault("_t1", {})
        if dev not in cache:
            mode = os.environ.get("KINDI_QXG_D8T1", "auto")
            if mode == "auto":
                import torch
                cache[dev] = torch.cuda.get_device_capability(dev)[0] < 7
            else:
                cache[dev] = mode == "1"
        return cache[dev]

    def dense(self, dev, st, ty, W, rows, cols, rb, X, T, Y):
        mode = os.environ.get("KINDI_QXG_DENSE1", "8")
        if T == 1 and mode != "0" and self.tc_on(dev):
            self.dense_tc(dev, st, ty, W, rows, cols, rb, X, 1, Y)
            return
        if T == 1 and mode != "0" and self.t1_on(dev):
            self.dense8T(dev, st, ty, W, rows, cols, rb, X, 1, Y)
            return
        if T == 1 and mode != "0":
            # one token: k_dense8 (coalesced, 8 values per lane; the default) or k_dense1 (a sub-block per lane);
            # few rows with long rows: several warps share a row
            nsub = cols // 32
            if mode == "8":
                self.dense8_1(dev, st, ty, W, rows, cols, rb, X, Y, self.dense8_plan(rows, cols)[0])
                return
            else:
                split = 8 if rows < 4096 and nsub >= 256 else 4 if rows < 4096 and nsub >= 128 else 1
                name = "k_dense1"
            rpb = 8 // split
            self.launch(dev, st, name, ((rows + rpb - 1) // rpb,), 256,
                        [("i", ty), ("p", W), ("i", rows), ("i", cols), ("l", rb), ("p", X), ("p", Y), ("i", split)])
            return
        split = 8 if rows < 8192 and cols >= 1024 else 1        # few rows: 8 warps share a row
        rpb = 8 // split
        for t0 in range(0, T, 16):
            n = min(16, T - t0)
            self.launch(dev, st, "k_dense", ((rows + rpb - 1) // rpb,), 256,
                        [("i", ty), ("p", W), ("i", rows), ("i", cols), ("l", rb), ("p", X + 4 * t0 * cols), ("i", n),
                         ("p", Y + 4 * t0 * rows), ("i", rows), ("i", split)])

    def gdn_pre(self, dev, st, x, prev, w, qk, v, state, T, C, kw, cqk, nk, b, a, ssm_a, dtb, beta, g, nv):
        """the prompt's Gated DeltaNet inputs from the projections (k_gdn_conv, k_gdn_state, k_gdn_norm)"""
        n = T * C
        self.launch(dev, st, "k_gdn_conv", ((n + 255) // 256,), 256,
                    [("p", x), ("p", prev), ("p", w), ("p", qk), ("p", v), ("p", state), ("i", T), ("i", C),
                     ("i", kw), ("i", cqk)])
        if state:
            self.launch(dev, st, "k_gdn_state", (((kw - 1) * C + 255) // 256,), 256,
                        [("p", x), ("p", prev), ("p", state), ("i", T), ("i", C), ("i", kw)])
        self.launch(dev, st, "k_gdn_norm", (2 * nk, T), 128,
                    [("p", qk), ("i", nk), ("p", b), ("p", a), ("p", ssm_a), ("p", dtb), ("p", beta), ("p", g),
                     ("i", nv)])

    # KINDI_QXG_GDN4=1: the prompt's recurrence on k_gdn4 (4 blocks a head, the window's sum order; the gate decides)
    GDN4 = os.environ.get("KINDI_QXG_GDN4", "0") == "1"

    def gdn(self, dev, st, T, nk, nv, q, k, v, g, beta, S, out):
        self.launch(dev, st, "k_gdn4" if self.GDN4 else "k_gdn", (nv, 4) if self.GDN4 else (nv,), 128,
                    [("i", T), ("i", nk), ("i", nv), ("p", q), ("p", k), ("p", v), ("p", g), ("p", beta), ("p", S),
                     ("p", out)])
