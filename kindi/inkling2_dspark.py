"""Python/ctypes wrapper and JIT builder for the Inkling-2-Mini DSpark CPU drafter (libinkling2_dspark.so)."""
import ctypes
import hashlib
import os
import subprocess

import numpy as np

from .cores import physical_cores
from .gguf import GGUF

DS_NL = 3
DS_D = 2304
DS_R = 256
DS_B = 8

DS_HEAD_ARGMAX_CB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float))


class _DsLayerSpec(ctypes.Structure):
    _fields_ = [
        ("attn_k_ctx", ctypes.c_void_p),
        ("attn_v_ctx", ctypes.c_void_p),
        ("attn_k_ctx_norm", ctypes.c_void_p),
        ("dflare_norm", ctypes.c_void_p),
        ("attn_norm", ctypes.c_void_p),
        ("attn_q", ctypes.c_void_p),
        ("attn_k", ctypes.c_void_p),
        ("attn_v", ctypes.c_void_p),
        ("attn_q_norm", ctypes.c_void_p),
        ("attn_k_norm", ctypes.c_void_p),
        ("attn_output", ctypes.c_void_p),
        ("ffn_norm", ctypes.c_void_p),
        ("ffn_gate", ctypes.c_void_p),
        ("ffn_up", ctypes.c_void_p),
        ("ffn_down", ctypes.c_void_p),
    ]


class _DsModelSpec(ctypes.Structure):
    _fields_ = [
        ("fuse", ctypes.c_void_p),
        ("dspark_norm", ctypes.c_void_p),
        ("mask_embd", ctypes.c_void_p),
        ("output_norm", ctypes.c_void_p),
        ("hidden_out_w", ctypes.c_void_p),
        ("hidden_out_b", ctypes.c_void_p),
        ("joint_proj_w", ctypes.c_void_p),
        ("joint_proj_b", ctypes.c_void_p),
        ("markov_w1", ctypes.c_void_p),
        ("vocab_pad", ctypes.c_int),
        ("max_ctx", ctypes.c_int),
        ("layers", _DsLayerSpec * DS_NL),
    ]


_LIB = None


def _compile_dspark():
    global _LIB
    if _LIB is not None:
        return _LIB
    os.environ.setdefault("GOMP_SPINCOUNT", "0")
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, "inkling2_dspark.c")
    with open(src, "rb") as f:
        code = f.read()
    digest = hashlib.sha256(code).hexdigest()[:12]
    cdir = os.path.join(os.path.expanduser("~"), ".cache", "kindi")
    os.makedirs(cdir, exist_ok=True)
    so = os.path.join(cdir, "libinkling2_dspark_%s.so" % digest)
    if not os.path.exists(so):
        tmp = so + ".tmp.%d" % os.getpid()
        cmd = ["gcc", "-O3", "-march=native", "-fopenmp", "-shared", "-fPIC", src, "-o", tmp, "-lm"]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if p.returncode != 0:
            raise RuntimeError("dspark gcc failed:\n%s" % p.stderr.decode("utf-8", "replace"))
        os.replace(tmp, so)
    lib = ctypes.CDLL(so)
    lib.ds_init.argtypes = [ctypes.POINTER(_DsModelSpec), ctypes.c_int]
    lib.ds_init.restype = ctypes.c_void_p
    lib.ds_free.argtypes = [ctypes.c_void_p]
    lib.ds_free.restype = None
    lib.ds_reset.argtypes = [ctypes.c_void_p]
    lib.ds_reset.restype = None
    lib.ds_inject.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    lib.ds_inject.restype = ctypes.c_int
    lib.ds_forward_block.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
    ]
    lib.ds_forward_block.restype = None
    lib.ds_draft_tokens.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, DS_HEAD_ARGMAX_CB, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
    ]
    lib.ds_draft_tokens.restype = ctypes.c_int
    _LIB = lib
    return lib


class DSparkDrafter(object):
    def __init__(self, gguf_path, max_ctx=4096, threads=None):
        self.lib = _compile_dspark()
        self.gguf = GGUF(gguf_path)
        self.tmap = {}
        self._refs = []
        base = self.gguf.data_start
        import mmap
        self._fh = open(gguf_path, "rb")
        mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        self._mm = mm
        for name, ne, ty, off in self.gguf.tensors:
            self.tmap[name] = (ne, ty, base + off, mm)

        kv = self.gguf.kv
        if kv.get("general.architecture") != "dflash":
            raise ValueError("Expected dflash GGUF, got %r" % (kv.get("general.architecture"),))
        self.target_layers = [int(x) for x in kv.get("dflash.target_layers", [5, 23, 41, 48])]
        self.block_size = int(kv.get("dflash.block_size", DS_B))
        self.vocab_unpadded = int(kv.get("dflash.vocab_size_unpadded", 200058))
        self.vocab_pad = int(self.tmap["markov_w1.weight"][0][1])
        self.max_ctx = int(max_ctx)

        def ptr_f32(name):
            ne, ty, off, m = self.tmap[name]
            assert ty == 0, "%s expected F32(0), got %d" % (name, ty)
            cnt = 1
            for x in ne:
                cnt *= int(x)
            arr = np.frombuffer(m, dtype=np.float32, count=cnt, offset=off)
            self._refs.append(arr)
            return int(arr.ctypes.data)

        def ptr_u16(name):
            ne, ty, off, m = self.tmap[name]
            assert ty == 30, "%s expected BF16(30), got %d" % (name, ty)
            cnt = 1
            for x in ne:
                cnt *= int(x)
            arr = np.frombuffer(m, dtype=np.uint16, count=cnt, offset=off)
            self._refs.append(arr)
            return int(arr.ctypes.data)

        def ptr_q8(name):
            ne, ty, off, m = self.tmap[name]
            assert ty == 8, "%s expected Q8_0(8), got %d" % (name, ty)
            cnt = 1
            for x in ne:
                cnt *= int(x)
            nbytes = (cnt // 32) * 34
            arr = np.frombuffer(m, dtype=np.uint8, count=nbytes, offset=off)
            self._refs.append(arr)
            return int(arr.ctypes.data)

        spec = _DsModelSpec()
        spec.fuse = ptr_f32("dspark_fuse.weight")
        spec.dspark_norm = ptr_f32("dspark_norm.weight")
        spec.mask_embd = ptr_f32("mask_embd.weight")
        spec.output_norm = ptr_f32("output_norm.weight")
        spec.hidden_out_w = ptr_u16("dspark_hidden_out.weight")
        spec.hidden_out_b = ptr_f32("dspark_hidden_out.bias")
        spec.joint_proj_w = ptr_u16("joint_proj.weight")
        spec.joint_proj_b = ptr_f32("joint_proj.bias")
        spec.markov_w1 = ptr_u16("markov_w1.weight")
        spec.vocab_pad = self.vocab_pad
        spec.max_ctx = self.max_ctx

        for l in range(DS_NL):
            p = "blk.%d." % l
            ls = spec.layers[l]
            ls.attn_k_ctx = ptr_q8(p + "attn_k_ctx.weight")
            ls.attn_v_ctx = ptr_q8(p + "attn_v_ctx.weight")
            ls.attn_k_ctx_norm = ptr_f32(p + "attn_k_ctx_norm.weight")
            ls.dflare_norm = ptr_f32(p + "dflare_norm.weight")
            ls.attn_norm = ptr_f32(p + "attn_norm.weight")
            ls.attn_q = ptr_q8(p + "attn_q.weight")
            ls.attn_k = ptr_q8(p + "attn_k.weight")
            ls.attn_v = ptr_q8(p + "attn_v.weight")
            ls.attn_q_norm = ptr_f32(p + "attn_q_norm.weight")
            ls.attn_k_norm = ptr_f32(p + "attn_k_norm.weight")
            ls.attn_output = ptr_q8(p + "attn_output.weight")
            ls.ffn_norm = ptr_f32(p + "ffn_norm.weight")
            ls.ffn_gate = ptr_q8(p + "ffn_gate.weight")
            ls.ffn_up = ptr_q8(p + "ffn_up.weight")
            ls.ffn_down = ptr_q8(p + "ffn_down.weight")

        n_thr = int(threads or physical_cores())
        self._ctx = self.lib.ds_init(ctypes.byref(spec), n_thr)
        if not self._ctx:
            raise RuntimeError("ds_init failed")
        self._out_toks = (ctypes.c_int * DS_B)()

    def reset(self):
        if self._ctx:
            self.lib.ds_reset(self._ctx)

    def inject(self, pos0, T, feats_ptr):
        if T <= 0:
            return
        rc = self.lib.ds_inject(self._ctx, int(pos0), int(T), int(feats_ptr))
        if rc != 0:
            raise RuntimeError("ds_inject failed rc=%d (pos0=%d T=%d max_ctx=%d)" % (rc, pos0, T, self.max_ctx))

    def draft(self, anchor_tok, pos0, n_draft, anchor_embd_ptr, head_cb, head_user=0):
        rc = self.lib.ds_draft_tokens(
            self._ctx, int(anchor_tok), int(pos0), int(n_draft),
            int(anchor_embd_ptr), head_cb, ctypes.c_void_p(head_user or 0),
            self._out_toks,
        )
        if rc < 0:
            raise RuntimeError("ds_draft_tokens failed rc=%d" % rc)
        return [int(self._out_toks[i]) for i in range(rc)]

    def close(self):
        if getattr(self, "_ctx", None):
            self.lib.ds_free(self._ctx)
            self._ctx = None
        self.gguf = None
        self._refs = []
        for a in ("_mm", "_fh"):
            o = getattr(self, a, None)
            if o is not None:
                try:
                    o.close()
                except Exception:
                    pass
                setattr(self, a, None)

    def __del__(self):
        self.close()
