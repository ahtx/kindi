"""Turing (sm_75) int8 tensor-core products for Kindi (KINDI_TC=1).

This is the Turing sibling of kindi.tc: it provides the same external kernel names with the same signatures, the same
"tc" packed weight layout and the same numerics, so spec.py, chunk.py and the plan engine need no special cases.  The
kernels live in kindi/tc_turing.cu and use mma.sync.aligned.m8n8k16 (Turing's int8 IMMA) instead of Ampere's
m16n8k32.

Compilation: Turing needs no PTX header patching.  The image's NVRTC (CUDA 10.2) targets compute_75 natively and emits
PTX ISA 6.5, which is the first version that knows mma.m8n8k16, so the PTX is loaded as produced and the driver
JIT-compiles it for the card (the cubin is cached in the driver's JIT cache, CUDA_CACHE_PATH).  Turing has no
cp.async; the kernels stage shared memory with ordinary loads.
"""
import os
import re

import cupy as cp
from cupy.cuda import compiler, function

HERE = os.path.dirname(os.path.abspath(__file__))
PTX_MIN = (6, 5)          # mma.m8n8k16 needs PTX ISA 6.5; CUDA 10.2's NVRTC already emits it

# Same external kernel names as kindi.tc (same signatures and geometry).
NAMES = ("k_tc_pack", "k_tc_q8", "k_tc_mv2", "k_tc_mv4", "k_tc_mv8", "k_tc_sum", "k_tc_mm4", "k_tc_mm8", "k_tc_mm16",
         "k_tc_mv3_2", "k_tc_mv3_4", "k_tc_mv3_8", "k_tc_mm2")


def supported(card=None):
    """Turing (compute capability 7.5) has the int8 IMMA m8n8k16 that tc_turing.cu uses.  Volta (7.0) and older do
    not, and Ampere and later take the m16n8k32 path in kindi.tc."""
    p = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id if card is None else card)
    return p["major"] == 7 and p["minor"] == 5


def load_module(src):
    """compile for compute_75 and load the PTX as produced (the driver JIT builds it for the card); only an NVRTC that
    emits an older PTX ISA than m8n8k16 needs has its .version line raised"""
    prog = compiler._NVRTCProgram(src, "tc_turing.cu")
    ptx, _ = prog.compile(("-arch=compute_75",))
    if isinstance(ptx, bytes):
        ptx = ptx.decode()
    m = re.search(r"\.version (\d+)\.(\d+)", ptx)
    if m and (int(m.group(1)), int(m.group(2))) < PTX_MIN:
        ptx = re.sub(r"\.version \d+\.\d+", ".version %d.%d" % PTX_MIN, ptx, count=1)
    mod = function.Module()
    mod.load(ptx.encode())
    return mod


class TCTuringKernels(object):
    def __init__(self, extra=""):
        src = open(os.path.join(HERE, "tc_turing.cu"), encoding="utf-8").read() + extra
        self.mod = load_module(src)
        for k in NAMES:
            setattr(self, k, self.mod.get_function(k))
