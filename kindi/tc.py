"""Tensor-core (int8 mma.sync) products for sm_80+ cards, compiled at start like the other kernels (Pascal sm_61:
the same products with dp4a, tcp.cu; see spec.SpecKernels).

The image's NVRTC (CUDA 10.2) only targets compute_75 and has no mma.sync m16n8k32, but it passes inline PTX through
unchecked: tc.cu is compiled to PTX for compute_75, the PTX header is raised to ISA 7.1 / sm_8x, and the driver's
JIT (any driver new enough for the card) compiles that PTX. The cubin lands in the driver's JIT cache
(CUDA_CACHE_PATH), so later starts do not recompile."""
import os
import re

import cupy as cp
from cupy.cuda import compiler, function

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ("k_tc_pack", "k_tc_q8", "k_tc_mv2", "k_tc_mv4", "k_tc_mv8", "k_tc_sum", "k_tc_mm4", "k_tc_mm8", "k_tc_mm16",
         "k_tc_mv3_2", "k_tc_mv3_4", "k_tc_mv3_8", "k_tc_mm2")


# Pascal sm_61 / sm_62 (dp4a, no tensor cores): the same kernels on dp4a (tcp.cu), same names and arguments
NAMES_PASCAL = ("k_tc_pack", "k_tc_q8", "k_tc_mv2", "k_tc_mv4", "k_tc_mv8", "k_tc_sum", "k_tc_mv3_2", "k_tc_mv3_4",
                "k_tc_mv3_8", "k_tp_mm")


def pascal(card=None):
    """the dp4a kernels: sm_61 / sm_62, or any sm_61+ CUDA card with KINDI_TC_DP4A=1 (a test of tcp.cu on a newer
    card: the products are bit-identical to the tensor-core ones)"""
    if cp.cuda.runtime.is_hip:
        return False
    p = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id if card is None else card)
    cc = p["major"] * 10 + p["minor"]
    return cc in (60, 61, 62) or (cc >= 61 and os.environ.get("KINDI_TC_DP4A", "0") == "1")


def lane(card=None):
    """sm_60 (Tesla P100, no dp4a): the tc plans keep the 5-trit lane layout and run lane-table products (tcl.cu,
    fp32); KINDI_TC_LANE=1 forces them on other CUDA cards (a test), 0 turns them off (tcp.cu, dp4a emulated)"""
    if cp.cuda.runtime.is_hip:
        return False
    p = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id if card is None else card)
    v = os.environ.get("KINDI_TC_LANE", "")
    return v == "1" or (v != "0" and (p["major"], p["minor"]) == (6, 0))


def supported(card=None):
    p = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id if card is None else card)
    return p["major"] >= 8 or p["major"] == 6     # sm_60 (P100): tcp.cu on kdp4a's integer fallback


def ptx_target(arch):
    """the PTX target for this card: PTX ISA 7.1 knows sm_80 and sm_86 only (ptxas: "PTX .version 7.1 does not
    support .target sm_89"), so a later card (sm_87, Ada sm_89, Hopper, Blackwell) gets sm_80 code, which its
    driver builds for the card"""
    return arch if arch in ("sm_80", "sm_86") else "sm_80"


def _patched_module(src, arch, opts=()):
    prog = compiler._NVRTCProgram(src, "tc.cu")
    ptx, _ = prog.compile(("-arch=compute_75",) + tuple(opts))
    if isinstance(ptx, bytes):
        ptx = ptx.decode()
    ptx = re.sub(r"\.version \d+\.\d+", ".version 7.1", ptx, count=1)
    ptx = re.sub(r"\.target sm_\d+", ".target " + ptx_target(arch), ptx, count=1)
    m = function.Module()
    m.load(ptx.encode())
    return m


class TCKernels(object):
    def __init__(self, extra=""):
        p = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        arch = "sm_%d%d" % (p["major"], p["minor"])
        src = open(os.path.join(HERE, "tc.cu"), encoding="utf-8").read() + extra
        self.mod = _patched_module(src, arch)
        for k in NAMES:
            setattr(self, k, self.mod.get_function(k))
