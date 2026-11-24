"""Turing (sm_75) int8 tensor cores for the plan engine's tensor-core path (KINDI_TC=1; PR #75, atimelessrage).

OFF BY DEFAULT until it is measured on an RTX 20 card against the current Turing path and llama.cpp (same card,
model and context) with matching greedy output. KINDI_TC_TURING=1 turns it on for a TU10x card; KINDI_TC_TURING=force
runs the Turing build on any card from sm_75 up (an Ampere or later card runs compute_75 code), so `kindi verify` and
tools/test_tc_turing.py check its arithmetic on a card at hand.

The kernels are tc.cu and tca.cu themselves, built with KINDI_TURING defined: every m16n8k32 int8 product becomes four
m8n8k16 (exact integer sums, so the products are bit-identical to the Ampere build), the fp16 attention product
m16n8k16 becomes two m16n8k8, and cp.async (sm_80) becomes plain loads before the same barriers. So the Turing path
has every feature of the Ampere one (the 4B's half units, KINDI_TC_HP, the b2 layout) with no second copy of the
kernels. The PTX is compute_75 as NVRTC writes it (no .target patch); the driver builds it for the card.

Only TU102 / TU104 / TU106 have tensor cores (RTX 20, Titan RTX, Quadro RTX, Tesla T4, CMP 40HX). TU116 / TU117 (GTX 16,
Quadro T400..T2000, MX) are sm_75 too but have none; the device name tells them apart.
"""
import os
import re

import cupy as cp
from cupy.cuda import compiler, function

PTX_MIN = (6, 5)          # mma.m8n8k16 needs PTX ISA 6.5; CUDA 10.2's NVRTC writes it
_TENSOR = ("RTX 20", "TITAN RTX", "QUADRO RTX", "TESLA T4", " T4 ", "CMP 40HX")


def tensor_cores(name):
    """True for an sm_75 device name of a TU10x part (one with int8 tensor cores)"""
    if isinstance(name, bytes):
        name = name.decode("ascii", "replace")
    return any(s in " " + name.upper().replace("-", " ") + " " for s in _TENSOR)


def forced():
    """KINDI_TC_TURING=force: the Turing build on any card from sm_75 up (arithmetic checks on a newer card)"""
    return os.environ.get("KINDI_TC_TURING", "0") == "force"


def supported(card=None):
    """KINDI_TC_TURING=1 and a TU10x card, or KINDI_TC_TURING=force and any NVIDIA card from sm_75 up"""
    v = os.environ.get("KINDI_TC_TURING", "0")
    if v not in ("1", "force") or cp.cuda.runtime.is_hip:
        return False
    p = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id if card is None else card)
    if v == "force":
        return (p["major"], p["minor"]) >= (7, 5)
    return (p["major"], p["minor"]) == (7, 5) and tensor_cores(p["name"])


def source(src):
    """the module source with the Turing build switched on"""
    return "#define KINDI_TURING 1\n" + src


def load_module(src):
    """compile for compute_75 and load the PTX as written; only an NVRTC older than PTX ISA 6.5 has its .version
    raised"""
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
