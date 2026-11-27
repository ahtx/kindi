"""Turing (sm_75) int8 tensor cores for the plan engine's tensor-core path (KINDI_TC=1; PR #75, atimelessrage).

OFF BY DEFAULT until it is measured on an RTX 20 card against the current Turing path and llama.cpp (same card,
model and context) with matching greedy output. KINDI_TC_TURING=1 turns it on for a TU10x card; KINDI_TC_TURING=force
runs the Turing build on any card from sm_75 up (an Ampere or later card runs compute_75 code), so `kindi verify` and
tools/test_tc_turing.py check its arithmetic on a card at hand.

The kernels are tc.cu and tca.cu themselves, built with KINDI_TURING defined: every m16n8k32 int8 product becomes four
m8n8k16 (exact integer sums, so the products are bit-identical to the Ampere build), the fp16 attention product
m16n8k16 becomes two m16n8k8, and cp.async (sm_80) becomes plain loads before the same barriers. So the Turing path
has every feature of the Ampere one (the 4B's half units, KINDI_TC_HP, the b2 layout) with no second copy of the
kernels. The PTX is compute_75 with tc.py's ISA 7.1 header and .target sm_75; the driver builds it for the card.

Measured 11-24 on an RTX 2070 SUPER (130 W cap), ctx 8448, write tokens/s at positions 100 / 2000 / 8000 (kindi bench;
llama.cpp PrismML c96baf4 llama-bench tg32 at the same depths, -fa 1):
  2B: default path 179.1 at 2000; this path 333.7; llama.cpp 345.9 / 336.9 / 314.9. verify: 1 of 4 prompts differs
      (35/40, the 8-bit near-tie of B4); with KINDI_TC_HP=1 0/40 (an HP bug on this build, open).
  4B: KeyError 'output.weight' in tp.build_plans on a fresh 4B pack (open; the same build passed on an RTX 3080 with
      an older 4B pack).
  9B: default 76.3 / 75.0 / 72.0; this path 117.6 / 115.1 / 109.7, verify 4/4; llama.cpp 130.8 / 127.9 / 120.3.
So it stays off: faster than the default Turing path, not faster than llama.cpp, and not exact on the 2B and 4B.

Only TU102 / TU104 / TU106 have tensor cores (RTX 20, Titan RTX, Quadro RTX, Tesla T4, CMP 40HX). TU116 / TU117 (GTX 16,
Quadro T400..T2000, MX) are sm_75 too but have none; the device name tells them apart.
"""
import os

import cupy as cp

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
    """compile for compute_75 with tc.py's loader: the PTX header says ISA 7.1 (m8n8k16 needs 6.5; a newer NVRTC
    writes a version the card's driver may not know) and .target sm_75; the driver builds it for the card"""
    from . import tc
    return tc._patched_module(src, "sm_75")
