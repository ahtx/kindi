# SPDX-License-Identifier: AGPL-3.0-only
"""CPU-only checks for the Turing tensor-core backend's selection and loader.

The numerical kernels need a CUDA device (tests/test_tc_turing_reference.py). These tests exercise the
decisions ``kindi.tc_turing`` takes before a kernel is launched, with a stand-in ``cupy`` so they run in CI:
compute capability 7.5 with a tensor-core Turing device name is the only supported one, the module compiles for
compute_75, and it exports the same
kernel names as the Ampere backend it replaces.
"""
import contextlib
import importlib
import os
import re
import sys
import types


class _Compile(object):
    def __init__(self, ptx):
        self.ptx = ptx
        self.opts = []

    def compile(self, opts):
        self.opts.append(opts)
        return self.ptx, None


class _Module(object):
    def __init__(self):
        self.loaded = None

    def load(self, ptx):
        self.loaded = ptx


@contextlib.contextmanager
def _backend(major, minor, name="NVIDIA GeForce RTX 2080 Ti", ptx=".version 6.0\n"):
    """Import kindi.tc_turing against a fake cupy whose device reports (major, minor, name); restore on exit."""
    names = ("cupy", "cupy.cuda", "cupy.cuda.runtime", "cupy.cuda.compiler", "cupy.cuda.function",
             "kindi.tc_turing")
    saved = {name: sys.modules.get(name) for name in names}
    before = set(sys.modules)
    pkg = sys.modules.get("kindi")
    old_attr = getattr(pkg, "tc_turing", None) if pkg is not None else None
    cp = types.ModuleType("cupy")
    cuda = types.ModuleType("cupy.cuda")
    compiler = types.ModuleType("cupy.cuda.compiler")
    function = types.ModuleType("cupy.cuda.function")
    runtime = types.ModuleType("cupy.cuda.runtime")
    runtime.getDeviceProperties = lambda card: {"major": major, "minor": minor, "name": name}
    compiler._NVRTCProgram = lambda src, name: _Compile(ptx)
    function.Module = _Module
    cp.cuda = cuda
    cuda.runtime, cuda.compiler, cuda.function = runtime, compiler, function
    sys.modules.update({"cupy": cp, "cupy.cuda": cuda, "cupy.cuda.runtime": runtime,
                        "cupy.cuda.compiler": compiler, "cupy.cuda.function": function})
    sys.modules.pop("kindi.tc_turing", None)
    try:
        yield importlib.import_module("kindi.tc_turing")
    finally:
        for name, old in saved.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
        for name in list(sys.modules):                  # kindi.spec imported under the fake cupy, if any
            if (name == "kindi" or name.startswith("kindi.")) and name not in before:
                sys.modules.pop(name, None)
        if pkg is not None:
            if old_attr is None:
                pkg.__dict__.pop("tc_turing", None)
            else:
                pkg.tc_turing = old_attr


def test_only_sm_75_is_supported():
    with _backend(7, 5) as tc:
        assert tc.supported(0)
    with _backend(7, 0) as tc:
        assert not tc.supported(0)                            # Volta
    with _backend(8, 0) as tc:
        assert not tc.supported(0)                            # Ampere takes kindi.tc
    with _backend(8, 9) as tc:
        assert not tc.supported(0)


def test_tu116_tu117_sm_75_devices_are_rejected():
    """The name, not just sm_75, must select the Turing tensor-core backend.

    TU116/TU117 report the same compute capability as TU102/TU104/TU106 but have no tensor cores; an mma
    instruction traps on them.  The names below cover the GTX 16 series, the Quadro T400/T600/T1000/T2000,
    the MX4xx/MX5xx parts and CMP 30HX; every one must stay on the plain engine.
    """
    yes = ("NVIDIA GeForce RTX 2060", "NVIDIA GeForce RTX 2080 Ti", "NVIDIA TITAN RTX",
           "Quadro RTX 4000", "NVIDIA Tesla T4", "NVIDIA T4", "NVIDIA CMP 40HX")
    no = ("NVIDIA GeForce GTX 1660 Ti", "NVIDIA GeForce GTX 1650", "NVIDIA GeForce GTX 1630",
          "Quadro T400", "Quadro T600", "Quadro T1000", "Quadro T2000", "NVIDIA GeForce MX450",
          "NVIDIA GeForce MX550", "NVIDIA CMP 30HX")
    for name in yes:
        with _backend(7, 5, name=name) as tc:
            assert tc.supported(0), name
    for name in no:
        with _backend(7, 5, name=name) as tc:
            assert not tc.supported(0), name


def test_module_is_compiled_for_compute_75():
    with _backend(7, 5) as tc:
        prog = []
        real = tc.compiler._NVRTCProgram
        tc.compiler._NVRTCProgram = lambda src, name: prog.append(real(src, name)) or prog[-1]
        mod = tc.load_module("__global__ void k(void){}")
        assert prog[0].opts == [("-arch=compute_75",)]
        assert mod.loaded.startswith(b".version 6.5\n")       # m8n8k16 needs PTX ISA 6.5


def test_every_advertised_kernel_is_in_the_source():
    src = open(os.path.join(os.path.dirname(__file__), "..", "kindi", "tc_turing.cu"), encoding="utf-8").read()
    assert "mma.sync.aligned.m8n8k16" in src
    for name in TURING_NAMES:
        assert re.search(r"\b%s\s*\(" % name, src), name
    assert len(TURING_NAMES) == 13


TURING_NAMES = ("k_tc_pack", "k_tc_q8", "k_tc_mv2", "k_tc_mv4", "k_tc_mv8", "k_tc_sum", "k_tc_mm4",
                "k_tc_mm8", "k_tc_mm16", "k_tc_mv3_2", "k_tc_mv3_4", "k_tc_mv3_8", "k_tc_mm2")
