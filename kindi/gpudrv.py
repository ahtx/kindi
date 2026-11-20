"""The GPU driver behind Kindi's launch plans and graphs: libcuda on NVIDIA (nvcuda.dll on Windows), libamdhip64 on AMD (ROCm).

Kindi's launch plans, events and graphs call the CUDA driver API by name (cuLaunchKernel, cuCtxSetCurrent, ...).
On NVIDIA, driver() is the platform's driver library itself, unchanged. On ROCm there is no libcuda: driver() is an
object with the same
names that forwards each call to its HIP twin. The argument layouts are the same (hipModuleLaunchKernel takes
cuLaunchKernel's arguments, graph node parameters have the same layout), except for contexts: a HIP "context" here
is the device index, read with hipGetDevice and set with hipSetDevice.

Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only. See LICENSE and COMMERCIAL-LICENSE.md.
"""
import ctypes
import os

try:
    import cupy as cp
    HIP = bool(getattr(cp.cuda.runtime, "is_hip", False))
except ImportError:  # the CPU engine
    cp, HIP = None, False

P = ctypes.c_void_p
_hip = [None]


def hip_lib():
    """libamdhip64 with the argument types Kindi uses"""
    if _hip[0] is None:
        h = ctypes.CDLL(os.environ.get("KINDI_HIP_LIB", "libamdhip64.so"))
        h.hipModuleLaunchKernel.argtypes = [P] + [ctypes.c_uint] * 7 + [P] * 3
        h.hipModuleLaunchKernel.restype = ctypes.c_int
        h.hipSetDevice.argtypes = [ctypes.c_int]
        h.hipGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
        h.hipEventCreateWithFlags.argtypes = [ctypes.POINTER(P), ctypes.c_uint]
        h.hipEventRecord.argtypes = [P, P]
        h.hipEventSynchronize.argtypes = [P]
        h.hipGraphCreate.argtypes = [ctypes.POINTER(P), ctypes.c_uint]
        h.hipGraphAddKernelNode.argtypes = [ctypes.POINTER(P), P, ctypes.POINTER(P), ctypes.c_size_t, P]
        h.hipGraphInstantiateWithFlags.argtypes = [ctypes.POINTER(P), P, ctypes.c_ulonglong]
        h.hipGraphExecKernelNodeSetParams.argtypes = [P, P, P]
        h.hipGraphLaunch.argtypes = [P, P]
        h.hipGraphDestroy.argtypes = [P]
        h.hipGraphExecDestroy.argtypes = [P]
        h.hipStreamBeginCapture.argtypes = [P, ctypes.c_int]
        h.hipStreamEndCapture.argtypes = [P, ctypes.POINTER(P)]
        h.hipGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        h.hipSetDeviceFlags.argtypes = [ctypes.c_uint]
        _hip[0] = h
    return _hip[0]


def _out(ref):
    """the object behind a ctypes.byref(...) argument"""
    return getattr(ref, "_obj", ref)


class _HipAsCuda(object):
    """the CUDA driver names Kindi calls, each forwarded to HIP"""

    def __init__(self):
        h = self.hip = hip_lib()
        self.cuLaunchKernel = h.hipModuleLaunchKernel
        self.cuEventRecord = h.hipEventRecord
        self.cuEventSynchronize = h.hipEventSynchronize
        self.cuGraphCreate = h.hipGraphCreate
        self.cuGraphInstantiateWithFlags = h.hipGraphInstantiateWithFlags
        self.cuGraphLaunch = h.hipGraphLaunch
        self.cuGraphDestroy = h.hipGraphDestroy
        self.cuGraphExecDestroy = h.hipGraphExecDestroy
        self.cuStreamBeginCapture_v2 = h.hipStreamBeginCapture       # same mode values (2 = relaxed)
        self.cuStreamEndCapture = h.hipStreamEndCapture

    def __getattr__(self, name):           # the HIP names themselves (kindi code written for ROCm uses them)
        return getattr(self.hip, name)

    def cuEventCreate(self, ev, flags):
        return self.hip.hipEventCreateWithFlags(ev, flags)

    def cuCtxGetCurrent(self, ref):
        d = ctypes.c_int()
        rc = self.hip.hipGetDevice(ctypes.byref(d))
        _out(ref).value = d.value
        return rc

    def cuCtxSetCurrent(self, h):
        v = h.value if isinstance(h, ctypes.c_void_p) else h
        return self.hip.hipSetDevice(int(v or 0))

    def cuGraphAddKernelNode(self, node, graph, deps, ndeps, params):
        return self.hip.hipGraphAddKernelNode(node, graph, deps, ndeps, ctypes.cast(params, P))

    def cuGraphExecKernelNodeSetParams(self, ex, node, params):
        return self.hip.hipGraphExecKernelNodeSetParams(ex, node, ctypes.cast(params, P))


_drv = [None]


def driver_name():
    """the NVIDIA driver library this platform exposes (ROCm uses hip_lib instead)"""
    return "nvcuda.dll" if os.name == "nt" else "libcuda.so.1"


def driver():
    """libcuda (NVIDIA; nvcuda.dll on Windows) or its HIP stand-in (ROCm). On NVIDIA the caller sets the argument
    types as before."""
    if _drv[0] is None:
        _drv[0] = _HipAsCuda() if HIP else ctypes.CDLL(driver_name())
    return _drv[0]
