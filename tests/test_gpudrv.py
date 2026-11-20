"""The NVIDIA driver library follows the platform: nvcuda.dll on Windows, libcuda.so.1 on Linux."""
import ctypes
import types

from kindi import gpudrv


def test_driver_name_windows(monkeypatch):
    monkeypatch.setattr(gpudrv, "os", types.SimpleNamespace(name="nt"))
    assert gpudrv.driver_name() == "nvcuda.dll"


def test_driver_name_linux(monkeypatch):
    monkeypatch.setattr(gpudrv, "os", types.SimpleNamespace(name="posix"))
    assert gpudrv.driver_name() == "libcuda.so.1"


def test_driver_loads_the_platform_library_once(monkeypatch):
    monkeypatch.setattr(gpudrv, "os", types.SimpleNamespace(name="nt"))
    monkeypatch.setattr(gpudrv, "HIP", False)
    monkeypatch.setattr(gpudrv, "_drv", [None])
    seen = []
    fake = object()

    def fake_cdll(name):
        seen.append(name)
        return fake

    monkeypatch.setattr(ctypes, "CDLL", fake_cdll)
    assert gpudrv.driver() is fake
    assert gpudrv.driver() is fake
    assert seen == ["nvcuda.dll"]
