"""CPU-only tests for the Intel capability probe (REQ-0020 M2 precursor)."""
import ctypes
import json

from kindi import intel_caps


class _FakeFn(object):
    def __init__(self, fn):
        self.fn = fn
        self.restype = None
        self.argtypes = None

    def __call__(self, *args):
        return self.fn(*args)


def _addr(value):
    if value is None:
        return 0
    if isinstance(value, ctypes.c_void_p):
        return value.value or 0
    obj = getattr(value, "_obj", None)
    if obj is not None:
        return ctypes.addressof(obj)
    return ctypes.addressof(value)


def _put(value, size, pointer, size_ret):
    data = ctypes.string_at(ctypes.addressof(value), ctypes.sizeof(value))
    if pointer is not None and size:
        ctypes.memmove(_addr(pointer), data, min(len(data), int(size)))
    if size_ret is not None:
        size_ret._obj.value = ctypes.sizeof(value)
    return 0


def _put_bytes(data, size, pointer, size_ret):
    if pointer is not None and size:
        ctypes.memmove(_addr(pointer), data, min(len(data), int(size)))
    if size_ret is not None:
        size_ret._obj.value = len(data)
    return 0


class FakeOpenCL(object):
    """Enough of the OpenCL loader for capability queries: no kernel entry points."""

    def __init__(self, device_count=1, vendor="Intel(R) Corporation",
                 name="Intel(R) Arc(TM) A770 Graphics",
                 opencl_c="OpenCL C 1.2 ",
                 extensions="cl_khr_fp64 cl_khr_subgroups",
                 max_alloc_mib=4095):
        self.device_count = device_count
        self.vendor = vendor
        self.name = name
        self.opencl_c = opencl_c
        self.extensions = extensions
        self.max_alloc_mib = max_alloc_mib
        self.clGetPlatformIDs = _FakeFn(self._platform_ids)
        self.clGetPlatformInfo = _FakeFn(self._platform_info)
        self.clGetDeviceIDs = _FakeFn(self._device_ids)
        self.clGetDeviceInfo = _FakeFn(self._device_info)

    def _platform_ids(self, entries, handles, total):
        total._obj.value = 1
        if handles is not None:
            handles[0] = 0x100
        return 0

    def _platform_info(self, platform, param, size, value, size_ret):
        texts = {0x0902: b"Intel(R) OpenCL Graphics (discrete)\x00",
                 0x0903: b"Intel(R) Corporation\x00",
                 0x0901: b"OpenCL 3.0 \x00"}
        return _put_bytes(texts.get(param, b"unknown\x00"), size, value, size_ret)

    def _device_ids(self, platform, device_type, entries, handles, total):
        total._obj.value = self.device_count
        if not self.device_count:
            return -1
        if handles is not None:
            for index in range(self.device_count):
                handles[index] = 0x200 + index
        return 0

    def _device_info(self, device, param, size, value, size_ret):
        texts = {0x102B: (self.name + "\x00").encode("utf-8"),
                 0x102C: (self.vendor + "\x00").encode("utf-8"),
                 0x102D: b"32.0.101.9175\x00",
                 0x102F: b"OpenCL 3.0 NEO \x00",
                 0x103D: (self.opencl_c + "\x00").encode("utf-8"),
                 0x1030: (self.extensions + "\x00").encode("utf-8")}
        if param in texts:
            return _put_bytes(texts[param], size, value, size_ret)
        scalars = {0x1002: ctypes.c_uint32(512), 0x100C: ctypes.c_uint32(2100),
                   0x101F: ctypes.c_uint64(16 * 1024 ** 3),
                   0x1010: ctypes.c_uint64(self.max_alloc_mib * 1024 * 1024),
                   0x1023: ctypes.c_uint64(64 * 1024),
                   0x1004: ctypes.c_size_t(1024)}
        if param in scalars:
            return _put(scalars[param], size, value, size_ret)
        return -30


def _run(fake, **kwargs):
    return intel_caps.run_probe(opener=lambda _name: fake, **kwargs)


def test_caps_pass_reports_premises_and_chunking():
    report = _run(FakeOpenCL())
    assert report["result"] == "pass"
    assert report["probe"] == "intel-caps"
    assert report["read_only"] is True
    assert report["support_claim"] is False
    device = report["device"]
    assert device["name"] == "Intel(R) Arc(TM) A770 Graphics"
    assert device["max_alloc_mib"] == 4095
    premises = device["premises"]
    assert premises["opencl_c_1_2"] is True
    assert premises["fp16_arithmetic"] is False
    assert premises["subgroups"] is True
    assert premises["dpas_via_opencl"] is False
    assert premises["single_27b_codes_fit"] is False
    assert premises["chunking_required"] is True
    assert report["warnings"] == []


def test_caps_fit_when_alloc_cap_clears_codes():
    report = _run(FakeOpenCL(max_alloc_mib=8192))
    assert report["result"] == "pass"
    assert report["device"]["premises"]["single_27b_codes_fit"] is True
    assert report["device"]["premises"]["chunking_required"] is False


def test_caps_warns_on_non_arc_device():
    report = _run(FakeOpenCL(name="Intel(R) UHD Graphics 630"),
                  device_name="UHD")
    assert report["result"] == "pass"
    assert any("not an Arc" in warning for warning in report["warnings"])


def test_caps_requires_an_intel_gpu():
    assert _run(FakeOpenCL(device_count=0))["error"].startswith("no Intel")
    report = _run(FakeOpenCL(vendor="NVIDIA Corporation"))
    assert report["result"] == "fail"
    assert "no Intel OpenCL GPU device" in report["error"]


def test_caps_rejects_bad_arguments():
    assert _run(FakeOpenCL(), device=-1)["error"].startswith("--device")
    assert _run(FakeOpenCL(), device_name="  ")["error"].startswith("--device-name")
    report = _run(FakeOpenCL(), device_name="No Such Card")
    assert report["result"] == "fail"
    assert "no Intel GPU matches" in report["error"]
    report = intel_caps.run_probe(loader="missing-cl",
                                  opener=lambda _name: (_ for _ in ()).throw(
                                      OSError("no loader")))
    assert report["result"] == "fail"
    assert "could not load missing-cl" in report["error"]


def test_caps_cli_prints_json_and_returns_failure(monkeypatch, capsys, tmp_path):
    from kindi import __main__ as cli

    report = _run(FakeOpenCL())
    monkeypatch.setattr(intel_caps, "run_probe", lambda *args, **kwargs: report)
    conf = tmp_path / "kindi.conf"
    conf.write_text("[kindi]\n", encoding="utf-8")
    monkeypatch.setenv("KINDI_CONFIG", str(conf))

    assert cli.main(["intel-caps", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["probe"] == "intel-caps"

    failed = dict(report, result="fail", error="synthetic failure")
    monkeypatch.setattr(intel_caps, "run_probe", lambda *args, **kwargs: failed)
    assert cli.main(["intel-caps"]) == 1
    assert "synthetic failure" in capsys.readouterr().out
