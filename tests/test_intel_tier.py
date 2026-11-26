"""CPU-only tests for the Intel cold-tier gate probe (follow-up A3)."""
import ctypes
import json

from kindi import intel_tier


class _FakeFn(object):
    def __init__(self, fn):
        self.fn = fn
        self.restype = None
        self.argtypes = None

    def __call__(self, *args):
        return self.fn(*args)


def _int(value):
    return value.value if isinstance(value, ctypes._SimpleCData) else int(value)


def _addr(value):
    if value is None:
        return 0
    if isinstance(value, ctypes.c_void_p):
        return value.value or 0
    obj = getattr(value, "_obj", None)
    if obj is not None:
        return ctypes.addressof(obj)
    return ctypes.addressof(value)


def _put_bytes(data, size, pointer, size_ret):
    size = _int(size)
    if pointer is not None and size:
        ctypes.memmove(_addr(pointer), data, min(len(data), size))
    if size_ret is not None:
        size_ret._obj.value = len(data)
    return 0


class FakeOpenCL(object):
    """Platform/device queries plus a byte-faithful buffer store; no GPU."""

    def __init__(self, device_count=1, max_alloc_mib=4095,
                 name="Intel(R) Arc(TM) A770 Graphics"):
        self.device_count = device_count
        self.max_alloc_mib = max_alloc_mib
        self.name = name
        self.buffers = {}
        self.next_buffer = 0x1000
        self.clGetPlatformIDs = _FakeFn(self._platform_ids)
        self.clGetPlatformInfo = _FakeFn(self._platform_info)
        self.clGetDeviceIDs = _FakeFn(self._device_ids)
        self.clGetDeviceInfo = _FakeFn(self._device_info)
        self.clCreateContext = _FakeFn(lambda *args: 0x300)
        self.clReleaseContext = _FakeFn(lambda *args: 0)
        self.clCreateCommandQueue = _FakeFn(lambda *args: 0x400)
        self.clReleaseCommandQueue = _FakeFn(lambda *args: 0)
        self.clCreateBuffer = _FakeFn(self._buffer_create)
        self.clReleaseMemObject = _FakeFn(self._buffer_release)
        self.clEnqueueWriteBuffer = _FakeFn(self._write)
        self.clEnqueueReadBuffer = _FakeFn(self._read)
        self.clFinish = _FakeFn(lambda *args: 0)

    def _platform_ids(self, entries, handles, total):
        total._obj.value = 1
        if handles is not None:
            handles[0] = 0x100
        return 0

    def _platform_info(self, platform, param, size, value, size_ret):
        return _put_bytes(b"Intel(R) OpenCL Graphics (discrete)\x00", size, value, size_ret)

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
                 0x102C: b"Intel(R) Corporation\x00",
                 0x102D: b"32.0.101.9175\x00"}
        if param in texts:
            return _put_bytes(texts[param], size, value, size_ret)
        if param == 0x1010:
            value_ = ctypes.c_uint64(self.max_alloc_mib * 1024 * 1024)
            data = ctypes.string_at(ctypes.addressof(value_), ctypes.sizeof(value_))
            return _put_bytes(data, size, value, size_ret)
        return -30

    def _buffer_create(self, context, flags, size, pointer, error):
        size = _int(size)
        self.buffers[self.next_buffer] = bytearray(size)
        error._obj.value = 0
        handle = self.next_buffer
        self.next_buffer += 1
        return handle

    def _buffer_release(self, handle):
        self.buffers.pop(_int(handle), None)
        return 0

    def _write(self, queue, handle, blocking, offset, size, pointer, *rest):
        handle, offset, size = _int(handle), _int(offset), _int(size)
        data = ctypes.string_at(_addr(pointer), size)
        self.buffers[handle][offset:offset + size] = data
        return 0

    def _read(self, queue, handle, blocking, offset, size, pointer, *rest):
        handle, offset, size = _int(handle), _int(offset), _int(size)
        ctypes.memmove(_addr(pointer), bytes(self.buffers[handle][offset:offset + size]), size)
        return 0


def _run(fake, **kwargs):
    kwargs.setdefault("mib", 1)
    kwargs.setdefault("repeats", 1)
    return intel_tier.run_probe(opener=lambda _name: fake, **kwargs)


def test_tier_pass_reports_copy_and_gate():
    report = _run(FakeOpenCL())
    assert report["result"] == "pass"
    assert report["probe"] == "intel-tier"
    assert report["support_claim"] is False
    assert report["copy"]["mismatches"] == 0
    assert report["copy"]["mibs"] and report["copy"]["mibs"] > 0
    assert report["disk"]["mibs"] and report["disk"]["mibs"] > 0
    assert set(report["gate"]) >= {"threshold", "ratio", "go"}
    assert isinstance(report["gate"]["go"], bool)


def test_tier_refuses_chunk_above_alloc_cap():
    report = intel_tier.run_probe(mib=8, repeats=1,
                                    opener=lambda _name: FakeOpenCL(max_alloc_mib=4))
    assert report["result"] == "fail"
    assert "per-allocation cap" in report["error"]
    assert report["gate"]["chunk_fits_alloc_cap"] is False


def test_tier_gate_logic():
    assert intel_tier._gate(2000.0, 500.0, True)["go"] is True
    assert intel_tier._gate(500.0, 500.0, True)["go"] is False
    assert intel_tier._gate(2000.0, 500.0, False)["go"] is False
    assert intel_tier._gate(None, 500.0, True)["go"] is False


def test_tier_rejects_bad_arguments():
    assert _run(FakeOpenCL(), mib=0)["error"].startswith("--mib")
    assert _run(FakeOpenCL(), mib=2048)["error"].startswith("--mib")
    assert _run(FakeOpenCL(), repeats=0)["error"].startswith("--repeats")
    assert _run(FakeOpenCL(), device_name="  ")["error"].startswith("--device-name")
    assert _run(FakeOpenCL(device_count=0))["result"] == "fail"


def test_tier_cli_prints_json_and_returns_failure(monkeypatch, capsys, tmp_path):
    from kindi import __main__ as cli

    report = _run(FakeOpenCL())
    monkeypatch.setattr(intel_tier, "run_probe", lambda *args, **kwargs: report)
    conf = tmp_path / "kindi.conf"
    conf.write_text("[kindi]\n", encoding="utf-8")
    monkeypatch.setenv("KINDI_CONFIG", str(conf))

    assert cli.main(["intel-tier", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["probe"] == "intel-tier"

    failed = dict(report, result="fail", error="synthetic failure")
    monkeypatch.setattr(intel_tier, "run_probe", lambda *args, **kwargs: failed)
    assert cli.main(["intel-tier"]) == 1
    assert "synthetic failure" in capsys.readouterr().out
