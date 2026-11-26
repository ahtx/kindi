"""CPU-only tests for the OpenCL kernel smoke (REQ-0020 M2 kernel path)."""
import ctypes
import json
from array import array

from kindi import intel_cl

CL_MEM_COPY_HOST_PTR = 1 << 5


class _FakeFn(object):
    def __init__(self, fn):
        self.fn = fn
        self.restype = None
        self.argtypes = None

    def __call__(self, *args):
        return self.fn(*args)


def _handle(value):
    return value.value if isinstance(value, ctypes.c_void_p) else int(value)


def _number(value):
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
    def __init__(self, device_count=1, build_result=0, vendor="Intel(R) Corporation"):
        self.device_count = device_count
        self.build_result = build_result
        self.vendor = vendor
        self.buffers = {}
        self.args = {}
        self.kernels = {}
        self.source = None
        self.released_buffers = []
        self.released_kernels = []
        self.program_released = False
        self.context_released = False
        self.queue_released = False
        self.corrupt_vector_add = False
        self.next_buffer = 0x1000
        self.clGetPlatformIDs = _FakeFn(self._platform_ids)
        self.clGetPlatformInfo = _FakeFn(self._platform_info)
        self.clGetDeviceIDs = _FakeFn(self._device_ids)
        self.clGetDeviceInfo = _FakeFn(self._device_info)
        self.clCreateContext = _FakeFn(self._context_create)
        self.clReleaseContext = _FakeFn(self._context_release)
        self.clCreateCommandQueue = _FakeFn(self._queue_create)
        self.clReleaseCommandQueue = _FakeFn(self._queue_release)
        self.clCreateProgramWithSource = _FakeFn(self._program_create)
        self.clBuildProgram = _FakeFn(self._build)
        self.clGetProgramBuildInfo = _FakeFn(self._build_info)
        self.clReleaseProgram = _FakeFn(self._program_release)
        self.clCreateKernel = _FakeFn(self._kernel_create)
        self.clReleaseKernel = _FakeFn(self._kernel_release)
        self.clCreateBuffer = _FakeFn(self._buffer_create)
        self.clReleaseMemObject = _FakeFn(self._buffer_release)
        self.clSetKernelArg = _FakeFn(self._set_arg)
        self.clEnqueueNDRangeKernel = _FakeFn(self._enqueue)
        self.clEnqueueReadBuffer = _FakeFn(self._read)
        self.clFinish = _FakeFn(self._finish)

    def _platform_ids(self, entries, handles, total):
        total._obj.value = 1
        if handles is not None:
            handles[0] = 0x100
        return 0

    def _platform_info(self, platform, param, size, value, size_ret):
        texts = {0x0902: b"Intel(R) OpenCL Graphics (discrete)\0",
                 0x0903: b"Intel(R) Corporation\0",
                 0x0901: b"OpenCL 3.0 \0"}
        return _put_bytes(texts.get(param, b"unknown\0"), size, value, size_ret)

    def _device_ids(self, platform, device_type, entries, handles, total):
        total._obj.value = self.device_count
        if not self.device_count:
            return -1
        if handles is not None:
            for index in range(self.device_count):
                handles[index] = 0x200 + index
        return 0

    def _device_info(self, device, param, size, value, size_ret):
        texts = {0x102B: b"Intel(R) Arc(TM) A770 Graphics\0",
                 0x102C: (self.vendor + "\0").encode("utf-8"),
                 0x102D: b"32.0.101.9175\0",
                 0x102F: b"OpenCL 3.0 NEO \0",
                 0x103D: b"OpenCL C 1.2 \0"}
        if param in texts:
            return _put_bytes(texts[param], size, value, size_ret)
        scalars = {0x1002: ctypes.c_uint32(512), 0x100C: ctypes.c_uint32(2100),
                   0x101F: ctypes.c_uint64(16 * 1024 ** 3),
                   0x1004: ctypes.c_size_t(1024)}
        if param in scalars:
            return _put(scalars[param], size, value, size_ret)
        return -30

    def _context_create(self, properties, count, devices, notify, user, error):
        error._obj.value = 0
        return 0x300

    def _context_release(self, context):
        self.context_released = True
        return 0

    def _queue_create(self, context, device, properties, error):
        error._obj.value = 0
        return 0x400

    def _queue_release(self, queue):
        self.queue_released = True
        return 0

    def _program_create(self, context, count, strings, lengths, error):
        self.source = strings[0]
        error._obj.value = 0
        return 0x500

    def _build(self, program, count, devices, options, notify, user):
        return self.build_result

    def _build_info(self, program, device, param, size, value, size_ret):
        log = b"error: fake compile failure\0" if self.build_result else b"fake build log\0"
        return _put_bytes(log, size, value, size_ret)

    def _program_release(self, program):
        self.program_released = True
        return 0

    def _kernel_create(self, program, name, error):
        handle = 0x600 + len(self.kernels)
        self.kernels[handle] = name.decode("utf-8") if isinstance(name, bytes) else str(name)
        self.args[handle] = {}
        error._obj.value = 0
        return handle

    def _kernel_release(self, kernel):
        self.released_kernels.append(_handle(kernel))
        return 0

    def _buffer_create(self, context, flags, size, host_ptr, error):
        count = _number(size) // 4
        if _number(flags) & CL_MEM_COPY_HOST_PTR and host_ptr is not None:
            data = array("f")
            data.frombytes(ctypes.string_at(_addr(host_ptr), _number(size)))
        else:
            data = array("f", [0.0]) * count
        handle = self.next_buffer
        self.next_buffer += 1
        self.buffers[handle] = data
        error._obj.value = 0
        return handle

    def _buffer_release(self, buffer):
        self.released_buffers.append(_handle(buffer))
        return 0

    def _set_arg(self, kernel, index, size, arg_value):
        address = _addr(arg_value)
        if _number(size) == ctypes.sizeof(ctypes.c_void_p):
            self.args[_handle(kernel)][index] = ctypes.cast(
                address, ctypes.POINTER(ctypes.c_void_p)).contents.value
        else:
            self.args[_handle(kernel)][index] = ctypes.cast(
                address, ctypes.POINTER(ctypes.c_uint32)).contents.value
        return 0

    def _enqueue(self, queue, kernel, work_dim, offset, global_size, local_size, wait_count,
                 waits, event):
        name = self.kernels[_handle(kernel)]
        args = self.args[_handle(kernel)]
        if name == "vecadd":
            self._vecadd(args)
        elif name == "gemm":
            self._gemm(args)
        else:
            raise AssertionError("unknown kernel %s" % name)
        return 0

    def _vecadd(self, args):
        count = args[3]
        a = self.buffers[args[0]]
        b = self.buffers[args[1]]
        out = self.buffers[args[2]]
        for index in range(count):
            out[index] = a[index] + b[index]
        if self.corrupt_vector_add and count:
            out[0] = out[0] + 1.0

    def _gemm(self, args):
        dim = args[3]
        a = self.buffers[args[0]]
        b = self.buffers[args[1]]
        out = self.buffers[args[2]]
        for row in range(dim):
            for column in range(dim):
                total = 0.0
                for k in range(dim):
                    total += a[row * dim + k] * b[k * dim + column]
                out[row * dim + column] = total

    def _read(self, queue, buffer, blocking, offset, size, pointer, wait_count, waits, event):
        data = self.buffers[_handle(buffer)]
        ctypes.memmove(_addr(pointer), data.tobytes(), int(size))
        return 0

    def _finish(self, queue):
        return 0


def _run(fake, **kwargs):
    kwargs.setdefault("n", 4096)
    kwargs.setdefault("dim", 16)
    kwargs.setdefault("repeats", 2)
    return intel_cl.run_smoke(loader="fake-cl", opener=lambda _name: fake, **kwargs)


def test_kernel_smoke_runs_both_kernels_and_cleans_up():
    fake = FakeOpenCL()
    report = _run(fake)
    assert report["result"] == "pass"
    assert report["build"]["ok"] is True
    assert report["build"]["log"] == "fake build log"
    assert report["platform"] == "Intel(R) OpenCL Graphics (discrete)"
    assert report["device"]["name"] == "Intel(R) Arc(TM) A770 Graphics"
    assert report["device"]["compute_units"] == 512
    assert report["device"]["global_mem_mib"] == 16384
    assert report["vector_add"]["mismatches"] == 0
    assert report["gemm"]["mismatches"] == 0
    assert report["gemm"]["gflops"] is not None
    assert report["returncodes"]["clBuildProgram"] == "0x00000000"
    assert fake.args[0x600][3] == 4096
    assert fake.args[0x601][3] == 16
    assert len(fake.released_buffers) == 6
    assert len(fake.released_kernels) == 2
    assert fake.program_released is True
    assert fake.context_released is True
    assert fake.queue_released is True
    assert "no SYCL" in intel_cl.render(report)[0]
    assert "result: pass" in intel_cl.render(report)[-1]


def test_kernel_smoke_fails_on_a_corrupt_vector_add():
    fake = FakeOpenCL()
    fake.corrupt_vector_add = True
    report = _run(fake)
    assert report["result"] == "fail"
    assert report["vector_add"]["mismatches"] == 1
    assert "vector-add mismatch" in report["error"]
    assert report["gemm"]["mismatches"] is None
    assert len(fake.released_buffers) == 3
    assert fake.program_released is True


def test_kernel_smoke_reports_a_build_failure_and_its_log():
    fake = FakeOpenCL(build_result=-11)
    report = _run(fake)
    assert report["result"] == "fail"
    assert report["build"]["ok"] is False
    assert "clBuildProgram returned" in report["error"]
    assert "fake compile failure" in report["error"]
    assert report["vector_add"]["mismatches"] is None
    assert fake.context_released is True
    assert fake.queue_released is True
    assert fake.program_released is True
    assert len(fake.released_buffers) == 0


def test_kernel_smoke_requires_an_intel_gpu():
    assert "no Intel OpenCL GPU device" in _run(FakeOpenCL(device_count=0))["error"]
    report = _run(FakeOpenCL(vendor="NVIDIA Corporation"))
    assert report["result"] == "fail"
    assert "no Intel OpenCL GPU device" in report["error"]
    assert report["device_count"] == 0


def test_kernel_smoke_reports_out_of_range_device():
    report = _run(FakeOpenCL(), device=2)
    assert report["result"] == "fail"
    assert "outside the 1 visible Intel GPU" in report["error"]


def test_kernel_smoke_rejects_bad_arguments_before_loading():
    assert _run(FakeOpenCL(), n=0)["error"].startswith("--n")
    assert _run(FakeOpenCL(), dim=7)["error"].startswith("--dim")
    assert _run(FakeOpenCL(), repeats=0)["error"].startswith("--repeats")
    assert _run(FakeOpenCL(), device=-1)["error"].startswith("--device")
    report = intel_cl.run_smoke(loader="missing-cl", opener=lambda _name: (_ for _ in ()).throw(
        OSError("no loader")))
    assert report["result"] == "fail"
    assert "could not load missing-cl" in report["error"]
    assert report["loader"]["available"] is False


def test_intel_cl_cli_prints_json_and_returns_failure(monkeypatch, capsys, tmp_path):
    from kindi import __main__ as cli

    report = _run(FakeOpenCL())
    monkeypatch.setattr(intel_cl, "run_smoke", lambda *args, **kwargs: report)
    conf = tmp_path / "kindi.conf"
    conf.write_text("[kindi]\n", encoding="utf-8")
    monkeypatch.setenv("KINDI_CONFIG", str(conf))

    assert cli.main(["intel-cl", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["result"] == "pass"

    failed = dict(report, result="fail", error="synthetic failure")
    monkeypatch.setattr(intel_cl, "run_smoke", lambda *args, **kwargs: failed)
    assert cli.main(["intel-cl"]) == 1
    assert "synthetic failure" in capsys.readouterr().out
