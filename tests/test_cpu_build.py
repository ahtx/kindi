"""The CPU engine must call the configured compiler with an argument vector, never a shell string."""
import ctypes
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from kindi import cpu

SRC = Path(__file__).resolve().parents[1] / "kindi" / "kcpu.c"


def _cc():
    """A compiler for the preprocessor checks, preferring KINDI_CC."""
    for candidate in (os.environ.get("KINDI_CC"), "gcc", "clang"):
        if candidate and shutil.which(candidate):
            return shutil.which(candidate)
    pytest.skip("neither gcc nor clang is installed")


def test_build_passes_paths_as_arguments(monkeypatch, tmp_path, capsys):
    seen = {}
    monkeypatch.setattr(cpu, "_native_has_fma", lambda: True)

    def fake_run(cmd):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(cpu.subprocess, "run", fake_run)
    src, so = str(tmp_path / "kcpu.c"), str(tmp_path / "libkcpu.so")
    cpu._build(src, so)
    assert seen["cmd"] == cpu._build_cmd(src, so)
    assert "building CPU library with %s" % " ".join(seen["cmd"]) in capsys.readouterr().err


def test_linux_build_uses_gcc_with_the_documented_flags(tmp_path):
    src, so = str(tmp_path / "kcpu.c"), str(tmp_path / "libkcpu.so")
    assert cpu._build_cmd(src, so, win=False) == [
        "gcc", "-O3", "-march=native", "-fopenmp", "-fno-math-errno",
        "-shared", "-fPIC", "-o", so, src, "-lm"]


@pytest.mark.parametrize("isa", cpu.ISA_ORDER)
def test_each_isa_build_uses_explicit_flags(isa, tmp_path):
    src, so = str(tmp_path / "kcpu.c"), str(tmp_path / "libkcpu.so")
    cmd = cpu._build_cmd(src, so, win=False, isa=isa)
    assert "-march=native" not in cmd
    assert all(flag in cmd for flag in cpu.ISA_BUILDS[isa])


def test_each_isa_has_its_own_cache_path(tmp_path):
    paths = {isa: cpu._so_path(str(tmp_path), isa) for isa in cpu.ISA_ORDER}
    assert len(set(paths.values())) == len(paths)
    assert cpu._so_path(str(tmp_path)) not in paths.values()


def test_windows_build_uses_clang_libomp_and_no_posix_flags(monkeypatch, tmp_path):
    llvm = tmp_path / "llvm"
    (llvm / "lib").mkdir(parents=True)
    monkeypatch.setenv("KINDI_CC", "clang")
    monkeypatch.setenv("KINDI_LLVM", str(llvm))
    src, dll = str(tmp_path / "kcpu.c"), str(tmp_path / "libkcpu.dll")
    cmd = cpu._build_cmd(src, dll, win=True, native_fma=False)
    assert cmd[0] == "clang" and cmd[1] == "-O3"
    assert "-fopenmp" in cmd and "-ffp-contract=off" in cmd
    assert "-fPIC" not in cmd and "-lm" not in cmd
    assert cmd[cmd.index("-o") + 1] == dll
    assert src in cmd
    assert cmd[-7:-4] == ["-L", str(llvm / "lib"), "-llibomp"]
    assert cmd[-4:] == ["-Xlinker", "/nodefaultlib:libcmt",
                        "-Xlinker", "/defaultlib:msvcrt"]


def test_windows_build_keeps_static_crt_when_fma_is_available(monkeypatch, tmp_path):
    llvm = tmp_path / "llvm"
    (llvm / "lib").mkdir(parents=True)
    monkeypatch.setenv("KINDI_CC", "clang")
    monkeypatch.setenv("KINDI_LLVM", str(llvm))
    src, dll = str(tmp_path / "kcpu.c"), str(tmp_path / "libkcpu.dll")
    cmd = cpu._build_cmd(src, dll, win=True, native_fma=True)
    assert "-Xlinker" not in cmd
    assert cmd[-3:] == ["-L", str(llvm / "lib"), "-llibomp"]


def test_cpu_cache_path_is_keyed_by_toolchain(monkeypatch, tmp_path):
    monkeypatch.setenv("KINDI_CC", "gcc")
    first = cpu._so_path(str(tmp_path))
    monkeypatch.setenv("KINDI_CC", "clang")
    second = cpu._so_path(str(tmp_path))
    assert first != second
    assert os.path.basename(first).startswith("libkcpu-")
    assert os.path.dirname(first) == str(tmp_path)


def test_build_failure_is_a_clear_exit(monkeypatch):
    monkeypatch.setattr(cpu, "_native_has_fma", lambda: True)
    monkeypatch.setattr(cpu.subprocess, "run",
                        lambda cmd: subprocess.CompletedProcess(cmd, 1))
    with pytest.raises(SystemExit, match="could not build the CPU engine"):
        cpu._build("kcpu.c", "libkcpu.so")


def test_source_has_the_isa_guard():
    assert "requires AVX" in SRC.read_text(encoding="utf-8")


def test_avx_only_build_preprocesses():
    cc = _cc()
    proc = subprocess.run([cc, "-fopenmp", "-mavx", "-mno-avx2", "-mno-fma", "-mno-f16c", "-E", str(SRC)],
                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")


def test_avx_only_library_builds_and_loads(monkeypatch, tmp_path):
    """Compile and load the fallback, not just preprocess it.

    The model-free test suite otherwise cannot catch a link/runtime break in the AVX-only
    branch.  The resulting library is placed at the keyed path the production loader uses, so
    this also checks the loader and the Python ctypes ABI without a pack.
    """
    if os.name == "nt":
        cc = shutil.which(os.environ.get("KINDI_CC") or "clang")
        if not cc:
            pytest.skip("clang is not installed; set KINDI_CC to build the AVX-only fallback")
        monkeypatch.setenv("KINDI_CC", cc)
    else:
        if not shutil.which("gcc"):
            pytest.skip("gcc is not installed; it is needed to build the AVX-only fallback")

    monkeypatch.setenv("KINDI_CPU_CACHE", str(tmp_path))
    monkeypatch.setenv("KINDI_CPU_ISA", "avx-only")
    so = Path(cpu._so_path(str(tmp_path), "avx-only"))
    cpu._build(str(SRC), str(so), isa="avx-only")

    library = cpu._lib()
    assert library.kc_isa_name() == b"avx-only"
    assert library.kc_sizeof_model() == ctypes.sizeof(cpu.Model)
    assert library.kc_sizeof_layer() == ctypes.sizeof(cpu.Layer)
    library.kc_init()


def test_avx2_f16c_without_fma_preprocesses():
    cc = _cc()
    proc = subprocess.run([cc, "-fopenmp", "-mavx2", "-mno-fma", "-mf16c", "-E", str(SRC)],
                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")


def test_avx2_f16c_no_fma_library_builds_and_loads(monkeypatch, tmp_path):
    """The AVX2/F16C/no-FMA branch must compile and load, not just preprocess."""
    if os.name == "nt":
        cc = shutil.which(os.environ.get("KINDI_CC") or "clang")
        if not cc:
            pytest.skip("clang is not installed; set KINDI_CC to build the no-FMA AVX2 fallback")
        monkeypatch.setenv("KINDI_CC", cc)
    else:
        if not shutil.which("gcc"):
            pytest.skip("gcc is not installed; it is needed to build the no-FMA AVX2 fallback")

    monkeypatch.setenv("KINDI_CPU_CACHE", str(tmp_path))
    monkeypatch.setenv("KINDI_CPU_ISA", "avx2-f16c-no-fma")
    so = Path(cpu._so_path(str(tmp_path), "avx2-f16c-no-fma"))
    cpu._build(str(SRC), str(so), isa="avx2-f16c-no-fma")

    library = cpu._lib()
    assert library.kc_isa_name() == b"avx2-f16c-no-fma"
    assert library.kc_sizeof_model() == ctypes.sizeof(cpu.Model)
    assert library.kc_sizeof_layer() == ctypes.sizeof(cpu.Layer)
    library.kc_init()


def test_full_isa_library_builds_and_loads(monkeypatch, tmp_path):
    """The native fast path compiles, loads and reports the tier the selector chose."""
    if cpu.cpu_tier() not in ("avx2-fma-f16c", "unknown"):
        pytest.skip("this host does not expose AVX2/FMA/F16C")
    if os.name == "nt":
        cc = shutil.which(os.environ.get("KINDI_CC") or "clang")
        if not cc:
            pytest.skip("clang is not installed; set KINDI_CC to build the native path")
        monkeypatch.setenv("KINDI_CC", cc)
    elif not shutil.which("gcc"):
        pytest.skip("gcc is not installed; it is needed to build the native path")

    monkeypatch.setenv("KINDI_CPU_CACHE", str(tmp_path))
    monkeypatch.setenv("KINDI_CPU_ISA", "avx2-fma-f16c")
    so = Path(cpu._so_path(str(tmp_path), "avx2-fma-f16c"))
    cpu._build(str(SRC), str(so), isa="avx2-fma-f16c")

    library = cpu._lib()
    assert library.kc_isa_name() == b"avx2-fma-f16c"
    assert library.kc_isa() == 4
    assert library.kc_sizeof_model() == ctypes.sizeof(cpu.Model)
    assert library.kc_sizeof_layer() == ctypes.sizeof(cpu.Layer)


def test_non_avx_build_is_rejected_with_the_required_isa():
    cc = _cc()
    proc = subprocess.run([cc, "-fopenmp", "-mno-avx", "-mno-avx2", "-mno-fma", "-mno-f16c", "-E", str(SRC)],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.returncode != 0
    assert "requires AVX" in proc.stderr.decode("utf-8", "replace")


def test_supported_isa_source_preprocesses():
    cc = _cc()
    proc = subprocess.run([cc, "-fopenmp", "-mavx2", "-mfma", "-mf16c", "-E", str(SRC)],
                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")


@pytest.mark.parametrize("flags", (["-mavx2", "-mno-fma", "-mno-f16c"],
                                   ["-mavx2", "-mfma", "-mno-f16c"]))
def test_other_avx2_variants_preprocess(flags):
    cc = _cc()
    proc = subprocess.run([cc, "-fopenmp", *flags, "-E", str(SRC)],
                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
