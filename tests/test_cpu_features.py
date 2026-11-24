"""The CPU feature report and the preflight before building kcpu.c."""
import pytest

from kindi import cpu

INFO = """processor	: 0
model name	: Example i5
flags		: sse avx avx2 fma f16c
"""


def _report(monkeypatch, text=INFO, physical=6, logical=12):
    monkeypatch.setattr(cpu, "_read", lambda path: text if path == "/proc/cpuinfo" else None)
    monkeypatch.setattr(cpu, "_physical_cores", lambda: physical)
    monkeypatch.setattr(cpu.os, "cpu_count", lambda: logical)
    monkeypatch.setattr(cpu, "_compiler_flags", lambda: None)
    return cpu.feature_report()


def _features(**flags):
    base = {"avx": True, "avx2": False, "fma": False, "f16c": False}
    base.update(flags)
    return {"model": "test", "physical": 1, "logical": 1, "features": base}


def test_feature_report_reads_the_linux_cpu(monkeypatch):
    report = _report(monkeypatch)
    assert report == {
        "model": "Example i5",
        "physical": 6,
        "logical": 12,
        "features": {"avx": True, "avx2": True, "fma": True, "f16c": True},
    }


def test_avx_only_is_enough(monkeypatch):
    report = _report(monkeypatch, INFO.replace(" avx2 fma f16c", ""))
    assert cpu.missing_features(report) == ()
    assert cpu.cpu_tier(report) == "avx-only"


def test_avx2_and_f16c_without_fma_uses_the_no_fma_path(monkeypatch):
    report = _report(monkeypatch, INFO.replace(" fma", ""))
    assert report["features"] == {"avx": True, "avx2": True, "fma": False, "f16c": True}
    assert cpu.missing_features(report) == ()
    assert cpu.cpu_tier(report) == "avx2-f16c-no-fma"


def test_full_isa_uses_the_fma_path(monkeypatch):
    assert cpu.cpu_tier(_report(monkeypatch)) == "avx2-fma-f16c"


def test_missing_avx_is_named(monkeypatch):
    report = _report(monkeypatch, INFO.replace(" avx ", " "))
    assert cpu.missing_features(report) == ("AVX",)


def test_unknown_flags_do_not_block_the_compile_time_guard(monkeypatch):
    report = _report(monkeypatch, text=None)
    assert report["features"] == {"avx": None, "avx2": None, "fma": None, "f16c": None}
    assert cpu.missing_features(report) == ()
    assert cpu.cpu_tier(report) == "unknown"


@pytest.mark.parametrize(("features", "tier"), (
    ({"avx2": True, "fma": True, "f16c": True}, "avx2-fma-f16c"),
    ({"avx2": True, "f16c": True}, "avx2-f16c-no-fma"),
    ({"avx2": True, "fma": True}, "avx2-fma"),
    ({"avx2": True}, "avx2"),
    ({}, "avx-only"),
))
def test_select_isa_picks_the_best_supported_tier(features, tier):
    report = _features(**features)
    assert cpu.cpu_tier(report) == tier
    assert cpu.select_isa(report, forced="auto") == tier


def test_select_isa_keeps_native_when_flags_are_unknown():
    report = {"model": "test", "physical": 1, "logical": 1,
              "features": {"avx": None, "avx2": None, "fma": None, "f16c": None}}
    assert cpu.select_isa(report, forced="auto") is None


def test_forced_isa_must_be_supported_and_known():
    with pytest.raises(SystemExit, match="needs AVX, AVX2"):
        cpu.select_isa(_features(), forced="avx2")
    with pytest.raises(SystemExit, match="must be auto or one of"):
        cpu.select_isa(_features(), forced="sse2")


def test_forced_isa_can_lower_or_raise_the_tier(monkeypatch):
    report = _features(avx2=True, fma=True, f16c=True)
    assert cpu.select_isa(report, forced="avx-only") == "avx-only"
    monkeypatch.setenv("KINDI_CPU_ISA", "avx2")
    assert cpu.select_isa(report) == "avx2"


def test_check_features_explains_the_missing_isa(monkeypatch):
    report = _report(monkeypatch, INFO.replace(" avx ", " "))
    monkeypatch.setattr(cpu, "feature_report", lambda: report)
    with pytest.raises(SystemExit, match="needs AVX; this CPU lacks AVX"):
        cpu.check_features()
    assert cpu.cpu_tier(report) == "unsupported"
