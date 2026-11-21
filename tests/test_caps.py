"""`kindi caps`: identification that still answers when the card tooling is missing."""
from kindi import caps


def test_parse_smi_reads_the_documented_columns():
    text = ("0, NVIDIA GeForce RTX 4060 Ti, GPU-78de4e15-f013-ab5a-9843-1545356215e3, "
            "00000000:01:00.0, 8.9, 8188, 7902, 617.42\n"
            "1, NVIDIA GeForce GTX 1080, GPU-11111111-2222-3333-4444-555555555555, "
            "00000000:02:00.0, 6.1, 8192, 8100, 617.42\n")
    rows = caps.parse_smi(text)
    assert [r["name"] for r in rows] == ["NVIDIA GeForce RTX 4060 Ti", "NVIDIA GeForce GTX 1080"]
    assert [r["compute_cap"] for r in rows] == ["8.9", "6.1"]
    assert rows[0]["memory_total_mib"] == 8188 and rows[0]["memory_free_mib"] == 7902
    assert rows[0]["pci_bus_id"] == "00000000:01:00.0"
    assert rows[0]["driver"] == "617.42"


def test_parse_smi_accepts_a_driver_without_compute_cap():
    # Pre-CUDA-11.6 drivers have no compute_cap column at all.
    text = "0, NVIDIA GeForce RTX 3060, GPU-abc, 00000000:01:00.0, 12288, 12000, 535.104\n"
    (row,) = caps.parse_smi(text)
    assert row["compute_cap"] is None
    assert row["memory_total_mib"] == 12288
    assert row["driver"] == "535.104"


def test_parse_smi_marks_an_unreported_capability_as_unknown():
    text = "0, Tesla P40, GPU-x, 00000000:01:00.0, [N/A], 24576, 24000, 617.42\n"
    assert caps.parse_smi(text)[0]["compute_cap"] is None


def test_parse_smi_ignores_empty_and_short_rows():
    assert caps.parse_smi("\n\n0, only, three\n") == []
    assert caps.parse_smi("") == []


def test_card_features_gate_on_compute_capability():
    pascal = caps.card_features("6.1")
    assert pascal["tensor_cores"] is False and pascal["bf16"] is False and pascal["fp8"] is False
    turing = caps.card_features("7.5")
    assert turing["tensor_cores"] is True and turing["bf16"] is False
    ampere = caps.card_features("8.6")
    assert ampere["bf16"] is True and ampere["tf32"] is True and ampere["fp8"] is False
    ada = caps.card_features("8.9")
    assert ada["fp8"] is True and ada["fp4"] is False
    hopper = caps.card_features("9.0")
    assert hopper["fp8"] is True and hopper["fp4"] is False
    blackwell = caps.card_features("12.0")
    assert blackwell["fp8"] is True and blackwell["fp4"] is True
    assert caps.card_features("10.0")["fp4"] is True


def test_card_features_are_silicon_facts_not_library_algorithms():
    """The hardware table may say the silicon has FP8; it must not report an algorithm such as
    FlashAttention-2, which belongs to the usable-path half (`kindi_paths`) instead."""
    feats = caps.card_features("8.9")
    assert set(feats) == {"tensor_cores", "bf16", "tf32", "fp8", "fp4", "async_copy"}
    assert "flash_attention_2_class" not in feats


def test_card_features_refuses_to_guess_without_a_capability():
    assert caps.card_features(None) is None
    assert caps.card_features("") is None
    assert caps.card_features("N/A") is None


def test_kindi_paths_only_name_the_gates_that_exist():
    pascal = caps.kindi_paths("6.1")
    assert "FP8 Gemma 4 path (on by default, KINDI_G4_FP8=0 disables)" not in pascal
    assert "plan engine (tensor-core MMA)" not in pascal
    ada = caps.kindi_paths("8.9")
    assert any("plan engine" in p for p in ada)
    assert any("FP8" in p for p in ada)
    assert caps.kindi_paths(None) == ["5-trit kernels", "dense CPU+GPU tandem split"]


def test_parse_lscpu_reads_caches_and_extended_flags():
    text = ("Flags:            fpu vme avx avx2 fma f16c avx512f amx_tile\n"
            "L2 cache:         256 KiB\n"
            "L3 cache:         12 MiB\n")
    out = caps.parse_lscpu(text)
    assert out["l2_kb"] == 256 and out["l3_kb"] == 12
    assert out["avx512"] is True and out["amx"] is True


def test_parse_lscpu_reports_a_missing_extended_flag_as_false():
    out = caps.parse_lscpu("Flags: fpu avx avx2 fma f16c\n")
    assert out["avx512"] is False and out["amx"] is False


def test_parse_wmi_reads_one_processor_or_an_array():
    one = caps.parse_wmi('{"L2CacheSize": 1536, "L3CacheSize": 12288}')
    assert one["l2_kb"] == 1536 and one["l3_kb"] == 12288
    many = caps.parse_wmi('[{"L2CacheSize": 512, "L3CacheSize": 8192}, {"L3CacheSize": 999}]')
    assert many["l2_kb"] == 512 and many["l3_kb"] == 8192
    assert caps.parse_wmi("not json")["l3_kb"] is None


def test_cpu_report_keeps_the_engine_probe_and_adds_what_it_cannot_see(monkeypatch):
    monkeypatch.setattr(caps, "cpu_extras", lambda: {"l2_kb": 1536, "l3_kb": 12288,
                                                     "avx512": None, "amx": None})
    report = caps.cpu_report()
    assert report["physical"] and report["logical"]
    assert report["tier"] in ("avx2-fma-f16c", "avx2-f16c-no-fma", "avx2-fma", "avx2",
                              "avx-only", "unknown", "unsupported")
    assert report["threads_recommended"] == report["physical"]
    assert report["l3_kb"] == 12288


def test_probe_without_any_card_tool_says_none_detected(monkeypatch):
    monkeypatch.setattr(caps, "_from_smi", lambda: [])
    monkeypatch.setattr(caps, "_from_nvml", lambda: [])
    monkeypatch.setattr(caps, "_from_cupy", lambda: [])
    monkeypatch.setattr(caps, "cpu_extras", lambda: {"l2_kb": None, "l3_kb": None,
                                                     "avx512": None, "amx": None})
    report = caps.probe()
    assert report["gpus"] == [] and report["gpu_source"] == "none"
    text = "\n".join(caps.render(report))
    assert "none detected" in text
    assert "CPU path" in text


def test_probe_annotates_every_card_with_features_and_paths(monkeypatch):
    monkeypatch.setattr(caps, "_from_smi", lambda: caps.parse_smi(
        "0, NVIDIA GeForce RTX 4060 Ti, GPU-u, 00000000:01:00.0, 8.9, 8188, 7902, 617.42\n"))
    monkeypatch.setattr(caps, "cpu_extras", lambda: {"l2_kb": 1536, "l3_kb": 12288,
                                                     "avx512": None, "amx": None})
    report = caps.probe()
    (card,) = report["gpus"]
    assert card["features"]["fp8"] is True
    assert any("FP8" in p for p in card["kindi"])
    text = "\n".join(caps.render(report))
    assert "sm_89" in text and "Kindi can use" in text and "card 0" in text


def test_windows_features_are_read_from_kernel32(monkeypatch):
    import ctypes

    class _K32(object):
        def IsProcessorFeaturePresent(self, number):
            return {39: 1, 40: 1, 41: 0}.get(number, 0)

    class _Windll(object):
        kernel32 = _K32()

    monkeypatch.setattr(ctypes, "windll", _Windll(), raising=False)
    got = caps.windows_features()
    assert got == {"avx": True, "avx2": True, "avx512": False}
    assert caps.WINDOWS_PF == (("avx", 39), ("avx2", 40), ("avx512", 41))


def test_a_missing_kernel32_leaves_the_flags_unknown(monkeypatch):
    import ctypes

    class _Broken(object):
        def __getattr__(self, name):
            raise OSError("no kernel32 here")

    monkeypatch.setattr(ctypes, "windll", _Broken(), raising=False)
    assert caps.windows_features() == {"avx": None, "avx2": None, "avx512": None}


def test_detected_windows_flags_fill_gaps_but_never_override_the_engine(monkeypatch):
    from kindi import cpu as cpu_backend

    monkeypatch.setattr(caps, "cpu_extras", lambda: {"l2_kb": None, "l3_kb": None,
                                                     "avx512": None, "amx": None})
    monkeypatch.setattr(caps, "windows_features",
                        lambda: {"avx": False, "avx2": True, "avx512": False})
    monkeypatch.setattr(caps.os, "name", "nt")
    monkeypatch.setattr(cpu_backend, "feature_report",
                        lambda: {"model": "cpu", "physical": 4, "logical": 8,
                                 "features": {"avx": True, "avx2": None, "fma": None,
                                              "f16c": None}})

    report = caps.cpu_report()
    assert report["features"]["avx"] is True        # the engine's own answer wins
    assert report["features"]["avx2"] is True       # gap filled from kernel32
    assert report["avx512"] is False                # also filled, and a real value not None
    text = "\n".join(caps.render({"gpus": [], "gpu_source": "none", "cpu": report}))
    assert "AVX2=yes" in text
    assert "FMA3/F16C not exposed by this OS" in text


def test_toolchain_reports_the_resolved_compiler(monkeypatch):
    monkeypatch.setattr(caps.shutil, "which", lambda name: "C:/llvm/bin/clang.exe")
    found = caps.toolchain()
    assert found["ok"] is True and found["path"].endswith("clang.exe")


def test_a_missing_toolchain_is_a_named_blocker_not_a_silence(monkeypatch):
    monkeypatch.setattr(caps.shutil, "which", lambda name: None)
    monkeypatch.setattr(caps, "cpu_extras", lambda: {"l2_kb": None, "l3_kb": None,
                                                     "avx512": None, "amx": None})
    missing = caps.toolchain()
    assert missing["ok"] is False and missing["path"] is None
    text = "\n".join(caps.render({"gpus": [], "gpu_source": "none", "cpu": caps.cpu_report()}))
    assert "cannot be built here" in text
    assert "KINDI_CC" in text


def test_render_says_unknown_rather_than_guessing_a_capability(monkeypatch):
    monkeypatch.setattr(caps, "_from_smi", lambda: caps.parse_smi(
        "0, Tesla P40, GPU-u, 00000000:01:00.0, [N/A], 24576, 24000, 617.42\n"))
    monkeypatch.setattr(caps, "cpu_extras", lambda: {"l2_kb": None, "l3_kb": None,
                                                     "avx512": None, "amx": None})
    text = "\n".join(caps.render(caps.probe()))
    assert "hardware features: unknown" in text
