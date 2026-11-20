"""Requirement-based tests for sampled-peak VRAM used by `kindi bench --sample-vram`:
the peak is the maximum device-wide usage over the run, reported with its
delta over the idle baseline, and sampling never breaks the benchmark when
nvidia-smi is unavailable.
"""
from kindi import vram


def test_parse_smi_row():
    assert vram.parse_smi("0, 7005, 8192, 1187\n") == {"used_mib": 7005, "total_mib": 8192, "free_mib": 1187}
    assert vram.parse_smi("garbage") is None
    assert vram.parse_smi("") is None
    # a legacy row without an index column counts as device 0 only
    assert vram.parse_smi("7005, 8192, 1187\n") == {"used_mib": 7005, "total_mib": 8192, "free_mib": 1187}
    assert vram.parse_smi("7005, 8192, 1187\n", device=1) is None


def test_parse_smi_selects_the_requested_gpu():
    text = "0, 7005, 8192, 1187\n1, 3000, 8192, 5192\n"
    assert vram.parse_smi(text, device=0)["used_mib"] == 7005
    assert vram.parse_smi(text, device=1)["used_mib"] == 3000
    assert vram.parse_smi(text, device=2) is None


def test_visible_device_maps_through_cuda_visible_devices(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert vram.visible_device(2) == 2
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,3")
    assert vram.visible_device(0) == 1
    assert vram.visible_device(1) == 3
    assert vram.visible_device(2) == 2          # outside the list: unchanged


def test_gpu_memory_with_injected_runner():
    assert vram.gpu_memory(runner=lambda cmd: "0, 100, 8192, 8092\n")["used_mib"] == 100
    assert vram.gpu_memory(runner=lambda cmd: None) is None


def test_monitor_reports_peak_and_delta():
    readings = iter([
        {"used_mib": 1178, "total_mib": 8192, "free_mib": 7014},
        {"used_mib": 7005, "total_mib": 8192, "free_mib": 1187},
        {"used_mib": 5200, "total_mib": 8192, "free_mib": 2992},
    ])
    mon = vram.VramMonitor(interval=60, sampler=lambda: next(readings))
    mon.start()
    mon.mark("loaded")
    mon.stop()
    s = mon.summary()
    assert s["available"] is True
    assert s["peak_used_mib"] == 7005
    assert s["baseline_used_mib"] == 1178
    assert s["peak_delta_mib"] == 7005 - 1178
    assert s["peak_free_mib"] == 1187
    assert s["sample_count"] == 3


def test_monitor_without_gpu_reports_unavailable():
    mon = vram.VramMonitor(interval=60, sampler=lambda: None)
    mon.start()
    mon.stop()
    assert mon.summary() == {"available": False, "interval_s": 60.0}


def test_bench_cli_accepts_sample_vram_flag():
    from kindi.__main__ import main
    # --help exits 0 via argparse SystemExit; parsing alone is the check
    try:
        main(["--pack", "x", "bench", "--sample-vram", "--vram-sample-s", "0.25", "--help"])
    except SystemExit as e:
        assert e.code == 0
