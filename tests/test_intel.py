"""G0 slice of REQ-0021: Intel iGPU discovery/eligibility/budget (CPU-only)."""
import json
import os
import subprocess
import sys

import pytest

from kindi import intel


def _fake_sysfs(tmp_path, cards):
    """Build a fake /sys/class/drm tree: [(card, vendor, device, driver)]."""
    root = tmp_path / "drm"
    for card, vendor, device, driver in cards:
        d = root / card / "device"
        d.mkdir(parents=True)
        (d / "vendor").write_text(vendor)
        (d / "device").write_text(device)
        (d / "uevent").write_text("MAJOR=226\nMINOR=0\nDEVNAME=%s\nDRIVER=%s\n" % (card, driver))
    dri = tmp_path / "dri"
    dri.mkdir()
    (dri / "renderD128").touch()
    (dri / "renderD129").touch()
    return str(root), str(dri)


def test_discover_finds_intel_card_and_skips_others(tmp_path):
    drm, dri = _fake_sysfs(tmp_path, [
        ("card0", "0x8086", "0x9a49", "i915"),       # Iris Xe
        ("card1", "0x10de", "0x2484", "nvidia"),      # not Intel: skipped
    ])
    devs = intel.discover(drm_root=drm, dri_root=dri)
    assert len(devs) == 1
    dev = devs[0]
    assert dev["device_id"] == 0x9A49
    assert dev["name"] == "Intel Iris Xe Graphics"
    assert dev["arch"] == "Xe-LP"
    assert dev["driver"] == "i915"
    assert dev["render_node"].endswith("renderD128")
    assert dev["integrated"] is True
    assert dev["source"] == "sysfs"


def test_discover_without_sysfs_is_honest(tmp_path):
    assert intel.discover(drm_root=str(tmp_path / "nope"), dri_root=str(tmp_path)) == []


def test_eligibility_floor():
    assert intel.eligible({"arch": "Xe-LP", "driver": "i915"})
    assert intel.eligible({"arch": "Xe2-LPG", "driver": "xe"})
    assert intel.eligible({"arch": "Xe-HPG", "driver": ""})
    assert not intel.eligible({"arch": None, "driver": "i915"})          # unknown id
    assert not intel.eligible({"arch": "Gen9", "driver": "i915"})        # UHD 630 class
    assert not intel.eligible({"arch": "Xe-LP", "driver": "snd_hda"})    # not bound to GPU driver


def test_eligibility_reason_names_the_cause():
    assert "unknown PCI" in intel.eligibility_reason({"arch": None, "device_id": 0x1234})
    assert "below the oneAPI floor" in intel.eligibility_reason(
        {"arch": "Gen9", "name": "Intel UHD 630", "driver": "i915"})
    assert "eligible" in intel.eligibility_reason(
        {"arch": "Xe2-LPG", "name": "Intel Arc 140V (built-in)", "driver": "xe"})


def test_shared_budget_rules():
    assert intel.shared_budget_mb(65536) == 32768          # Windows half-RAM cap
    assert intel.shared_budget_mb(65536, 57) == 37355      # Core Ultra 2+ override default
    assert intel.shared_budget_mb(65536, 93) == 60948      # Panther Lake example
    assert intel.shared_budget_mb(65536, 150) == 65536     # clamped


def test_fits_includes_reserve():
    assert intel.fits(32768, 5200, 2000)                   # 2B-class pack
    assert not intel.fits(7300, 5200, 2000)                # reserve pushes it out
    assert intel.fits(7300, 5200, 2000, reserve_mb=0)


def test_tandem_gpu_contract():
    class Good(object):
        D = 2048
        vocab = 1000

        def get_hidden(self):
            return None

        def set_hidden(self, h):
            pass

    class Bad(object):
        pass

    assert intel.tandem_gpu_contract(Good()) == []
    assert set(intel.tandem_gpu_contract(Bad())) == {"get_hidden", "set_hidden", "D", "vocab"}


def test_make_engine_refuses_intel_loudly(monkeypatch):
    from kindi import engine
    monkeypatch.setenv("KINDI_DEVICE", "intel")
    with pytest.raises(SystemExit) as exc:
        engine.make_engine("nonexistent-pack")
    assert "not implemented yet" in str(exc.value)
    assert "REQ-0020/REQ-0021" in str(exc.value)


def test_no_hard_oneapi_dependency():
    # kindi/intel.py must import and work with no oneAPI/SYCL present (this host).
    assert not os.path.exists("/opt/intel/oneapi")
    import kindi.intel  # noqa: F401


def _cpuinfo_fixture(tmp_path):
    path = tmp_path / "cpuinfo"
    path.write_text(
        "processor\t: 0\n"
        "vendor_id\t: GenuineIntel\n"
        "cpu family\t: 6\n"
        "model\t\t: 158\n"
        "model name\t: Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz\n"
        "stepping\t: 10\n"
        "cpu cores\t: 6\n"
        "physical id\t: 0\n"
        "core id\t\t: 0\n"
        "flags\t\t: fpu vme avx avx2 fma f16c avx_vnni amx_tile\n\n"
        "processor\t: 1\n"
        "vendor_id\t: GenuineIntel\n"
        "cpu family\t: 6\n"
        "model\t\t: 158\n"
        "model name\t: Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz\n"
        "stepping\t: 10\n"
        "cpu cores\t: 6\n"
        "physical id\t: 0\n"
        "core id\t\t: 1\n"
        "flags\t\t: fpu vme avx avx2 fma f16c avx_vnni amx_tile\n")
    return str(path)


def _command(stdout="", returncode=0, stderr=""):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


def _probe_runner(seen):
    def runner(command):
        seen.append(list(command))
        if command[0] == "lscpu":
            return _command("Vendor ID: GenuineIntel\nModel name: Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz\n"
                            "CPU family: 6\nModel: 158\nStepping: 10\nCPU(s): 2\n"
                            "Core(s) per socket: 6\nSocket(s): 1\nFlags: avx avx2 fma avx_vnni amx_tile\n")
        if command[0] == "lspci":
            return _command(
                "0000:03:00.0 VGA compatible controller [0300]: Intel Corporation DG2 [Arc A770] "
                "[8086:56a0] (rev 08)\n"
                "\tSubsystem: Intel Corporation Device [8086:1020]\n"
                "\tKernel driver in use: i915\n")
        if command[0] == "sycl-ls":
            return _command("[level_zero:gpu:0] Intel(R) Arc(TM) A770 Graphics 16GB\n"
                            "[opencl:gpu:0] Intel(R) Arc(TM) A770 Graphics 16GB\n")
        if command[0] == "clinfo":
            return _command("Platform Name: Intel(R) OpenCL Graphics\n")
        if command[0] == sys.executable:
            return _command(json.dumps({"version": "2025.1.0", "devices": ["CPU", "GPU.0"]}) + "\n")
        raise AssertionError("unexpected command: %r" % (command,))
    return runner


def test_probe_collects_cpu_devices_npu_and_tool_versions(tmp_path):
    drm, dri = _fake_sysfs(tmp_path, [("card0", "0x8086", "0x56a0", "i915")])
    accel = tmp_path / "accel"
    npu = accel / "accel0" / "device"
    npu.mkdir(parents=True)
    (npu / "vendor").write_text("0x8086")
    (npu / "device").write_text("0x7d1d")
    (npu / "uevent").write_text("DRIVER=intel_vpu\n")
    oneapi = tmp_path / "oneapi"
    oneapi.mkdir()
    seen = []

    report = intel.probe(drm_root=drm, dri_root=dri, accel_root=str(accel),
                         cpuinfo_path=_cpuinfo_fixture(tmp_path),
                         oneapi_root=str(oneapi), runner=_probe_runner(seen))

    assert report["probe"] == "intel-info"
    assert report["schema"] == 1
    assert report["read_only"] is True
    assert report["support_claim"] is False
    assert report["backend"] == "not implemented"
    assert [command[0] for command in seen] == ["lscpu", "lspci", "sycl-ls", "clinfo", sys.executable]

    cpu = report["cpu"]
    assert cpu["vendor"] == "GenuineIntel"
    assert cpu["model"] == "Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz"
    assert cpu["family"] == 6
    assert cpu["model_id"] == 158
    assert cpu["stepping"] == 10
    assert cpu["physical"] == 6
    assert cpu["logical"] == 2
    assert cpu["features"] == {"avx2": True, "fma": True, "avx_vnni": True, "amx": True}

    assert len(report["devices"]) == 1
    device = report["devices"][0]
    assert device["bdf"] == "0000:03:00.0"     # corroborated by lspci
    assert device["vendor_id"] == 0x8086
    assert device["device_id"] == 0x56A0
    assert device["name"] == "Intel Arc A770"
    assert device["arch"] is None          # named, but not a measured support claim
    assert device["driver"] == "i915"
    assert device["render_node"].endswith("renderD128")
    assert device["integrated"] is False
    assert device["source"] == "sysfs"

    assert report["npu"]["available"] is True
    assert report["npu"]["present"] is True
    assert report["npu"]["devices"][0]["driver"] == "intel_vpu"
    assert report["oneapi"] == {"root": str(oneapi), "present": True, "source": "default"}
    assert report["level_zero"]["devices"] == [{
        "backend": "level_zero", "kind": "gpu", "index": 0,
        "description": "Intel(R) Arc(TM) A770 Graphics 16GB"}]
    assert report["openvino"] == {"available": True, "version": "2025.1.0",
                                  "devices": ["CPU", "GPU.0"], "error": None}
    assert report["commands"]["clinfo"]["available"] is True

    text = intel.render(report)
    assert "no Intel backend or support claim" in text
    assert "Intel Arc A770" in text


def test_probe_keeps_missing_sources_explicit(tmp_path):
    def missing(command):
        raise FileNotFoundError(2, "No such file or directory", command[0])

    report = intel.probe(
        drm_root=str(tmp_path / "no-drm"), dri_root=str(tmp_path / "no-dri"),
        accel_root=str(tmp_path / "no-accel"),
        cpuinfo_path=str(tmp_path / "no-cpuinfo"),
        oneapi_root=str(tmp_path / "no-oneapi"), runner=missing)

    assert report["cpu"]["vendor"] is None
    assert report["cpu"]["features"] == {"avx2": None, "fma": None, "avx_vnni": None, "amx": None}
    assert report["devices"] == []
    assert report["npu"]["available"] is False
    assert report["npu"]["present"] is False
    assert "error" in report["npu"]
    assert report["oneapi"]["present"] is False
    for name in ("lscpu", "lspci", "sycl-ls", "clinfo"):
        result = report["commands"][name]
        assert result["available"] is False
        assert "command not found" in result["error"]
    assert report["level_zero"]["devices"] == []
    assert report["level_zero"]["available"] is False
    assert "command not found" in report["level_zero"]["error"]
    assert report["openvino"]["available"] is False
    assert "command not found" in report["openvino"]["error"]


def test_intel_info_cli_prints_json(monkeypatch, capsys):
    from kindi import __main__ as cli
    report = {"probe": "intel-info", "read_only": True, "cpu": {}, "devices": []}
    monkeypatch.setattr(intel, "probe", lambda: report)
    assert cli.main(["intel-info", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == report
