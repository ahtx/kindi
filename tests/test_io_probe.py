# SPDX-License-Identifier: AGPL-3.0-only
"""`kindi io`: the measured memory-tier rates, and the honesty of each number."""
import json

from kindi import io_probe


def test_ram_copy_measures_a_real_rate():
    r = io_probe.ram_copy(mib=4, repeats=2)
    assert r["available"] is True
    assert r["mib"] == 4
    assert r["mib_s"] > 0 and r["seconds"] > 0
    assert "memcpy" in r["method"]


def test_disk_read_is_unbuffered_or_labelled_and_leaves_no_file(tmp_path):
    r = io_probe.disk_read(mib=8, directory=str(tmp_path))
    assert r["available"] is True
    assert r["mib_s"] > 0
    assert isinstance(r["unbuffered"], bool)
    assert ("buffered" in r["method"]) == (not r["unbuffered"])
    assert list(tmp_path.iterdir()) == []


def test_buffered_fallback_is_labelled(tmp_path, monkeypatch):
    def refuse(*a, **k):
        raise OSError("no direct I/O here")

    monkeypatch.setattr(io_probe, "_windows_nobuffer_read", refuse)
    monkeypatch.setattr(io_probe, "_posix_direct_read", refuse)
    r = io_probe.disk_read(mib=4, directory=str(tmp_path))
    assert r["unbuffered"] is False
    assert "buffered" in r["method"]
    assert r["mib_s"] > 0


def test_pcie_is_unavailable_without_cuda(monkeypatch):
    monkeypatch.setattr(io_probe, "_cuda", lambda: None)
    r = io_probe.pcie_copy()
    assert r["available"] is False
    assert "CuPy" in r["reason"]
    assert io_probe.probe(mib=4, skip=("ram", "disk"))["pcie"]["available"] is False


def test_probe_skips_what_is_asked():
    report = io_probe.probe(mib=4, skip=("ram", "disk", "pcie"))
    assert all(report[k]["available"] is False and report[k]["reason"] == "skipped"
               for k in ("ram", "disk", "pcie"))


def test_rates_are_normalized_and_a_buffered_disk_is_flagged():
    report = {
        "ram": {"available": True, "mib_s": 13456.7},
        "disk": {"available": True, "mib_s": 2509.8, "unbuffered": False},
        "pcie": {"available": True, "h2d_gib_s": 6.0, "d2h_gib_s": 6.1, "device": "RTX"},
    }
    rates = io_probe.rates(report)
    assert rates["ram_bytes_s"] == 13456.7 * 2 ** 20
    assert rates["pcie_bytes_s"] == 6.0 * 2 ** 30
    assert "disk_bytes_s" not in rates and rates["disk_buffered"] is True
    assert "buffered" in io_probe.rate_line(rates)
    assert "RAM 13456.7 MiB/s" in io_probe.rate_line(rates)
    assert "PCIe H2D 6.0 GiB/s" in io_probe.rate_line(rates)


def test_render_names_the_method_and_the_boundary():
    report = io_probe.probe(mib=4, skip=("disk", "pcie"))
    text = "\n".join(io_probe.render(report))
    assert "RAM copy:" in text and "MiB/s" in text
    assert "no other engine's rate is ours" in text


def test_unknown_skip_name_is_rejected_before_any_probe(capsys, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("a probe ran despite an invalid --skip name")

    monkeypatch.setattr(io_probe, "ram_copy", fail)
    monkeypatch.setattr(io_probe, "disk_read", fail)
    monkeypatch.setattr(io_probe, "pcie_copy", fail)
    assert io_probe.main(["--mib", "4", "--skip", "ram,diskk", "--json"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "unknown probe: diskk" in captured.err
    assert "valid probes: ram, disk, pcie" in captured.err


def test_cli_json_is_machine_readable(capsys):
    assert io_probe.main(["--mib", "4", "--skip", "disk,pcie", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert set(report) == {"ram", "disk", "pcie"}
    assert report["ram"]["available"] is True
    assert report["disk"]["reason"] == "skipped"


def test_cli_json_file_saves_a_report_that_plan_and_autotune_can_reuse(tmp_path, capsys):
    path = tmp_path / "rates" / "io.json"
    assert io_probe.main(["--mib", "4", "--skip", "disk,pcie", "--json", str(path)]) == 0
    assert set(json.loads(capsys.readouterr().out)) == {"ram", "disk", "pcie"}
    saved = io_probe.read_report(str(path))
    assert saved["ram"]["available"] is True
    rates = io_probe.load_rates(str(path))
    assert rates["ram_bytes_s"] == saved["ram"]["mib_s"] * 2 ** 20


RATE_REPORT = {
    "ram": {"available": True, "mib_s": 100.0},
    "disk": {"available": True, "mib_s": 50.0, "unbuffered": True},
    "pcie": {"available": True, "h2d_gib_s": 2.0, "d2h_gib_s": 2.1, "device": "test"},
}


def test_a_missing_or_unusable_report_falls_back_to_one_probe(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(io_probe, "probe", lambda **kw: calls.append(kw["mib"]) or RATE_REPORT)
    missing = tmp_path / "missing.json"
    lines = []
    rates = io_probe.load_or_probe(str(missing), mib=32, log=lines.append)
    assert calls == [32]
    assert rates["pcie_bytes_s"] == 2.0 * 2 ** 30
    assert any("no saved I/O report at" in line and "measuring" in line for line in lines)

    broken = tmp_path / "broken.json"
    broken.write_text("{}", encoding="utf-8")
    lines = []
    assert io_probe.load_or_probe(str(broken), mib=16, log=lines.append)["ram_bytes_s"] == 100 * 2 ** 20
    assert calls == [32, 16]
    assert any("unusable saved I/O report" in line for line in lines)


def test_no_report_still_falls_back_and_says_how_to_save_one(monkeypatch):
    monkeypatch.setattr(io_probe, "probe", lambda **kw: RATE_REPORT)
    lines = []
    assert io_probe.load_or_probe(None, mib=8, log=lines.append)["ram_bytes_s"] == 100 * 2 ** 20
    assert any("no saved I/O report" in line and "kindi io --json FILE" in line for line in lines)


def test_the_cli_command_is_wired(capsys):
    from kindi import __main__ as cli
    assert cli.main(["io", "--mib", "4", "--skip", "disk,pcie", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["pcie"]["reason"] == "skipped"
