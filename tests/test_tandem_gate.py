# SPDX-License-Identifier: AGPL-3.0-only
"""The opt-in tandem GPU gate's planning, sampling and record logic.

These tests never import CUDA and never build an engine. The real gate is
``tools/tandem_gate.py``, which refuses without a dense pack and a usable card; this file
pins the path-free record, the ceiling planner, the sampled peak arithmetic and the dry run.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("tandem_gate", ROOT / "tools" / "tandem_gate.py")
gate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gate)

MIB = gate.MIB
DIMS = {"layers": 32, "full_attention_interval": 4, "kv_heads": 4, "head_dim": 256}


def _sizes():
    # every fake layer costs ~2.1 MiB, so a 200 MiB ceiling leaves the attention layers on
    # the GPU and moves the 24 GDN layers to the CPU.
    layer = 4096 * 2048 // 4 + 4096 * 16 * 2
    return {"layers": {il: layer for il in range(DIMS["layers"])}, "f32_layers": {},
            "embed": 0, "head": 0, "f32": 0, "f32_global": 0}


def _fake_pack(root, backend=None):
    pack = root / "bonsai-27b"
    pack.mkdir()
    manifest = {
        "model": "Ternary-Bonsai-2-27B",
        "source_file": "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
        "source_sha256": "ab" * 32,
        "dimensions": dict(DIMS),
        "bcodes_bytes": 1,
        "dpm_bytes": 1,
        "f32_bytes": 0,
    }
    if backend:
        manifest["backend"] = backend
    (pack / "kindi_pack.json").write_text(json.dumps(manifest), encoding="utf-8")
    tensors = {}
    for il in range(DIMS["layers"]):
        tensors["blk.%d.ffn_up.weight" % il] = {"rows": 4096, "ne0": 2048,
                                                "bc_off": 0, "dpm_off": 0}
    (pack / "bcodes_index.json").write_text(json.dumps({"tensors": tensors}), encoding="utf-8")
    (pack / "index.json").write_text(json.dumps({"tensors": {}}), encoding="utf-8")
    return pack


def _card(free=8 << 30, total=8 << 30):
    return {"count": 1, "name": "Test Card", "cc": [8, 9], "free": free, "total": total,
            "cupy": "?", "driver": 0, "runtime": 0}


def test_pack_identity_has_no_directory_and_reports_the_tensor_shape(tmp_path):
    pack = _fake_pack(tmp_path)
    path, manifest, identity = gate.pack_identity(str(pack))
    assert path == str(pack)
    assert manifest["model"] == "Ternary-Bonsai-2-27B"
    assert identity["layers"] == 32 and identity["name"] == "bonsai-27b"
    assert str(tmp_path) not in json.dumps(identity)
    assert identity["source_sha256"] == "ab" * 32


def test_pack_identity_rejects_non_dense_backends_and_missing_packs(tmp_path):
    pack = _fake_pack(tmp_path, backend="kindi_gemma4")
    with pytest.raises(gate.GateError) as caught:
        gate.pack_identity(str(pack))
    assert caught.value.code == 2
    with pytest.raises(gate.GateError) as caught:
        gate.pack_identity(str(tmp_path / "missing"))
    assert caught.value.code == 2


def test_token_ids_are_validated():
    assert gate._token_ids("1, 2;3") == [1, 2, 3]
    for text in ("", "1,,2", "-1", str(gate.VOCAB)):
        with pytest.raises(argparse.ArgumentTypeError):
            gate._token_ids(text)


def test_the_ceiling_planner_moves_gdn_layers_first():
    plan = gate.plan_for_gate(DIMS, _sizes(), 1024, 8 << 30, 0, 200)
    assert plan["cpu_layers"], "the 200 MiB ceiling must force a split"
    assert all((il + 1) % 4 for il in plan["cpu_layers"])
    assert plan["gpu_layers"] == [il for il in range(32) if il not in set(plan["cpu_layers"])]
    assert plan["estimate_bytes"] <= plan["budget_bytes"] == 200 * MIB
    assert plan["full_estimate_bytes"] > plan["budget_bytes"]


def test_the_planner_refuses_when_no_split_is_needed_or_nothing_fits():
    with pytest.raises(gate.GateError, match="already fits"):
        gate.plan_for_gate(DIMS, _sizes(), 1024, 64 << 30, 0, 4096)
    with pytest.raises(gate.GateError, match="cannot fit"):
        gate.plan_for_gate(DIMS, _sizes(), 1024, 8 << 30, 0, 100)


def test_peak_summary_uses_the_start_free_figure_and_the_sampled_minimum():
    free = 8 << 30
    got = gate.peak_summary(free, free, [free, free - 100 * MIB, free - 40 * MIB])
    assert got["peak_alloc_bytes"] == 100 * MIB
    assert got["peak_alloc_mib"] == 100.0
    assert got["min_free_bytes"] == free - 100 * MIB
    assert got["peak_used_bytes"] == 100 * MIB
    assert got["samples"] == 3
    empty = gate.peak_summary(free, free, [])
    assert empty["peak_alloc_bytes"] == 0 and empty["samples"] == 1


def test_verdict_names_the_budget_reserve_context_and_verify_failures():
    plan = {"budget_bytes": 100 * MIB}
    measure = {"peak_alloc_bytes": 120 * MIB, "peak_alloc_mib": 120.0,
               "min_free_bytes": 50 * MIB, "min_free_mib": 50.0}
    engine = {"context": 1024}
    got = gate.gate_verdict(plan, measure, engine, None, {"status": "pass"}, 0, 1024)
    assert not got["fits"]
    assert any("over the 100 MiB budget" in reason for reason in got["reasons"])
    ok = gate.gate_verdict({"budget_bytes": 100 * MIB},
                           {"peak_alloc_bytes": 80 * MIB, "peak_alloc_mib": 80.0,
                            "min_free_bytes": 500 * MIB, "min_free_mib": 500.0},
                           engine, None, {"status": "pass"}, 0, 1024)
    assert ok["fits"]
    reduced = gate.gate_verdict({"budget_bytes": 100 * MIB},
                                {"peak_alloc_bytes": 80 * MIB, "peak_alloc_mib": 80.0,
                                 "min_free_bytes": 500 * MIB, "min_free_mib": 500.0},
                                {"context": 512}, None, {"status": "pass"}, 0, 1024)
    assert not reduced["fits"] and any("reduced the context" in r for r in reduced["reasons"])
    failed = gate.gate_verdict({"budget_bytes": 100 * MIB},
                               {"peak_alloc_bytes": 80 * MIB, "peak_alloc_mib": 80.0,
                                "min_free_bytes": 500 * MIB, "min_free_mib": 500.0},
                               engine, None, {"status": "fail", "exit_code": 1}, 0, 1024)
    assert not failed["fits"] and any("verify failed" in r for r in failed["reasons"])


def test_verify_environment_drops_inherited_kindi_switches(monkeypatch):
    monkeypatch.setenv("KINDI_SPEC", "1")
    monkeypatch.setenv("KINDI_FD", "1")
    monkeypatch.setenv("KINDI_CPU_CACHE", "/tmp/kcpu")
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    env = gate._verify_env(argparse.Namespace(vram_budget_mib=4096, reserve_mib=512))
    assert "KINDI_FD" not in env and env["KINDI_SPEC"] == "0"
    assert env["KINDI_CARDS"] == "1" and env["KINDI_TANDEM_GPU"] == "plan"
    assert env["KINDI_CPU_CACHE"] == "/tmp/kcpu"
    assert env["OMP_NUM_THREADS"] == "3"


def test_dry_run_records_the_plan_without_a_card_or_an_engine(tmp_path, monkeypatch, capsys):
    pack = _fake_pack(tmp_path)
    out = tmp_path / "gate.json"
    monkeypatch.setattr(gate, "probe_gpu", lambda: _card())
    code = gate.main(["--pack", str(pack), "--ctx", "1024", "--vram-budget-mib", "200",
                      "--reserve-mib", "0", "--dry-run", "--json", str(out)])
    assert code == 0
    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["schema"] == gate.SCHEMA
    assert record["verdict"]["dry_run"] is True and record["verdict"]["fits"] is True
    assert record["measure"] is None and record["engine"] is None
    assert record["plan"]["cpu_layers"] and all((il + 1) % 4 for il in record["plan"]["cpu_layers"])
    assert record["plan"]["estimate_bytes"] <= record["plan"]["budget_bytes"] == 200 * MIB
    text = out.read_text(encoding="utf-8")
    assert str(tmp_path) not in text
    assert "dry run" in capsys.readouterr().out


def test_missing_pack_exits_two_before_probing_for_a_card(tmp_path, monkeypatch):
    def boom():
        raise AssertionError("the GPU must not be probed for a missing pack")

    monkeypatch.setattr(gate, "probe_gpu", boom)
    assert gate.main(["--pack", str(tmp_path / "missing"), "--vram-budget-mib", "4096"]) == 2
