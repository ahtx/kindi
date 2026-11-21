"""The tracked stock-v0.6.1 greedy goldens are well formed and comparable."""
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import stock_parity as parity


def _load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def prompts():
    return _load(parity.DEFAULT_PROMPTS)


@pytest.fixture(scope="module")
def goldens():
    return _load(parity.DEFAULT_GOLDENS)


def test_prompts_are_the_fixed_lengths_and_prefixes(prompts):
    assert prompts["format"] == 1
    assert prompts["generated_tokens"] == parity.GENERATED_TOKENS
    assert prompts["context"] == parity.CONTEXT
    by_position = {p["position"]: p for p in prompts["prompts"]}
    assert sorted(by_position) == sorted(parity.POSITIONS)
    for pos in parity.POSITIONS:
        assert len(by_position[pos]["ids"]) == pos
        assert by_position[pos]["sha256"] == parity._sha256_text(
            parity._canonical(by_position[pos]["ids"]))
    assert by_position[100]["ids"] == by_position[2000]["ids"][:100]
    assert by_position[2000]["ids"] == by_position[8000]["ids"][:2000]


def test_prompt_source_hash_is_the_tracked_source(prompts):
    source = parity.SOURCE_TEMPLATE * parity.SOURCE_REPEATS
    assert prompts["source_sha256"] == parity._sha256_text(source)


def test_prompt_loader_rejects_a_changed_source_hash(tmp_path):
    prompts = _load(parity.DEFAULT_PROMPTS)
    prompts["source_sha256"] = "0" * 64
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps(prompts), encoding="utf-8")
    with pytest.raises(RuntimeError, match="source hash mismatch"):
        parity._load_prompts(path)


def test_stock_tag_is_the_pinned_commit():
    assert parity._check_stock_tag() is None
    got = subprocess.check_output(["git", "-C", str(parity.ROOT), "rev-parse", "v0.6.1^{commit}"],
                                  text=True).strip()
    assert got == parity.STOCK_COMMIT


def test_goldens_cover_every_model_and_position(prompts, goldens):
    assert goldens["format"] == 1
    assert goldens["stock"]["tag"] == parity.STOCK_TAG
    assert goldens["stock"]["commit"] == parity.STOCK_COMMIT
    assert goldens["stock"]["prompts_sha256"] == parity._prompts_sha256(prompts)
    assert goldens["stock"]["source_sha256"] == prompts["source_sha256"]
    assert sorted(goldens["models"]) == list(parity.MODELS)
    for model in parity.MODELS:
        row = goldens["models"][model]
        assert len(row["manifest_sha256"]) == 64
        assert sorted(row["positions"]) == sorted(str(p) for p in parity.POSITIONS)
        for pos in parity.POSITIONS:
            out = row["positions"][str(pos)]
            assert len(out["ids"]) == parity.GENERATED_TOKENS
            assert all(isinstance(token, int) for token in out["ids"])
            assert isinstance(out["text"], str)


def test_exact_comparison_passes(goldens):
    result = {"models": {
        model: {"positions": goldens["models"][model]["positions"]}
        for model in parity.MODELS
    }}
    rows, mismatches = parity._compare(goldens, result, "fork")
    assert not mismatches
    assert len(rows) == len(parity.MODELS) * len(parity.POSITIONS)
    assert {row[3] for row in rows} == {"exact"}


def test_comparison_can_scope_to_one_model(goldens):
    result = {"models": {"2b": {"positions": goldens["models"]["2b"]["positions"]}}}
    rows, mismatches = parity._compare(goldens, result, "fork", models=("2b",))
    assert not mismatches
    assert len(rows) == len(parity.POSITIONS)
    assert {row[0] for row in rows} == {"2b"}


def test_comparison_can_scope_to_positions(goldens):
    positions = (100, 2000)
    result = {"models": {
        model: {"positions": {str(pos): goldens["models"][model]["positions"][str(pos)]
                               for pos in positions}}
        for model in parity.MODELS
    }}
    rows, mismatches = parity._compare(goldens, result, "fork", positions=positions)
    assert not mismatches
    assert len(rows) == len(parity.MODELS) * len(positions)
    assert {row[1] for row in rows} == set(positions)


def _report_args(**overrides):
    values = {
        "json": None,
        "models": parity.MODELS,
        "positions": parity.POSITIONS,
        "generated_tokens": parity.GENERATED_TOKENS,
        "ctx": parity.CONTEXT,
        "prompts": str(parity.DEFAULT_PROMPTS),
        "goldens": str(parity.DEFAULT_GOLDENS),
        "pack_root": str(parity.DEFAULT_PACK_ROOT),
        "pack_2b": None,
        "pack_9b": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_canonical_check_refreshes_the_tracked_report_by_default():
    assert parity._report_path(_report_args()) == parity.DEFAULT_REPORT


def test_narrowed_or_custom_check_needs_json_to_write_a_report():
    assert parity._report_path(_report_args(models=("2b",))) is None
    assert parity._report_path(_report_args(ctx=4096)) is None
    assert parity._report_path(_report_args(prompts="other-prompts.json")) is None
    assert parity._report_path(_report_args(pack_2b="/tmp/pack")) is None
    assert parity._report_path(_report_args(models=("2b",), json="partial.json")) == Path(
        "partial.json")


def test_check_rejects_goldens_from_another_stock_tag(goldens, prompts):
    bad = json.loads(json.dumps(goldens))
    bad["stock"]["commit"] = "0" * 40
    with pytest.raises(RuntimeError, match="pinned"):
        parity._check_goldens(bad, _report_args(), prompts)


def test_check_rejects_a_missing_golden_position(goldens, prompts):
    bad = json.loads(json.dumps(goldens))
    del bad["models"]["2b"]["positions"]["2000"]
    with pytest.raises(RuntimeError, match="no 2b row at position 2000"):
        parity._check_goldens(bad, _report_args(), prompts)


def test_check_rejects_a_short_golden(goldens, prompts):
    bad = json.loads(json.dumps(goldens))
    bad["models"]["2b"]["positions"]["100"]["ids"].pop()
    with pytest.raises(RuntimeError, match="do not contain 32 tokens"):
        parity._check_goldens(bad, _report_args(), prompts)


def test_check_writes_the_default_report(monkeypatch, tmp_path):
    report = tmp_path / "parity-report.json"
    rows = [("2b", 100, "stock", "exact", None)]
    monkeypatch.setattr(parity, "DEFAULT_REPORT", report)
    monkeypatch.setattr(parity, "_stock_tree", lambda: tmp_path)
    monkeypatch.setattr(parity, "_run_check", lambda *args: (rows, []))
    assert parity.main(["--check"]) == 0
    written = _load(report)
    assert written["ok"] is True
    assert written["rows_exact"] == 1
    assert written["rows"][0]["side"] == "stock"


def test_failed_check_writes_a_failure_report(monkeypatch, tmp_path):
    report = tmp_path / "parity-report.json"

    def fail():
        raise RuntimeError("stock tag is not pinned")

    monkeypatch.setattr(parity, "DEFAULT_REPORT", report)
    monkeypatch.setattr(parity, "_stock_tree", fail)
    assert parity.main(["--check"]) == 2
    written = _load(report)
    assert written["ok"] is False
    assert written["rows_exact"] == 0
    assert written["error"] == "check failed; see stderr"


def test_comparison_names_the_first_difference(goldens):
    result = {"models": {
        model: {"positions": {
            pos: {"ids": list(value["ids"])}
            for pos, value in goldens["models"][model]["positions"].items()
        }}
        for model in parity.MODELS
    }}
    result["models"]["2b"]["positions"]["100"]["ids"][3] ^= 1
    rows, mismatches = parity._compare(goldens, result, "fork")
    assert mismatches == [("2b", 100, "fork", 3)]
    assert ("2b", 100, "fork", "MISMATCH", 3) in rows
