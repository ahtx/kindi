"""The T4.2 gateway must not build the wrong engine silently.

`kindi.engine.make_engine` is the one place a pack turns into an engine, so every
pack either resolves through its backend adapter or says why it did not.
"""
import json

import pytest

import kindi.backends as backends_package
from kindi import engine


def _pack(tmp_path, man):
    (tmp_path / "kindi_pack.json").write_text(json.dumps(man), encoding="utf-8")
    return str(tmp_path)


def _recording_resolve(monkeypatch):
    """Replace the registry lookup: records (model id, config) and returns an engine-shaped backend."""
    seen = []

    class _Backend(object):
        class capabilities(object):
            name = "recording"

        engine = "the-engine"

    def fake_resolve(model_id, config=None):
        seen.append((model_id, dict(config or {})))
        return _Backend()

    # engine.make_engine imports it as `from .backends import resolve`, i.e. the package attribute
    monkeypatch.setattr(backends_package, "resolve", fake_resolve)
    return seen


def test_a_dense_pack_no_adapter_claims_says_so_and_uses_the_legacy_builder(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "make_engine_legacy", lambda *a, **k: "legacy-engine")
    logged = []
    pack = _pack(tmp_path, {"model": "Some-Other-Bonsai", "bcodes_bytes": 1024})
    assert engine.make_engine(pack, log=logged.append) == "legacy-engine"
    assert any("no adapter claims" in line and "legacy dense builder" in line for line in logged)


def test_a_pack_whose_family_has_no_adapter_is_refused_not_built_dense(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "make_engine_legacy",
                        lambda *a, **k: pytest.fail("the gateway built the dense engine"))
    pack = _pack(tmp_path, {"backend": "kindi_cascade", "model_path": "x.gguf"})
    with pytest.raises(SystemExit, match="kindi_cascade"):
        engine.make_engine(pack)


def test_an_adapted_pack_resolves_through_the_registry(tmp_path, monkeypatch):
    seen = _recording_resolve(monkeypatch)
    monkeypatch.setattr(engine, "adapter_model_id", lambda man: "recording-1b")
    logged = []
    pack = _pack(tmp_path, {"backend": "kindi_recording"})
    assert engine.make_engine(pack, log=logged.append) == "the-engine"
    assert [model_id for model_id, _ in seen] == ["recording-1b"]
    assert seen[0][1]["pack"] == pack
    assert any("recording-1b" in line and "recording" in line for line in logged)


def test_a_manifest_with_neither_family_nor_dense_weights_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "make_engine_legacy",
                        lambda *a, **k: pytest.fail("the gateway built the dense engine"))
    pack = _pack(tmp_path, {"model": "Hf"})
    with pytest.raises(SystemExit, match="not a pack this version can build"):
        engine.make_engine(pack)


def test_the_legacy_valve_refuses_a_family_it_cannot_build(tmp_path, monkeypatch):
    monkeypatch.setenv("KINDI_LEGACY_ENGINE_PATH", "1")
    monkeypatch.setattr(engine, "make_engine_legacy",
                        lambda *a, **k: pytest.fail("the valve built the dense engine"))
    pack = _pack(tmp_path, {"backend": "kindi_gemma4", "model_path": "g.gguf"})
    with pytest.raises(SystemExit, match="KINDI_LEGACY_ENGINE_PATH=1"):
        engine.make_engine(pack)


def test_the_legacy_valve_names_itself_in_the_log(tmp_path, monkeypatch):
    monkeypatch.setenv("KINDI_LEGACY_ENGINE_PATH", "1")
    monkeypatch.setattr(engine, "make_engine_legacy", lambda *a, **k: "legacy-engine")
    logged = []
    pack = _pack(tmp_path, {"model": "Hf", "bcodes_bytes": 1024})
    assert engine.make_engine(pack, log=logged.append) == "legacy-engine"
    assert any("KINDI_LEGACY_ENGINE_PATH=1" in line for line in logged)
