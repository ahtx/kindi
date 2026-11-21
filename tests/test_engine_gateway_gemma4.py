"""The engine gateway must route a Gemma 4 pack through the registered adapter."""

import json

import pytest

import kindi.backends as backends_package
from kindi import engine


def _pack(tmp_path, man):
    (tmp_path / "kindi_pack.json").write_text(json.dumps(man), encoding="utf-8")
    return str(tmp_path)


def _recording_resolve(monkeypatch):
    seen = []

    class _Backend(object):
        class capabilities(object):
            name = "gemma4"

        engine = "the-engine"

    def fake_resolve(model_id, config=None):
        seen.append((model_id, dict(config or {})))
        return _Backend()

    monkeypatch.setattr(backends_package, "resolve", fake_resolve)
    return seen


def test_the_gemma4_family_routes_through_its_adapter(tmp_path, monkeypatch):
    seen = _recording_resolve(monkeypatch)
    pack = _pack(tmp_path, {"backend": "kindi_gemma4", "model_path": "g.gguf"})
    assert engine.make_engine(pack, log=lambda line: None) == "the-engine"
    assert [model_id for model_id, _ in seen] == ["gemma4-e4b"]


def test_the_legacy_valve_refuses_the_gemma4_family(tmp_path, monkeypatch):
    monkeypatch.setenv("KINDI_LEGACY_ENGINE_PATH", "1")
    monkeypatch.setattr(engine, "make_engine_legacy",
                        lambda *a, **k: pytest.fail("the valve built the dense engine"))
    pack = _pack(tmp_path, {"backend": "kindi_gemma4", "model_path": "g.gguf"})
    with pytest.raises(SystemExit, match="KINDI_LEGACY_ENGINE_PATH=1"):
        engine.make_engine(pack)
