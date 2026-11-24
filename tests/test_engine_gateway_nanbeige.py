"""The engine gateway must route a Nanbeige pack through the registered adapter."""

import json

import kindi.backends as backends_package
from kindi import engine


def _pack(tmp_path, man):
    (tmp_path / "kindi_pack.json").write_text(json.dumps(man), encoding="utf-8")
    return str(tmp_path)


def _recording_resolve(monkeypatch):
    seen = []

    class _Backend(object):
        class capabilities(object):
            name = "nanbeige"

        engine = "the-engine"

    def fake_resolve(model_id, config=None):
        seen.append((model_id, dict(config or {})))
        return _Backend()

    monkeypatch.setattr(backends_package, "resolve", fake_resolve)
    return seen


def test_the_nanbeige_family_routes_through_its_adapter(tmp_path, monkeypatch):
    seen = _recording_resolve(monkeypatch)
    pack = _pack(tmp_path, {"backend": "kindi_nanbeige", "model_path": "n.gguf"})
    assert engine.make_engine(pack, log=lambda line: None) == "the-engine"
    assert [model_id for model_id, _ in seen] == ["nanbeige-33b"]
