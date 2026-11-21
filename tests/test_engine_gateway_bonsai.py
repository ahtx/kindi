"""The engine gateway must route a Bonsai pack through the registered adapter."""

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
            name = "bonsai"

        engine = "the-engine"

    def fake_resolve(model_id, config=None):
        seen.append((model_id, dict(config or {})))
        return _Backend()

    monkeypatch.setattr(backends_package, "resolve", fake_resolve)
    return seen


def test_the_hf_27b_manifest_resolves_through_the_bonsai_adapter(tmp_path, monkeypatch):
    """The 27B pack's GGUF name is the placeholder "Hf"; the size in `source_file` still names it."""
    seen = _recording_resolve(monkeypatch)
    logged = []
    pack = _pack(tmp_path, {"model": "Hf", "source_file": "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
                            "bcodes_bytes": 1024})
    assert engine.make_engine(pack, log=logged.append) == "the-engine"
    assert [model_id for model_id, _ in seen] == ["bonsai2-27b"]
    assert seen[0][1]["pack"] == pack
    assert any("bonsai2-27b" in line and "bonsai" in line for line in logged)


def test_a_dense_pack_with_a_recognized_name_routes_through_the_adapter(tmp_path, monkeypatch):
    seen = _recording_resolve(monkeypatch)
    pack = _pack(tmp_path, {"model": "Ternary-Bonsai-2-4B", "bcodes_bytes": 1024})
    assert engine.make_engine(pack, log=lambda line: None) == "the-engine"
    assert [model_id for model_id, _ in seen] == ["bonsai2-4b"]
