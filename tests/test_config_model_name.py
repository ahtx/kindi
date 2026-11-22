# SPDX-License-Identifier: AGPL-3.0-only
"""Pin both ``model_name: auto`` paths for real Gemma 4 and Lightning manifests.

The claimed path resolves through the registered family adapter; the no-claim
path keeps the pre-adapter fallback for a hand-edited manifest no adapter
claims. In both, the manifest's ``model_id`` (else ``gemma4``) wins over the
Bonsai size inference, so merging an adapter cannot rename an existing pack.
"""
import json

import pytest

from kindi import config


@pytest.mark.parametrize("backend, model_id, want", [
    ("kindi_gemma4", "gemma4-e4b", "gemma4-e4b"),
    ("kindi_gemma4", None, "gemma4"),
    ("kindi_lightning", "lightning-30b", "lightning-30b"),
    ("kindi_lightning", None, "gemma4"),          # the documented legacy fallback
])
def test_unclaimed_gemma4_and_lightning_manifests_keep_the_legacy_name(
        tmp_path, monkeypatch, backend, model_id, want):
    man = {"backend": backend, "model": "Ternary-Bonsai-2-9B"}
    if model_id is not None:
        man["model_id"] = model_id
    (tmp_path / "kindi_pack.json").write_text(json.dumps(man), encoding="utf-8")
    (tmp_path / "kindi.conf").write_text("[kindi]\n", encoding="utf-8")
    monkeypatch.setenv("KINDI_PACK", str(tmp_path))
    monkeypatch.setenv("KINDI_MODEL_NAME", "auto")
    # Force the no-claim branch explicitly: a hand-edited manifest that no adapter
    # claims must keep the pre-registry fallback name.
    monkeypatch.setattr("kindi.modelids.adapter_model_id", lambda manifest: None)

    cfg = config.load(explicit=str(tmp_path / "kindi.conf"))

    assert cfg["model_name"] == want
    assert cfg["model_name"] != "bonsai2-9b"      # the size inference must not win here

# The real manifest fields that decide adapter resolution and `model_name`:
# `kindi/convert.py` writes `backend: kindi_gemma4` and `model_id: gemma4`;
# `kindi/lightning_cli.py` writes `backend: kindi_lightning` and
# `model_id: nemotron-lightning-30b`. The full manifests also carry paths,
# hashes and dimensions, none of which either decision reads.
REAL_GEMMA4 = {"backend": "kindi_gemma4", "model_id": "gemma4"}
REAL_LIGHTNING = {"backend": "kindi_lightning", "model_id": "nemotron-lightning-30b"}

CASES = [
    (REAL_GEMMA4, None, None, "gemma4"),
    (REAL_LIGHTNING, "nemotron-lightning-30b", "lightning", "nemotron-lightning-30b"),
]


@pytest.mark.parametrize("manifest, adapter_id, adapter_name, model_name", CASES)
def test_real_manifests_pin_the_claimed_and_no_claim_paths(
        tmp_path, monkeypatch, manifest, adapter_id, adapter_name, model_name):
    """The live adapter claims its real manifest; the other family stays unclaimed.

    The claimed row also drives `make_engine` through the adapter's manifest hook
    and a recording registry. The no-claim row proves the other real family is
    not claimed and the gateway refuses it rather than building a dense engine.
    In both, the manifest's own `model_id` is the served `model_name`.
    """
    import kindi.backends as backends_package
    from kindi import engine
    from kindi.modelids import adapter_model_id

    assert adapter_model_id(manifest) == adapter_id
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "kindi_pack.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "kindi.conf").write_text("[kindi]" + chr(10), encoding="utf-8")
    monkeypatch.setenv("KINDI_PACK", str(pack))
    monkeypatch.setenv("KINDI_MODEL_NAME", "auto")
    monkeypatch.delenv("KINDI_LEGACY_ENGINE_PATH", raising=False)

    cfg = config.load(explicit=str(tmp_path / "kindi.conf"))
    assert cfg["model_name"] == model_name
    assert cfg["model_name"] != "bonsai2-2b"

    if adapter_id is None:
        with pytest.raises(SystemExit, match=manifest["backend"]):
            engine.make_engine(str(pack), log=lambda line: None)
        return

    class _Backend(object):
        class capabilities(object):
            pass

        engine = "the-engine"

    _Backend.capabilities.name = adapter_name
    seen = []

    def fake_resolve(model_id, config=None):
        seen.append(model_id)
        return _Backend()

    monkeypatch.setattr(backends_package, "resolve", fake_resolve)
    logged = []
    assert engine.make_engine(str(pack), log=logged.append) == "the-engine"
    assert seen == [adapter_id]
    assert any(adapter_id in line and adapter_name in line for line in logged)
