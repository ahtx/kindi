# SPDX-License-Identifier: AGPL-3.0-only
"""The config keeps the pre-adapter ``model_name`` fallback for foreign manifests.

The backend registry resolves a pack's family when an adapter claims it. Until
the Gemma 4 and Lightning adapters are present (or for a hand-edited manifest no
adapter claims), ``model_name: auto`` must keep the fallback it had before the
registry: the manifest's ``model_id``, else ``gemma4``, for those two families.
Otherwise the Bonsai size inference would rename an existing pack.
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
    # The contract alone has no family adapters. Pin the no-claim branch explicitly so this test
    # keeps its meaning once the Gemma 4 and Lightning adapters are applied on top.
    monkeypatch.setattr("kindi.modelids.adapter_model_id", lambda manifest: None)

    cfg = config.load(explicit=str(tmp_path / "kindi.conf"))

    assert cfg["model_name"] == want
    assert cfg["model_name"] != "bonsai2-9b"      # the size inference must not win here
