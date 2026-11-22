# SPDX-License-Identifier: AGPL-3.0-only
"""The Hadamard-block-4096 (9B) tensor-core contract, without loading a pack.

A 9B pack folds with Hadamard block 4096. The tensor-core verify is token-exact only with the
high-precision activations, so make_engine turns them on for such a pack before kindi.spec is imported;
kindi.spec refuses the pack when an explicit KINDI_TC_HP=0 asks for the measured-worse setting.
"""
import types

import pytest


def test_hb_4096_defaults_to_high_precision(monkeypatch):
    from kindi import engine as _eng
    monkeypatch.setenv("KINDI_TC", "1")
    monkeypatch.delenv("KINDI_TC_HP", raising=False)
    assert _eng.tc_hp_for({"hadamard_block": 4096}) == "1"
    assert _eng.tc_hp_for({"hadamard_block": "4096"}) == "1"
    assert _eng.tc_hp_for({"hadamard_block": 1024}) is None
    assert _eng.tc_hp_for({}) is None                 # a pack without the key folds 512
    monkeypatch.setenv("KINDI_TC_HP", "0")            # an explicit choice always wins (it then refuses)
    assert _eng.tc_hp_for({"hadamard_block": 4096}) is None


def test_hp_default_is_only_for_the_tensor_core_path(monkeypatch):
    from kindi import engine as _eng
    monkeypatch.delenv("KINDI_TC_HP", raising=False)
    monkeypatch.delenv("KINDI_TC", raising=False)
    assert _eng.tc_hp_for({"hadamard_block": 4096}) is None


def _save_hooks(spec, tp):
    return (tp.Shard._slices, tp.Shard.build_plans, tp.Shard._load,
            spec._chunk.reserve, spec._chunk.build)


def _restore_hooks(spec, tp, saved):
    (tp.Shard._slices, tp.Shard.build_plans, tp.Shard._load,
     spec._chunk.reserve, spec._chunk.build) = saved
    spec._installed[0] = False


def test_plan_engine_refuses_9b_without_high_precision(monkeypatch):
    from kindi import spec, tp
    shard_cls = tp.Shard
    saved = _save_hooks(spec, tp)
    monkeypatch.setattr(spec, "HP", False)
    monkeypatch.setenv("KINDI_TC", "1")
    try:
        spec.install(log=lambda *a, **k: None)
        with pytest.raises(SystemExit, match=r"refuses Hadamard blocks of 4096 unless KINDI_TC_HP=1"):
            tp.Shard.build_plans(types.SimpleNamespace(HB=4096, tc_layout=True))
    finally:
        _restore_hooks(spec, tp, saved)


def test_plan_engine_uses_plain_plans_when_the_layout_is_declined(monkeypatch):
    """A repack that declines the tensor-core layout leaves the HB-4096 refusal unsaid."""
    from kindi import spec, tp
    shard_cls = tp.Shard
    saved = _save_hooks(spec, tp)
    calls = []

    def plain(self):
        calls.append(self)

    monkeypatch.setattr(spec, "HP", False)
    monkeypatch.setenv("KINDI_TC", "1")
    monkeypatch.setattr(shard_cls, "build_plans", plain)
    try:
        spec._installed[0] = False
        spec.install(log=lambda *a, **k: None)
        shard = types.SimpleNamespace(HB=4096, tc_layout=False)
        tp.Shard.build_plans(shard)                 # must not refuse: the plain plans run
        assert calls == [shard]
    finally:
        _restore_hooks(spec, tp, saved)


def test_load_refuses_only_after_repack_confirms_the_tensor_core_layout(monkeypatch):
    """The HB-4096 refusal is on the real TC path, after the repack proves the layout."""
    from kindi import spec, tp
    shard_cls = tp.Shard
    saved = _save_hooks(spec, tp)
    calls = []

    def fallback(self):
        calls.append("fallback")

    def confirmed(self, log):
        calls.append("repack")
        self.tc_layout, self.tc_lane = True, False

    monkeypatch.setattr(spec, "HP", False)
    monkeypatch.setenv("KINDI_TC", "1")
    monkeypatch.setattr(shard_cls, "_load", fallback)
    monkeypatch.setattr(spec, "_tc_backend", lambda card: object())
    monkeypatch.setattr(spec, "load_stream", lambda shard: False)
    monkeypatch.setattr(spec, "SpecKernels", lambda backend: object())
    monkeypatch.setattr(spec, "repack", confirmed)
    monkeypatch.setattr(spec._tc, "pascal", lambda card: False)
    try:
        spec._installed[0] = False
        spec.install(log=lambda *a, **k: None)
        shard = types.SimpleNamespace(card=0, tp=types.SimpleNamespace(W=1), HB=4096,
                                      g5l=True, log=lambda *a, **k: None)
        with pytest.raises(SystemExit, match=r"refuses Hadamard blocks of 4096 unless KINDI_TC_HP=1"):
            tp.Shard._load(shard)
        assert calls == ["fallback", "repack"]
    finally:
        _restore_hooks(spec, tp, saved)


def test_load_keeps_the_plain_fallback_when_the_layout_is_declined(monkeypatch):
    """Backend present, HP off, repack declines: load the plain path instead of stopping."""
    from kindi import spec, tp
    shard_cls = tp.Shard
    saved = _save_hooks(spec, tp)
    calls = []

    def fallback(self):
        calls.append("fallback")

    def declined(self, log):
        calls.append("repack")

    monkeypatch.setattr(spec, "HP", False)
    monkeypatch.setenv("KINDI_TC", "1")
    monkeypatch.setattr(shard_cls, "_load", fallback)
    monkeypatch.setattr(spec, "_tc_backend", lambda card: object())
    monkeypatch.setattr(spec, "load_stream", lambda shard: False)
    monkeypatch.setattr(spec, "SpecKernels", lambda backend: object())
    monkeypatch.setattr(spec, "repack", declined)
    monkeypatch.setattr(spec._tc, "pascal", lambda card: False)
    try:
        spec._installed[0] = False
        spec.install(log=lambda *a, **k: None)
        shard = types.SimpleNamespace(card=0, tp=types.SimpleNamespace(W=1), HB=4096,
                                      g5l=True, log=lambda *a, **k: None)
        tp.Shard._load(shard)                       # must not raise: no tensor-core layout
        assert calls == ["fallback", "repack"]
        assert not getattr(shard, "tc_layout", False)
    finally:
        _restore_hooks(spec, tp, saved)
