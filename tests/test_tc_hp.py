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


def test_plan_engine_refuses_9b_without_high_precision(monkeypatch):
    from kindi import spec, tp
    if spec.HP:                                        # the ambient environment already opted in
        pytest.skip("KINDI_TC_HP=1 is set in the environment")
    shard_cls = tp.Shard
    saved = (shard_cls._slices, shard_cls.build_plans, shard_cls._load,
             spec._chunk.reserve, spec._chunk.build)
    monkeypatch.setenv("KINDI_TC", "1")
    try:
        spec.install(log=lambda *a, **k: None)
        with pytest.raises(SystemExit, match=r"refuses Hadamard blocks of 4096 unless KINDI_TC_HP=1"):
            tp.Shard.build_plans(types.SimpleNamespace(HB=4096, tc_layout=False))
    finally:
        (shard_cls._slices, shard_cls.build_plans, shard_cls._load,
         spec._chunk.reserve, spec._chunk.build) = saved
        spec._installed[0] = False
