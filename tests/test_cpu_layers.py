# SPDX-License-Identifier: AGPL-3.0-only
"""The CPU engine's per-layer entry point for a future CPU+GPU split.

The fake-library tests need no model pack, C compiler or GPU.  The exactness test is opt-in with
KINDI_CPU_PACK because it loads a real Bonsai pack and runs the CPU engine; set it to the pack
directory to compare the per-layer sequence with the existing whole-model step().
"""
import ctypes
import os

import numpy as np
import pytest

from kindi import cpu


class _FakeLib(object):
    def __init__(self):
        self.calls = []

    def kc_embed(self, m, tok):
        self.calls.append(("embed", tok))

    def kc_layer(self, m, il, pos):
        self.calls.append(("layer", il, pos))
        return 0

    def kc_head(self, m):
        self.calls.append(("head",))
        return 7


def _fake_engine():
    eng = object.__new__(cpu.Engine)
    eng.m = cpu.Model()
    eng.NL = 3
    eng.ctx = 4
    eng.lib = _FakeLib()
    return eng


def test_layer_entry_point_rejects_bad_indices_before_c():
    eng = _fake_engine()
    with pytest.raises(IndexError):
        eng.layer(-1, 0)
    with pytest.raises(IndexError):
        eng.layer(3, 0)
    with pytest.raises(OverflowError):
        eng.layer(0, 4)
    assert eng.lib.calls == []


def test_layer_entry_point_calls_the_tandem_order():
    eng = _fake_engine()
    eng.feed_layers(5, 2)
    assert eng.lib.calls == [("embed", 5), ("layer", 0, 2), ("layer", 1, 2), ("layer", 2, 2)]
    eng.lib.calls[:] = []
    assert eng.step_layers(6, 3) == 7
    assert eng.lib.calls == [("embed", 6), ("layer", 0, 3), ("layer", 1, 3), ("layer", 2, 3),
                             ("head",)]


def test_layer_entry_point_checks_position_before_embedding():
    eng = _fake_engine()
    with pytest.raises(OverflowError):
        eng.feed_layers(1, 4)
    assert eng.lib.calls == []


def test_hidden_handoff_round_trips():
    eng = _fake_engine()
    eng.D = 2
    eng.hidden = np.zeros(2, dtype=np.float32)
    eng.set_hidden([1.0, 2.0])
    np.testing.assert_array_equal(eng.get_hidden(), [1.0, 2.0])
    eng.hidden[0] = 3.0
    np.testing.assert_array_equal(eng.get_hidden(), [3.0, 2.0])
    with pytest.raises(ValueError):
        eng.set_hidden([1.0])


@pytest.mark.skipif(not os.environ.get("KINDI_CPU_PACK"),
                    reason="set KINDI_CPU_PACK to a Bonsai pack to run the real exactness check")
def test_layerwise_matches_whole_model_step():
    eng = cpu.Engine(os.environ["KINDI_CPU_PACK"], ctx=8, log=None)
    ids = [1, 2, 3, 4, 5]

    eng.reset()
    want = [eng.step(tok, pos) for pos, tok in enumerate(ids)]
    want_hidden = eng.get_hidden()

    eng.reset()
    got = [eng.step_layers(tok, pos) for pos, tok in enumerate(ids)]
    got_hidden = eng.get_hidden()

    assert got == want
    np.testing.assert_array_equal(got_hidden, want_hidden)
