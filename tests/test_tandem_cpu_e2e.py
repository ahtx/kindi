# SPDX-License-Identifier: AGPL-3.0-only
"""End-to-end tandem exactness using the real CPU engine on both sides.

This is the CPU-only end-to-end check: a real Bonsai pack is loaded by the whole-model CPU
engine, then by both tandem placements with the CPU engine standing in for the GPU owner. It
exercises the real per-layer C entry points, the owned-slice load, and the hidden-state handoff
in both directions. The GPU backend is not exercised; a CUDA run stays the release gate.

Set ``KINDI_CPU_PACK`` to a Bonsai pack directory to enable it (the CPU library also needs gcc
with OpenMP). Without a pack the test skips, so the default CPU suite stays cheap.
"""
import os

import numpy as np
import pytest

from kindi import cpu, tandem

PROMPT = [1, 2, 3, 4]
REPLY = 6
CTX = 32


class _CPUOwner(cpu.Engine):
    """The CPU engine with the small surface TandemEngine expects from the GPU owner.

    A GPU owner also has ``finish``/``flush`` and the sampler's ``k``/``aout``. They are not
    used by the exactness path, but the constructor reads them, so the stand-in supplies them.
    """

    k = None
    aout = None

    def finish(self):
        return self.head()

    def flush(self):
        pass


def _greedy(eng):
    """Read the prompt and return REPLY greedy tokens, without using generate()."""
    eng.reset()
    for pos, tok in enumerate(PROMPT[:-1]):
        eng.feed(tok, pos)
    out = [eng.step(PROMPT[-1], len(PROMPT) - 1)]
    for pos in range(len(PROMPT), len(PROMPT) + REPLY - 1):
        out.append(eng.step(out[-1], pos))
    return out


@pytest.mark.skipif(not os.environ.get("KINDI_CPU_PACK"),
                    reason="set KINDI_CPU_PACK to a Bonsai pack to run the real CPU tandem check")
def test_both_placements_match_the_whole_model_cpu_engine(monkeypatch):
    pack = os.environ["KINDI_CPU_PACK"]
    whole = cpu.Engine(pack, ctx=CTX, log=None, threads=2)
    want = _greedy(whole)
    want_hidden = whole.get_hidden()

    # Both owners are the CPU engine: this checks the placement and handoff, not a card. Pin the
    # owner kind so no CUDA probe runs when the test is enabled on a GPU host.
    monkeypatch.setattr(tandem, "GPUEngine", _CPUOwner)
    monkeypatch.setenv("KINDI_TANDEM_GPU", "plain")

    for mode in tandem.MODES:
        eng = tandem.TandemEngine(pack, ctx=CTX, cpu_layers=1, mode=mode, log=None)
        # the slices are the point: only the owned blocks were loaded into each engine
        expected_cpu = (0,) if mode == "cpu-first" else (eng.NL - 1,)
        assert eng.cpu.layers == expected_cpu
        assert eng.gpu.layers == tuple(il for il in range(eng.NL) if il not in expected_cpu)

        got = _greedy(eng)
        assert got == want, "%s greedy reply diverged from the whole-model CPU engine" % mode
        head_owner = eng.gpu if eng._gpu_head else eng.cpu
        np.testing.assert_array_equal(head_owner.get_hidden(), want_hidden)
