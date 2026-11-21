# SPDX-License-Identifier: AGPL-3.0-only
"""The dense tandem placement and handoff, without a real GPU or CPU pack.

The fake backends record the layer ranges and the hidden-state exchange.  The real GPU path is
covered by the code-level checks in the release gate; this test suite keeps the routing contract
(which backend owns the embedding, head and each layer) and the two placement modes honest.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from kindi import tandem
from kindi.engine import Engine


class FakeEngine(object):
    D = 4

    def __init__(self, pack, ctx=4096, log=None, layers=None, embed=True, head=True):
        self.ctx = int(ctx)
        self.layers = None if layers is None else tuple(layers)
        self.embed_owned, self.head_owned = bool(embed), bool(head)
        self.calls = []
        self.h = np.zeros(self.D, dtype=np.float32)
        self.pending = 0
        self._kvi = {}
        self.vocab = {"tok": 0}
        self.ylog = np.arange(5, dtype=np.float32)
        self.ylog_n, self.ylog_map = 5, None
        self.k = object()
        self.aout = np.zeros(1, dtype=np.int32)

    def begin(self, tok):
        self.calls.append(("begin", int(tok)))
        self.h = np.full(self.D, float(tok), dtype=np.float32)

    def layer(self, il, pos):
        self.calls.append(("layer", int(il), int(pos)))
        self.h += 1.0

    def feed(self, tok, pos):
        self.calls.append(("feed", int(tok), int(pos)))

    def step(self, tok, pos, **kw):
        self.calls.append(("step", int(tok), int(pos)))
        return 42

    def finish(self):
        self.calls.append(("finish",))
        self.ylog = np.arange(5, dtype=np.float32) + 100
        return 77

    def head(self):
        self.calls.append(("head",))
        self.ylog = np.arange(5, dtype=np.float32) + 200
        return 88

    def get_hidden(self):
        self.calls.append(("get_hidden",))
        return self.h.copy()

    def set_hidden(self, values):
        self.calls.append(("set_hidden", list(values)))
        self.h = np.asarray(values, dtype=np.float32).copy()
        self.pending = 0

    def flush(self):
        self.calls.append(("flush",))
        self.pending = 0

    def reset(self):
        self.calls.append(("reset",))
        self.h.fill(0)
        self.pending = 0


@pytest.fixture
def fake_pack(tmp_path):
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "kindi_pack.json").write_text(json.dumps({"dimensions": {"layers": 4}}), encoding="utf-8")
    return str(pack)


class FakeCPUEngine(FakeEngine):
    """The CPU backend keeps compact per-owner state lists (it only loads its layers)."""

    def __init__(self, pack, ctx=4096, log=None, layers=None, embed=True, head=True):
        super().__init__(pack, ctx=ctx, log=log, layers=layers, embed=embed, head=head)
        if self.layers is not None:
            self.M, self.conv, self.Kc, self.Vc = [], [], [], []
            for il in self.layers:
                if il in self._kvi:
                    self.Kc.append(np.zeros((self.ctx, 1, 1), dtype=np.float16))
                    self.Vc.append(np.zeros((self.ctx, 1, 1), dtype=np.float16))
                else:
                    self.M.append(np.zeros(1, dtype=np.float32))
                    self.conv.append(np.zeros(1, dtype=np.float32))


class FakeGPUEngine(FakeEngine):
    """The GPU backend keeps global layer-indexed state (it loads a slice by layer id)."""


@pytest.fixture
def fake_backends(monkeypatch):
    monkeypatch.setattr(tandem, "CPUEngine", FakeCPUEngine)
    monkeypatch.setattr(tandem, "GPUEngine", FakeGPUEngine)


def test_cpu_first_owns_the_prefix_and_the_gpu_head(fake_pack, fake_backends):
    eng = tandem.TandemEngine(fake_pack, ctx=64, cpu_layers=2, mode="cpu-first", log=lambda *a: None)
    assert eng.cpu.layers == (0, 1)          # the CPU engine loads only its owned slice
    assert eng.cpu.embed_owned is True and eng.cpu.head_owned is False
    assert eng.gpu.layers == (2, 3)
    assert eng.gpu.embed_owned is False
    assert eng.gpu.head_owned is True
    assert eng.cpu_range == (0, 1) and eng.gpu_range == (2, 3)
    assert eng.ylog is eng.gpu.ylog

    assert eng.step(10, 0) == 77
    assert eng.cpu.calls == [("begin", 10), ("layer", 0, 0), ("layer", 1, 0), ("get_hidden",)]
    assert eng.gpu.calls[:4] == [("set_hidden", [12.0, 12.0, 12.0, 12.0]),
                                  ("layer", 2, 0), ("layer", 3, 0), ("finish",)]
    assert eng.ylog is eng.gpu.ylog


def test_gpu_first_owns_the_prefix_and_the_cpu_head(fake_pack, fake_backends):
    eng = tandem.TandemEngine(fake_pack, ctx=64, cpu_layers=2, mode="gpu-first", log=lambda *a: None)
    assert eng.cpu.layers == (2, 3)          # the CPU engine loads only its owned slice
    assert eng.cpu.embed_owned is False and eng.cpu.head_owned is True
    assert eng.gpu.layers == (0, 1)
    assert eng.gpu.embed_owned is True
    assert eng.gpu.head_owned is False
    assert eng.cpu_range == (2, 3) and eng.gpu_range == (0, 1)
    assert eng.ylog is eng.cpu.ylog

    assert eng.step(9, 0) == 88
    assert eng.gpu.calls[:3] == [("begin", 9), ("layer", 0, 0), ("layer", 1, 0)]
    assert eng.gpu.calls[3] == ("get_hidden",)
    assert eng.cpu.calls[:3] == [("set_hidden", [11.0, 11.0, 11.0, 11.0]),
                                 ("layer", 2, 0), ("layer", 3, 0)]
    assert eng.cpu.calls[-1] == ("head",)
    assert eng.ylog is eng.cpu.ylog


def test_feed_leaves_the_residual_without_evaluating_the_head(fake_pack, fake_backends):
    eng = tandem.TandemEngine(fake_pack, ctx=64, cpu_layers=1, mode="cpu-first", log=lambda *a: None)
    eng.feed(3, 7)
    assert ("finish",) not in eng.gpu.calls
    assert eng.gpu.calls[-1] == ("flush",)


def test_reset_clears_both_backends(fake_pack, fake_backends):
    eng = tandem.TandemEngine(fake_pack, ctx=64, cpu_layers=1, mode="gpu-first", log=lambda *a: None)
    eng.reset()
    assert ("reset",) in eng.cpu.calls and ("reset",) in eng.gpu.calls


def test_invalid_split_is_rejected_before_a_backend_is_built(fake_pack, fake_backends):
    with pytest.raises(SystemExit, match="between 1 and 3"):
        tandem.TandemEngine(fake_pack, cpu_layers=0, mode="cpu-first", log=lambda *a: None)
    with pytest.raises(SystemExit, match="cpu-first or gpu-first"):
        tandem.TandemEngine(fake_pack, cpu_layers=1, mode="middle", log=lambda *a: None)


def test_runner_reads_the_prompt_and_steps_the_last_token(fake_pack, fake_backends):
    eng = tandem.TandemEngine(fake_pack, ctx=64, cpu_layers=1, mode="cpu-first", log=lambda *a: None)
    runner = tandem.TandemRunner(eng, log=lambda *a: None)
    nxt, cached = runner.prepare([1, 2, 3], 3)
    assert nxt == 77 and cached == 0
    assert runner.tokens == [1, 2, 3]
    assert ("get_hidden",) in eng.cpu.calls
    assert runner.step(4) == 77
    assert runner.tokens == [1, 2, 3, 4]


def test_layer_tensor_selection_keeps_only_owned_layers_and_optional_ends():
    bi = {
        "token_embd.weight": {"rows": 1},
        "output.weight": {"rows": 2},
        "blk.0.attn_q.weight": {"rows": 3},
        "blk.1.ffn_up.weight": {"rows": 4},
        "blk.2.attn_q.weight": {"rows": 5},
    }
    got = Engine._select_layers(bi, (1, 2), embed=False, head=True, head_name="output.weight")
    assert set(got) == {"blk.1.ffn_up.weight", "blk.2.attn_q.weight", "output.weight"}
    got = Engine._select_layers(bi, (0,), embed=True, head=False, head_name="output.weight")
    assert set(got) == {"token_embd.weight", "blk.0.attn_q.weight"}
