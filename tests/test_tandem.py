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


def test_explicit_layer_list_routes_each_layer_to_its_owner(fake_pack, fake_backends):
    eng = tandem.TandemEngine(fake_pack, ctx=64, cpu_layers="0,2", mode="cpu-first", log=lambda *a: None)
    assert eng.cpu_range == (0, 2) and eng.gpu_range == (1, 3)
    assert eng.cpu_layers == [0, 2]
    assert eng.step(5, 0) == 77
    # CPU embeds, layer 0 runs on the CPU, hidden crosses, layer 1 runs on the GPU,
    # hidden crosses back, layer 2 runs on the CPU, hidden crosses, layer 3 + head on the GPU
    assert [c for c in eng.cpu.calls if c[0] == "layer"] == [("layer", 0, 0), ("layer", 2, 0)]
    assert [c for c in eng.gpu.calls if c[0] == "layer"] == [("layer", 1, 0), ("layer", 3, 0)]
    assert eng.cpu.calls[0] == ("begin", 5)
    # every ownership switch must hand the hidden state over (3 switches for 0,2)
    switches = sum(1 for c in eng.cpu.calls + eng.gpu.calls if c[0] == "set_hidden")
    assert switches == 3


def test_placement_specs_parse_to_layer_sets():
    assert tandem._parse_layers("6") == 6
    assert tandem._parse_layers("0,1,2") == (0, 1, 2)
    assert tandem._parse_layers("0-2,4-6") == (0, 1, 2, 4, 5, 6)
    assert tandem._parse_layers("gdn:6") == "gdn:6"


def test_gdn_first_keeps_attention_layers_on_the_gpu():
    dims = {"layers": 32, "full_attention_interval": 4}
    layers = tandem.gdn_first(dims, 6)
    assert layers == (0, 1, 2, 4, 5, 6)
    assert not any((il + 1) % 4 == 0 for il in layers)      # no attention layer on the CPU
    assert len(tandem.gdn_first(dims, 23)) == 23            # every GDN layer is allowed
    with pytest.raises(SystemExit, match="between 1 and 23"):
        tandem.gdn_first(dims, 24)                          # the 24th layer would be attention


def test_pack_bytes_counts_the_engine_allocation(tmp_path):
    pack = tmp_path / "p"
    pack.mkdir()
    # two ternary tensors: bcodes = rows*ne0/4 plus one fp16 scale per 128-column group; one f32 tensor
    (pack / "bcodes_index.json").write_text(json.dumps({"tensors": {
        "token_embd.weight": {"rows": 128, "ne0": 256},
        "blk.0.attn_qkv.weight": {"rows": 256, "ne0": 128},
        "blk.1.ffn_up.weight": {"rows": 64, "ne0": 128},
        "output.weight": {"rows": 128, "ne0": 256}}}), encoding="utf-8")
    (pack / "index.json").write_text(json.dumps({"tensors": {
        "token_embd.weight": {"rows": 128, "ne0": 256},
        "blk.0.attn_qkv.weight": {"rows": 256, "ne0": 128},
        "blk.1.ffn_up.weight": {"rows": 64, "ne0": 128},
        "blk.0.ssm_norm.weight": {"nfloats": 100},
        "output.weight": {"rows": 128, "ne0": 256}}}), encoding="utf-8")
    man = {"f32_bytes": 4096, "head": "output.weight"}
    got = tandem.pack_bytes(str(pack), man)
    assert got["layers"][0] == 256 * 128 // 4 + 256 * 1 * 2 + 400        # qkv + its f32 neighbour
    assert got["layers"][1] == 64 * 128 // 4 + 64 * 1 * 2
    assert got["embed"] == 128 * 256 // 4 + 128 * 2 * 2
    assert got["head"] == got["embed"] and got["f32"] == 4096


def test_gpu_estimate_charges_kv_only_for_attention_layers_it_keeps():
    dims = {"layers": 8, "full_attention_interval": 4, "kv_heads": 2, "head_dim": 16}
    sizes = {"layers": {il: 1000 for il in range(8)}, "embed": 500, "head": 500, "f32": 100}
    kv = tandem.kv_bytes(dims, 1024)
    assert kv == 1024 * 2 * 16 * 2 * 2
    scratch = tandem.SCRATCH_MIB * 2 ** 20
    full = tandem.gpu_estimate(dims, sizes, 1024)
    assert full == scratch + 100 + 500 + 500 + 8000 + 2 * kv      # attention layers 3 and 7
    # the CPU owns the embedding once layer 0 is off the GPU
    assert tandem.gpu_estimate(dims, sizes, 1024, (0,)) == full - 1000 - 500
    # offloading an attention layer also removes its K/V
    assert tandem.gpu_estimate(dims, sizes, 1024, (7,)) == full - 1000 - kv


def test_auto_planner_prefers_gdn_layers_and_respects_the_budget():
    dims = {"layers": 32, "full_attention_interval": 4, "kv_heads": 4, "head_dim": 256}
    sizes = {"layers": {il: 56 * 2 ** 20 for il in range(32)}, "embed": 0, "head": 258 * 2 ** 20,
             "f32": 28 * 2 ** 20}
    # 32 layers * 56 MiB + 8 attention * 96 MiB K/V + head + f32 + scratch ~= 2.9 GiB
    total = tandem.gpu_estimate(dims, sizes, 24576)
    assert tandem.plan_cpu_layers(dims, sizes, 24576, total + 1024 * 2 ** 20) == ()
    cpu = tandem.plan_cpu_layers(dims, sizes, 24576, total - 200 * 2 ** 20, reserve_mib=0)
    assert cpu and all((il + 1) % 4 for il in cpu)         # only GDN layers are moved
    assert cpu == tuple(sorted(cpu))
    assert tandem.gpu_estimate(dims, sizes, 24576, cpu) <= total - 200 * 2 ** 20
    big = tandem.plan_cpu_layers(dims, sizes, 24576, total - 700 * 2 ** 20, reserve_mib=0)
    assert len(big) > len(cpu) and all((il + 1) % 4 for il in big)


def test_auto_target_frees_at_least_the_requested_headroom():
    dims = {"layers": 32, "full_attention_interval": 4, "kv_heads": 4, "head_dim": 256}
    mb = 2 ** 20
    sizes = {"layers": {il: 56 * mb for il in range(32)}, "embed": 258 * mb, "head": 258 * mb, "f32": 28 * mb}
    full = tandem.gpu_estimate(dims, sizes, 8192)
    # the model fits easily, but the target asks for 1 GiB of headroom on top
    cpu = tandem.plan_cpu_layers(dims, sizes, 8192, 64 << 30, reserve_mib=0, target_mib=1024)
    assert cpu and all((il + 1) % 4 for il in cpu)
    assert full - tandem.gpu_estimate(dims, sizes, 8192, cpu) >= 1024 * mb
    # a target of 0 is a no-op
    assert tandem.plan_cpu_layers(dims, sizes, 8192, 64 << 30, reserve_mib=0, target_mib=0) == ()


def test_auto_planner_refuses_when_nothing_can_fit():
    dims = {"layers": 8, "full_attention_interval": 4, "kv_heads": 2, "head_dim": 16}
    sizes = {"layers": {il: 1000 for il in range(8)}, "embed": 10, "head": 10, "f32": 10}
    with pytest.raises(SystemExit, match="cannot fit"):
        tandem.plan_cpu_layers(dims, sizes, 1024, 0, reserve_mib=0)


def test_auto_placement_resolves_before_the_backends_are_built(fake_pack, fake_backends, monkeypatch):
    dims = {"layers": 4, "full_attention_interval": 2, "kv_heads": 1, "head_dim": 8}
    (Path(fake_pack) / "kindi_pack.json").write_text(json.dumps({"dimensions": dims}), encoding="utf-8")
    sizes = {"layers": {il: 1 for il in range(4)}, "embed": 1, "head": 1, "f32": 1}
    monkeypatch.setattr(tandem, "pack_bytes", lambda pack, man=None: sizes)
    monkeypatch.setattr(tandem, "gpu_estimate", lambda *a, **k: 8 << 30)
    monkeypatch.setattr(tandem, "plan_cpu_layers", lambda *a, **k: (1, 2))
    eng = tandem.TandemEngine(fake_pack, ctx=64, cpu_layers="auto", mode="cpu-first", log=lambda *a: None)
    assert eng.cpu_range == (1, 2) and eng.gpu_range == (0, 3)


def test_auto_placement_refuses_when_the_model_already_fits(fake_pack, fake_backends, monkeypatch):
    monkeypatch.setattr(tandem, "pack_bytes", lambda pack, man=None: {"layers": {0: 1, 1: 1}, "embed": 1,
                                                                        "head": 1, "f32": 1})
    monkeypatch.setattr(tandem, "gpu_estimate", lambda *a, **k: 1)
    with pytest.raises(SystemExit, match="already fits"):
        tandem.TandemEngine(fake_pack, ctx=64, cpu_layers="auto", mode="cpu-first", log=lambda *a: None)


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
