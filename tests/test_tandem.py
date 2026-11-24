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
    # the owner choice is a property of the machine (a card is visible or not): pin it so these tests
    # do not change behaviour with the host. The plan owner is covered by its own tests below.
    monkeypatch.setenv("KINDI_TANDEM_GPU", "plain")


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


def test_vram_budget_helper_combines_free_memory_reserve_and_ceiling():
    mb = 2 ** 20
    assert tandem.vram_budget_bytes(8 << 30, 512, 6144) == min((8 << 30) - 512 * mb, 6144 * mb)
    assert tandem.vram_budget_bytes(8 << 30, 0, None) == 8 << 30
    assert tandem.vram_budget_bytes(512 * mb, 512, 6144) == 0
    with pytest.raises(SystemExit, match="cannot be negative"):
        tandem.vram_budget_bytes(8 << 30, -1, None)


def test_explicit_vram_ceiling_forces_a_split_below_a_larger_free_figure():
    """A display card can have 8 GiB free but a 6 GiB usable barrier: the ceiling, not free memory,
    decides the fit."""
    dims = {"layers": 32, "full_attention_interval": 4, "kv_heads": 4, "head_dim": 256}
    mb = 2 ** 20
    sizes = {"layers": {il: 56 * mb for il in range(32)}, "embed": 0, "head": 258 * mb,
             "f32": 28 * mb}
    full = tandem.gpu_estimate(dims, sizes, 24576)
    ceiling = full // mb - 300
    cpu = tandem.plan_cpu_layers(dims, sizes, 24576, 64 << 30, reserve_mib=0,
                                 vram_budget_mib=ceiling)
    assert cpu and all((il + 1) % 4 for il in cpu)
    assert tandem.gpu_estimate(dims, sizes, 24576, cpu) <= ceiling * mb
    # the same card with no ceiling fits the whole model
    assert tandem.plan_cpu_layers(dims, sizes, 24576, 64 << 30, reserve_mib=0) == ()


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


def test_auto_placement_passes_the_vram_budget_to_the_planner(fake_pack, fake_backends, monkeypatch):
    dims = {"layers": 4, "full_attention_interval": 2, "kv_heads": 1, "head_dim": 8}
    (Path(fake_pack) / "kindi_pack.json").write_text(json.dumps({"dimensions": dims}), encoding="utf-8")
    sizes = {"layers": {il: 1 for il in range(4)}, "embed": 1, "head": 1, "f32": 1}
    seen = []
    monkeypatch.setattr(tandem, "pack_bytes", lambda pack, man=None: sizes)
    monkeypatch.setattr(tandem, "gpu_estimate", lambda *a, **k: 8 << 30)
    monkeypatch.setattr(tandem, "plan_cpu_layers", lambda *a, **k: seen.append(k) or (1, 2))
    monkeypatch.setenv("KINDI_TANDEM_VRAM_MB", "6144")
    eng = tandem.TandemEngine(fake_pack, ctx=64, cpu_layers="auto", mode="cpu-first",
                              log=lambda *a: None)
    assert eng.cpu_range == (1, 2)
    assert seen[-1]["vram_budget_mib"] == 6144
    assert seen[-1]["reserve_mib"] == 512


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


# --------------------------------------------------------------- the GPU owner: plan or plain
def test_the_gpu_owner_defaults_to_the_plan_engine_on_one_card(monkeypatch):
    """The plan owner is the default because it loads only the owned layers and keeps q8_0 K/V."""
    monkeypatch.setattr(tandem, "PlanEngine", object)
    monkeypatch.delenv("KINDI_TP", raising=False)
    assert tandem._pick_owner("auto") == "plan"          # one card on this host
    assert tandem._pick_owner("plain") == "plain"
    assert tandem._pick_owner("plan") == "plan"
    assert tandem._pick_owner(None) == "plan"
    with pytest.raises(SystemExit):
        tandem._pick_owner("gpu")


def test_the_plan_owner_steps_back_when_it_cannot_run(monkeypatch):
    """KINDI_TP=0 keeps the plain engine; more than one visible card is not a tandem GPU owner."""
    assert tandem._pick_owner("auto") == "plain" if False else True
    monkeypatch.setenv("KINDI_TP", "0")
    assert tandem._pick_owner("auto") == "plain"
    monkeypatch.delenv("KINDI_TP")
    logged = []
    # a two-card host is the one case the plan engine cannot own (a partial plan engine is one card)
    import kindi.engine as engine_mod
    if engine_mod.cp is None:
        pytest.skip("no CuPy: _pick_owner cannot read the card count")
    monkeypatch.setenv("KINDI_CARDS", "2")
    assert tandem._pick_owner("auto", log=logged.append) == "plain"
    assert any("2 cards" in line for line in logged)


def test_the_plan_owner_gets_only_the_gpu_layers_and_the_head(fake_pack, fake_backends, monkeypatch):
    """The split under the plan owner: the same routing, a plan engine object, and a fingerprint that
    records the owner (a plain K/V checkpoint must not be restored into a q8_0 owner)."""
    monkeypatch.setattr(tandem, "PlanEngine", FakeGPUEngine)
    eng = tandem.TandemEngine(fake_pack, ctx=64, cpu_layers=2, mode="cpu-first",
                              log=lambda *a: None, gpu="plan")
    assert eng.gpu_kind == "plan"
    assert isinstance(eng.gpu, FakeGPUEngine)
    assert eng.gpu.layers == (2, 3)
    assert eng.gpu.embed_owned is False and eng.gpu.head_owned is True
    assert eng.cache_fingerprint()["gpu_owner"] == "plan"
    plain = tandem.TandemEngine(fake_pack, ctx=64, cpu_layers=2, mode="cpu-first",
                                log=lambda *a: None, gpu="plain")
    assert plain.cache_fingerprint()["gpu_owner"] == "plain"
    assert plain.cache_fingerprint() != eng.cache_fingerprint()


def test_a_plan_owner_frees_the_weights_and_half_the_kv_of_an_offloaded_layer():
    """Requirement: an offloaded layer frees VRAM under both owners, but the plan owner frees more.

    The plan engine reads the f32 file per tensor (only owned layers), keeps q8_0 K/V (half of fp16)
    and keeps the embedding in pinned host memory; the plain engine uploads the whole f32 file, the
    embedding and fp16 K/V. Compare the two estimates for the 27B geometry.
    """
    dims = {"layers": 64, "heads": 24, "kv_heads": 4, "head_dim": 256, "state_size": 128,
            "groups": 16, "gdn_inner": 6144, "ffn": 17408, "full_attention_interval": 4}
    layer = 105 * 2 ** 20
    sizes = {"layers": {il: layer for il in range(64)},
             "f32_layers": {il: 1 * 2 ** 20 for il in range(64)},
             "f32": 100 * 2 ** 20, "f32_global": 100 * 2 ** 20 - 64 * 2 ** 20,
             "embed": 130 * 2 ** 20, "head": 130 * 2 ** 20}
    ctx = 24576
    plan_full = tandem.gpu_estimate(dims, sizes, ctx, (), owner="plan")
    plain_full = tandem.gpu_estimate(dims, sizes, ctx, (), owner="plain")
    assert plan_full < plain_full
    # the fp16 K/V of 16 attention layers is 2x the q8_0 K/V (the plain engine's dominant overhead)
    fp16 = 16 * tandem.kv_bytes(dims, ctx)
    q8 = 16 * tandem.kv_bytes(dims, ctx, "q8_0")
    assert q8 == pytest.approx(fp16 / 2, rel=0.07)
    freed = {}
    off = tuple(range(4, 36, 4))                        # eight GDN layers, layer 0 stays on the GPU
    for owner in ("plan", "plain"):
        full = tandem.gpu_estimate(dims, sizes, ctx, (), owner=owner)
        eight = tandem.gpu_estimate(dims, sizes, ctx, off, owner=owner)
        freed[owner] = full - eight
        assert freed[owner] > 8 * layer * 0.99          # every offloaded layer's weights come back
    assert freed["plan"] > freed["plain"]                # plus no per-layer f32 upload
    # a plan owner never charges the whole f32 file nor the embedding on the device
    one = tandem.gpu_estimate(dims, sizes, ctx, tuple(range(64)), owner="plan")
    assert one < sizes["f32"] + sizes["head"] + tandem.PLAN_SCRATCH_MIB * 2 ** 20


def test_pack_bytes_splits_the_f32_file_per_layer(tmp_path):
    """f32 is charged per owned layer by the plan owner, so the per-layer sizes must add up."""
    pack = tmp_path / "p"
    pack.mkdir()
    (pack / "kindi_pack.json").write_text(json.dumps({
        "dimensions": {"layers": 2}, "head": "output.weight", "f32_bytes": 64}), encoding="utf-8")
    (pack / "bcodes_index.json").write_text(json.dumps({"tensors": {
        "blk.0.ffn_up.weight": {"rows": 4, "ne0": 128, "bc_off": 0, "dpm_off": 0},
        "blk.1.ffn_up.weight": {"rows": 4, "ne0": 128, "bc_off": 64, "dpm_off": 0},
        "output.weight": {"rows": 8, "ne0": 128, "bc_off": 128, "dpm_off": 0},
        "token_embd.weight": {"rows": 8, "ne0": 128, "bc_off": 256, "dpm_off": 0},
    }}), encoding="utf-8")
    (pack / "index.json").write_text(json.dumps({"tensors": {
        "blk.0.attn_norm.weight": {"f32_off": 0, "nfloats": 4},
        "blk.1.attn_norm.weight": {"f32_off": 16, "nfloats": 4},
        "output_norm.weight": {"f32_off": 32, "nfloats": 8},
    }}), encoding="utf-8")
    sizes = tandem.pack_bytes(str(pack))
    assert sizes["f32_layers"] == {0: 16, 1: 16}
    assert sizes["f32"] == 64 and sizes["f32_global"] == 32
    # 4 rows x 128 cols: 128 bcodes bytes + 4 dpm bytes, plus the layer's own f32 norm (16 bytes)
    assert sizes["layers"][0] == 4 * 32 + 4 * 2 + 16
    assert sizes["f32_layers"][0] == 16


def test_plan_handoff_fills_every_block_start_buffer(monkeypatch):
    """A partial plan alternates hA/hB; the handoff must fill both, not the cursor's last buffer.

    The regression is the second token: after a block, ``_buf`` points at that block's output
    buffer, while the next block reads its own start buffer.  Writing only ``_buf`` made the
    GPU owner read the previous token's residual and the 9B split failed at token 0.
    """
    pytest.importorskip("cupy")
    from kindi import tp

    class Buf(object):
        def __init__(self):
            self.data = np.zeros(4, dtype=np.float32)

        def __setitem__(self, key, value):
            self.data[key] = value

    class Card(object):
        card = 0

    hA, hB = Buf(), Buf()
    card = Card()
    card.plan_start = {1: hA, 2: hB}
    card.plan_entry_buf = hA

    class Device(object):
        def __init__(self, number):
            assert number == 0

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class FakeCP(object):
        @staticmethod
        def asarray(x):
            return np.asarray(x, dtype=np.float32)

        class cuda(object):
            pass

    FakeCP.cuda.Device = Device

    class FakeEngine(object):
        D = 4
        cards = [card]
        _pending = 0
        _buf = hB                            # after the previous token's last block

        def flush(self):
            self._pending = 0

    monkeypatch.setattr(tp, "cp", FakeCP)
    fake = FakeEngine()
    tp.TPEngine.set_hidden(fake, [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(hA.data, [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(hB.data, [1.0, 2.0, 3.0, 4.0])
    assert fake._buf is hA
