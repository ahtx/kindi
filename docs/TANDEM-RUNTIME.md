# Tandem CPU+GPU runtime

Status: **dense layer split implemented; CPU exactness tests pass, and the opt-in GPU memory gate passes on the 9B pack (memory fit and token-exact verify); no 27B run is part of this PR.**

A dense model needs every matrix on every token, so the unit of placement is a complete transformer
block, not an expert. One backend runs a prefix or suffix of the blocks and hands the residual to the
other once per token. Each backend keeps its own GDN state and K/V cache, so no per-layer
synchronisation is needed inside a token.

## Two placements

| mode | embedding | layers | output head |
|---|---|---|---|
| `cpu-first` | CPU | CPU prefix `0..n-1`, GPU suffix `n..NL-1` | GPU |
| `gpu-first` | GPU | GPU prefix `0..NL-n-1`, CPU suffix `NL-n..NL-1` | CPU |

`--tandem-cpu-layers` takes a count (`6`), an explicit sorted layer set (`0,1,2,4,5,6` or
`0-2,4-6`), `gdn:N`, or `auto[:MiB]`. The default is a quarter of the layers. A split needs at least
one layer on each side.

## Choose the layers, not just the count

A CPU attention layer is the worst case: its per-token cost grows with context and its K/V is the
memory the split is trying to free. A gated-delta (GDN) layer is O(1) per token and has only a small
device state. `gdn:N` therefore moves the first `N` GDN layers and leaves every attention layer on the
GPU. An explicit layer set allows the same choice by hand.

`auto` computes the smallest `gdn:`-style set that makes the requested context fit. It uses the
pack's tensor sizes, the fp16 K/V of the attention layers the GPU keeps, the head, the f32 file and a
scratch allowance. It refuses instead of splitting when the model already fits, and reports the
missing amount when no split can fit. `auto:MiB` additionally asks the split to free that much
headroom. `KINDI_TANDEM_RESERVE_MB` sets the free-memory reserve (default 512 MiB).

## Loading only the owned slice

The CPU engine accepts `layers=`, `embed=` and `head=`. It converts and keeps only the blocks it
owns, plus the embedding or output head when it owns that end. The GPU engine likewise uploads only
its layer range, so an offloaded layer's weights never reach the card. This is what makes the split
reduce peak VRAM instead of loading the whole model and ignoring part of it.

The CPU side uses the AVX2/FMA/F16C backend and its existing fallbacks. The handoff surface is
`begin(tok)`, `layer(il, pos)`, `flush()`, `get_hidden()`, `set_hidden(values)` and `finish()`.

## GPU owner: plan or plain

`--tandem-gpu auto|plan|plain` selects the engine that runs the GPU half (`KINDI_TANDEM_GPU` sets
a default). `auto` picks the plan engine on a single card unless `KINDI_TP=0`.

- `plain` is the existing GPU engine with fp16 K/V. It uploads the whole f32 file, so an offloaded
  layer frees less than its weights.
- `plan` uses `TPEngine(layers=...)`. It reads only the owned layers' tensors, keeps q8_0 K/V for the
  attention layers it owns, and steps one prebuilt plan per block. It is a single-card owner.

Both owners expose `begin`/`layer`/`flush`/`get_hidden`/`set_hidden`/`finish`, so the CPU side does
not depend on which one is selected.

## Per-card VRAM ceiling and reserve

The planner uses `min(free - reserve, ceiling)`:

- `--vram-budget-mib N` (config `tandem_vram_mb`, environment `KINDI_TANDEM_VRAM_MB`) caps the GPU
  side. `0` means no ceiling. This is for a 6 GiB P106-100 or a display card that must keep a lower
  barrier than its nominal capacity.
- `--vram-reserve-mib N` (config `tandem_reserve_mb`, environment `KINDI_TANDEM_RESERVE_MB`) is the
  headroom left for the display, driver and allocation noise. The default is 512 MiB.

The ceiling applies to `auto` planning and is reported in the chosen split's log line. `auto:MiB`
keeps its old meaning: it is the extra headroom the split should free, not the card ceiling.

## Usage

```sh
kindi --pack ~/.kindi/packs/bonsai-2b verify --tandem cpu-first --tandem-cpu-layers 6
kindi --pack ~/.kindi/packs/bonsai-2b bench  --tandem gpu-first --tandem-cpu-layers 6
kindi --pack ~/.kindi/packs/bonsai-2b serve  --tandem cpu-first --tandem-cpu-layers 6
kindi --pack ~/.kindi/packs/bonsai-2b serve  --tandem cpu-first --tandem-cpu-layers auto       --tandem-gpu plan --vram-budget-mib 7168 --vram-reserve-mib 512
```

The CPU library needs `gcc` with OpenMP. A tandem run needs a CUDA/ROCm GPU;
`KINDI_DEVICE=cpu` is rejected.

## GPU memory gate (opt-in)

`tools/tandem_gate.py` is the measurement gate for the plan owner and the per-card ceiling. It
refuses without a dense pack and a usable CUDA card (exit 2), plans with the same
`vram_budget_bytes`/`plan_cpu_layers` functions the runtime uses, builds a real
`TandemEngine(..., gpu="plan", cpu_layers="auto")`, samples free memory every 10 ms through the
load and a fixed greedy run, and writes a path-free JSON record: pack identity and source hash,
card, ceiling/reserve, chosen layers, planner estimate, sampled peak and minimum free, the reply
tokens, an optional `kindi verify` result, and a verdict. Exit 0 means the sampled peak stayed
inside the budget, the requested context was not reduced, and verify passed or had no reference;
1 means the measurement or verify failed; 2 means missing pack/card/arguments. `--dry-run` prints
the split without building an engine.

    python tools/tandem_gate.py --pack ~/.kindi/packs/bonsai-27b \
      --ctx 24576 --vram-budget-mib 7168 --reserve-mib 512 \
      --json tandem-27b-24576.json

The 27B claim still needs a 27B record from a host with the 27B pack. The gate itself was
exercised on a 9B pack on 2026-11-22 (RTX 4060 Ti, sm_89, WSL2, gcc, ctx 4096, ceiling 2048 MiB,
reserve 512 MiB, 2 tokens): it planned CPU layers `0,1,2,4,5`, kept the 4096-token context, and
sampled a **1574.0 MiB** peak against the 2048 MiB budget (planner estimate 2004.9 MiB), so the
memory check passed. With the exactness child enabled, `kindi verify` passed all four 9B prompts
(40/40, 40/40, 13/13 and 40/40 tokens), so the gate returned **0**. The first run exposed a
handoff bug: a partial plan alternates its residual buffers (hA/hB), and `set_hidden` wrote the
cursor's last output buffer, which is not necessarily the next block's input. The handoff now
fills every block-start buffer, and the same verify passes at both one and five CPU layers. That
is a 9B check of the gate and the plan owner, not a 27B measurement; the 9B records are local and
not committed.

## Limits

- Dense Bonsai packs only; the MoE backends keep their own expert-tier paths.
- One GPU. The two-card tensor-parallel engine is not a tandem owner.
- Exact head only; no fast head, speculative decoding or batched prompt path.
- `TandemRunner` deliberately does not use the normal prompt cache.
- The routing tests use fake backends. Two opt-in tests run the real CPU engine when
  `KINDI_CPU_PACK` names a pack: `tests/test_cpu_layers.py` compares the per-layer sequence with
  the whole-model step, and `tests/test_tandem_cpu_e2e.py` compares both placements' greedy reply
  and handoff hidden state with the whole-model CPU engine. The tandem end-to-end test ran on the
  9B pack (ctx 32, 2 threads, a 4-token prompt and a 6-token reply): `cpu-first` and `gpu-first`
  were token-for-token and bit-identical to the whole-model run. That run used the CPU engine as both owners. The
  opt-in `tools/tandem_gate.py` passed its 9B memory and exactness checks as described above; no
  27B pack or 27B measurement is claimed.
