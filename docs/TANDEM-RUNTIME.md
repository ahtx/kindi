# Tandem CPU+GPU runtime

Status: **dense layer split implemented; CPU tests pass, but no GPU run is part of this PR.**

A dense model needs every matrix on every token, so the unit of placement is a complete transformer
block, not an expert. One backend runs a prefix or suffix of the blocks and hands the residual to the
other once per token. Each backend keeps its own GDN state and K/V cache, so no per-layer
synchronisation is needed inside a token.

## Two placements

| mode | embedding | layers | output head |
|---|---|---|---|
| `cpu-first` | CPU | CPU prefix `0..n-1`, GPU suffix `n..NL-1` | GPU |
| `gpu-first` | GPU | GPU prefix `0..NL-n-1`, CPU suffix `NL-n..NL-1` | CPU |

`--tandem-cpu-layers` takes a count (`6`) or an explicit sorted layer set (`0,1,2,4,5,6` or
`0-2,4-6`). The default is a quarter of the layers. A split needs at least one layer on each side.

## Loading only the owned slice

The CPU engine accepts `layers=`, `embed=` and `head=`. It converts and keeps only the blocks it
owns, plus the embedding or output head when it owns that end. The GPU engine likewise uploads only
its layer range, so an offloaded layer's weights never reach the card. This is what makes the split
reduce peak VRAM instead of loading the whole model and ignoring part of it.

The CPU side uses the AVX2/FMA/F16C backend and its existing fallbacks. The handoff surface is
`begin(tok)`, `layer(il, pos)`, `flush()`, `get_hidden()`, `set_hidden(values)` and `finish()`.

## Usage

```sh
kindi --pack ~/.kindi/packs/bonsai-2b verify --tandem cpu-first --tandem-cpu-layers 6
kindi --pack ~/.kindi/packs/bonsai-2b bench  --tandem gpu-first --tandem-cpu-layers 6
kindi --pack ~/.kindi/packs/bonsai-2b serve  --tandem cpu-first --tandem-cpu-layers 6
```

The CPU library needs `gcc` with OpenMP (or clang with libomp on Windows). A tandem run needs a
CUDA/ROCm GPU; `KINDI_DEVICE=cpu` is rejected.

## Limits

- Dense Bonsai packs only; the MoE backends keep their own expert-tier paths.
- One GPU. The two-card tensor-parallel engine is not a tandem owner in this PR.
- Exact head only; no fast head, speculative decoding or batched prompt path.
- `TandemRunner` deliberately does not use the normal prompt cache.
- The routing tests use fake backends. A real CPU per-layer test skips when no pack is installed.
  This PR does not claim a 27B or GPU measurement.
