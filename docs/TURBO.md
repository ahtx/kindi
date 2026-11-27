# Turbo

Kindi's defaults are the fastest settings that give the same output as its exact paths, or that pass the same quality gates in the modes that are already approximate. Turbo adds settings that are faster but give slightly different output. They pass the same quality gates.

Turn it on with `kindi serve --turbo`, `KINDI_TURBO=1`, or `turbo = true` in the config. For DeepSeek, add `--turbo` to the `kindi.dsv41_server` command. A setting's own environment variable always overrides turbo. `kindi turbo` (or `kindi serve --turbo --explain`) prints this table. At startup, `--turbo` logs one line for each setting it turns on.

| Model and hardware | Setting | Speed, normal to turbo | Greedy tokens | Seeded sampled replies | `kindi verify`, normal and turbo |
|---|---|---|---|---|---|
| DeepSeek V4.1 Flash 1017 UD-Q4_K_XL, RTX 3060 12 GB main card + Quadro M4000 8 GB, dual Xeon E5-2673 v4 | `KINDI_SPLIT_CARD=16` (16 streamed prompt experts per layer on the main card, default 12) | Read 90.7 to 96.0 tokens/s at 4,284 prompt tokens and 92.7 to 97.2 at 13,779. Write unchanged. | Identical at 579, 4,284 and 13,779 prompt tokens | Identical at 4,284 tokens. Different at 13,779 tokens. | PASS, 1,266 of 1,280 top-1 (98.91%), in both modes |

Bonsai, Flash-Next, Nemotron 3.5 Lightning, Nanbeige and Inkling 2 Mini have no turbo settings. Their faster settings are already the defaults.

The DeepSeek tests used 81,920 context, DSpark and 128 output tokens per request. Sampling used T=0.7, top-p 0.95 and a fixed seed.

Measured and left out of turbo:

- DeepSeek `--tmax 1024 --adaptive-prefill`. Without `KINDI_DSV_CKV4`, the RTX 3060 ran out of memory at 81,920 context. With CKV4, it read 13,779-token prompts at 98.7 to 99.1 tokens/s instead of 93.9 to 94.6. However, `kindi verify` with CKV4 gives 1,260 of 1,280 tokens with or without the larger chunks, which is below the gate.
- DeepSeek `KINDI_DSV_PREFILL_REUSE_UPLOADS=1` and `KINDI_MOE_PACK_GB=24`. Both give the same output but are no faster, so neither is a turbo setting.
- Flash-Next CPU MTP drafting is on by default, and its greedy tokens match plain decoding.
