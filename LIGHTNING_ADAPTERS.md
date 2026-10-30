# Native MoE adapters

The Lightning adapter separates packed CUDA operations from checkpoint semantics. `moe.py` and `lightning_batch.py` supply linear algebra, routing, dispatch, normalization and bounded-memory execution. `lightning.py` maps the pinned Nota checkpoint onto those operations. A model sharing the MoE label still needs its own validated adapter.

## Supported contracts

| Operation | Accepted contract |
| --- | --- |
| Packed linear | GGUF Q2_0 type 42, Q8_0 type 8, F32 type 0; row-aligned blocks; FP32 accumulation |
| Router | FP32 sigmoid, selection-only correction bias, top-k, normalization of original selected scores, explicit scale; ties prefer lower expert ID |
| Batched router | 1–256 experts; each token has independent IDs and weights |
| Expert dispatch | GPU histogram, offsets and permutation; bounded jobs; no CPU transfer of selected IDs |
| Expert feed-forward | Up → ReLU squared → down, plus shared expert; no SwiGLU assumption |
| Attention | Causal GQA with online softmax; FP32, FP16 or Q8_0 KV storage; FP32 arithmetic/output; NoPE in this adapter |
| Mamba | State `[heads, head_dim, state_size]`, convolution history `[channels, kernel-1]`; state size 128 for chunk execution |
| Memory | Packed weights + recurrent state + layer-count-based KV + expansion/split scratch + 512 MiB reserve; no recurrent allocation for decoder-only graphs |

Expert matrices are stored `[expert, output_row, input_column]`. A grouped output is scattered back to assignment order `token * top_k + slot` before combination. The strict implementation sums routed contributions in slot order, then adds the shared expert, then the residual. Changing that order can change routing later in the network.

`tests/test_generic_moe.py` constructs two independent four-layer attention/MoE decoder graphs without Mamba metadata: widths 64/128, experts 8/32, top-k 2/4 and expert widths 64/192. It checks NumPy reference logits, chunk parity, branching snapshots, rejected snapshots and memory planning. This validates reuse of these primitives; it does not establish support for a real Gemma checkpoint.

## Adding another model

1. Pin model/config/tokenizer revisions and identify every block, tensor layout, quantization format and special-token rule. Reject unsupported metadata before GPU upload.
2. Write an adapter that provides its layer schedule, tensor names and dimensions. Reuse packed loading and applicable CUDA operations.
3. Implement differences explicitly: positional encoding, QK norms, softcaps, router softmax versus sigmoid, expert grouping, shared-expert gates, activation, scale and residual order. Gemma-specific operations cannot be inferred from Lightning.
4. Define memory and state contracts, including whether a prefix needs KV alone or additional recurrent state. Include all numerical choices in the cache fingerprint.
5. Establish an independent reference, operator tests, complete layer/logit parity, long prefixes, cache restoration, cancellation and API acceptance before measuring speed.

Calls into low-level dispatch assume IDs produced by the validated router. External callers must not pass arbitrary out-of-range GPU indices. The GPU kernels do not silently implement unsupported architectures.

## Numerical profiles

The V3 container entrypoint selects `optimized` kernels, `tile16` dense/expert prefill, `split` attention and 128-token chunks. The release config sets context 32768 and FP32 KV by default. FP16 and Q8_0 are optional storage profiles. Direct library use retains `batched_mv`, `grouped_mv` and `tiled` defaults; select the V3 profile with `KINDI_LIGHTNING_DENSE_PREFILL=tile16`, `KINDI_LIGHTNING_EXPERT_PREFILL=tile16` and `KINDI_LIGHTNING_ATTENTION=split`.

`tile8` reuses each unpacked ternary weight for eight inputs while retaining each token's accumulation order. `split` computes 128-key attention tiles independently, then merges them in the original sequential order with explicit FP32 FMA rounding. The workspace defaults to 128 MiB and query microbatches fit that bound. Short contexts use `warp` attention. No fast-math switch, TF32, FP16 arithmetic, Tensor Cores or DP4A is required.

The accepted V3 `tile16` dense/expert modes process sixteen output rows per block, with two rows per warp sharing the same eight input assignments. Each row retains the `q2_mv` FMA and reduction order; Q8/F32 matrices retain their existing paths. Select both modes with `KINDI_LIGHTNING_DENSE_PREFILL=tile16` and `KINDI_LIGHTNING_EXPERT_PREFILL=tile16`. The chunk remains 128. On sm_35 the kernel uses 47 registers, no local spill storage and 4128 bytes of shared memory. Full-model V3 acceptance is summarized in BENCHMARKS-LIGHTNING.md; a matrix microbenchmark alone is insufficient.

`tests/bench_prefill_variants.py` compares tile8/16/16/8 with independent Engine instances at profile changes, full warmups, strict state bytes, all 32 decode logits, token coverage and NVML telemetry. Only one Engine is resident at a time. Its `--validation-only --tokens 32736` mode runs two independent full prefills and consumes 32 generated tokens, checking the complete 32768-token budget; those timings are diagnostics, not a repeated performance comparison. CuPy is imported before constructing an Engine because CuPy 9.6's cached optional pytest import exceptions retain import-stack frames on Python 3.6. The test checks actual Engine collection before loading the next profile.

FP16 uses round-to-nearest-even storage. Q8_0 stores 32 signed codes plus a half scale; codes use the unrounded FP32 scale and round-to-nearest-even. Non-finite values and scale overflow are rejected. `KINDI_LIGHTNING_KV_READ=auto|direct|expanded` controls reads. Auto reuses one layer-sized FP32 expansion buffer for compressed prefill and long Q8 decode; it does not retain expanded copies of every layer. Mamba state stays FP32.

Snapshots preserve the raw KV dtype. Version 4 fingerprints include KV storage/read policy, attention, scratch size, device and kernel/adapter hashes. Compressed-profile reference fixtures are documented in `lightning_kv_reference.json`: the PyTorch oracle independently quantizes native pre-storage K/V at lossy boundaries. These are aligned references, not fully free-running references. Teacher-forced calibration separately measures each lossy profile against native FP32.

Current full-model acceptance is capped at 32768 tokens including generated output. The three-depth retrieval test compares its final 512-token tail against sequential execution from the same prefix; it must not be described as two independent full prefills. Separate full sequential/batched FP32 evidence is available. Operator tests at larger synthetic sizes do not establish full-model context acceptance.

An unrecoverable CUDA error makes the engine reject new work and `/health` return 503. Restart this backend process after diagnosis; Docker health status alone does not trigger an automatic restart. Legacy `cuda-memcheck` is excluded on the K6000 host because an observed `nvidia_uvm` kernel oops occurred while it ran.

`tiled`, `cublas`, `grouped_mv4` and expert `cublas` remain experiments. The dense cuBLAS profile changes reduction order. After the shared-expert accumulation fix it still fails the strict 4096-token state tolerance on RTX 4080; it is not the accepted production profile. Near-tie routing is diagnosed separately rather than hidden by increasing tolerances.
