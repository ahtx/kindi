# Kindi ARM64 vs the field — ternary/low-bit LLM inference on ARM CPUs
### 2026-11-17 — measured + prior-art comparison, optimization roadmap

Companion to `phone-tests/FINDINGS-2026-11-17-s24-vs-s23.md` (measurement forensics) and
`Kairo Verge/scripts/phone-benchmarks/PROTOCOL.md` (reproducible recipe).

## 1. The field (what exists, what it validated)

| Engine | Approach | ARM evidence (published, regime noted) |
|---|---|---|
| **llama.cpp** (ggml-org) | MAD: dequant → int8 dotprod (SDOT/UDOT) / i8mm (SMMLA); TQ1_0 1.69bpw, TQ2_0 2.06bpw ternary packings; runtime ISA dispatch armv8.0→9.2 | **Our S23 (SD 8 Gen 2), measured today**: Gemma-3-1B Q4_K_M chat decode **22.1 tok/s** steady; llama-bench **tg128 29.1 / pp512 63.5** (b12189, armv8.6 kernel auto-selected) |
| **bitnet.cpp** (microsoft/BitNet, ACL 2025) | llama.cpp fork; I2_S (MAD int8), TL1 (2.0bpw LUT), TL2 (1.67bpw LUT). Kernel source verified: `127/max` int8 act-quant, subset-sum LUTs, byte-split wide entries for pshufb/tbl, **int16 accumulate → widen int32 at block end → scale** | 1.37–5.07× vs fp16 on ARM CPUs; 2B-4T ternary supported on ARM (TL1/I2_S) |
| **T-MAC** (microsoft/T-MAC, EuroSys 2025) | LUT mpGEMM, no dequant; g=4 → 16-entry int8 LUT in ONE NEON reg (vqtbl1q); fp16 entries split into 2 byte-LUTs; TVM-tuned | **BitNet-3B: 20 tok/s 1-core / 48 tok/s 4-core on Surface Laptop 7 = 4–5× llama.cpp; 11 tok/s on Raspberry Pi 5 (Cortex-A76 — same core as our A32's big cluster)**. NMSE ≈ 3.3e-3 (fast-aggregation option: 8e-3, quality trade) |
| **Vec-LUT / vlut.cpp** (OpenBitSys, MobiSys 2026) | Vector LUT: 1→N lookup across parallel tokens → random lookup becomes contiguous vector adds; **cache-aware streamed lookup** (tile-local LUTs); I1 packing **1.60 bpw** (most compact ternary) | up to **4.2×** vs T-MAC/bitnet.cpp/llama.cpp; **2 CPU cores on Snapdragon 8 Elite beat the Hexagon NPU path 1.1×**; explicit Termux/Android support |
| **T-SAR** (arXiv 2511.13676) | analysis: **LUT memory traffic >75% of requests** in LUT kernels → in-place SIMD ALU reorg | identifies the cache/streaming bottleneck Vec-LUT attacks |
| **ciot/t2p** (blog 2026-10-17, Snapdragon X 27B ternary) | base-3 NEON + 2-bit smmla repack | plain C 19.6 → NEON 124.7 → smmla 231.6 Gweights/s; decode 1.5×, prefill 3.8×; **within 7% of memory bandwidth** |
| **BitNetKernels** (Ehsan-Ib, M4 Pro) | LUT vqtbl2q, bit-exact int | 1.17–1.24× vs MS TL1; **equality_check.cpp BEFORE any timing** (test pattern borrowed); vqtbl2q roofline 97.4 G-lookups/s |

**Regime match:** all of these are the *same regime* as Kindi ARM — ternary weights, CPU
SIMD, LUT-or-dotprod GEMV, memory-bound decode. T-MAC's Pi-5 number (A76 @ 2.4GHz, 17 GB/s)
is the closest published anchor to our A32 (A76 @ 2.0GHz).

## 2. Kindi ARM today — measured (this session, controlled protocol)

| Variant | Kernel | S23 big-core decode | Correctness |
|---|---|---|---|
| **v0** scalar (original port) | scalar fp32 pair-LUT, data-dependent branches | **3.29 / 3.17 tok/s** (g64/g256), verify 4/4, prefill 4.19/3.43 | bit-exact tokens ✓ |
| **v1** branchless | 9-entry tables t[8]=0, no branches | **8.06 / 7.77 tok/s = 2.45× v0**, verify 4/4, prefill 10.2/9.7 | **bit-exact ✓ (measured on phone)** |
| **v2** NEON fp32 | vqtbl4q byte-plane gather, same accum order | **6.45 / 6.21 tok/s — SLOWER than v1** (verify 4/4) | bit-exact ✓ but **negative speed result**: 4×vqtbl1q + 3×zip + vld4q per word saturate the shuffle port; instruction-count reasoning lost to real hardware |
| **v3** int16 LUT | int8 act-quant (127/max per 128-group) + vqtbl2q + int16 accum | **A32: 6.52/6.40 tok/s = 4.9× v0, verify 40/40 ×4 TOKEN-EXACT**; S23 running | NMSE 1.6e-4 predicted; **on-phone token gate PASSED strictest form** |
| **v3.1** streamed | v3 + words-outer lookup: each pair-table loaded ONCE per group, not per row-block (Vec-LUT cache insight); int16 partials in scratch | *(queued)* | **proven bit-identical to v3** (15 cases incl. 248320×2048 head, ARM/qemu) |

**Fleet baseline (v0 scalar, controlled):** S23 big cores **3.29/3.17** tok/s · A32 2×A76
**1.32/1.31** tok/s (verify 4/4 both; A32 .so md5 identical to S24's pulled artifact —
same-source builds converge). **v1 branchless scales across devices: A32 = 3.08/3.03 tok/s
= 2.33× its v0, verify 4/4** (S23 got 2.45×) — kernel-level win, not device-specific.

**A32 v3 vs the published anchor:** T-MAC reports BitNet-**3B** at 11 tok/s on Pi 5
(4×A76@2.4GHz) = 2.75 tok/s/core; Kindi v3 runs Bonsai-**2B** at 6.52 on 2×A76@2.0GHz
= **3.26 tok/s/core — same league as the published SOTA LUT kernel**, and llama.cpp on
this same A32 does 8.8–11.2 tok/s on the *smaller-byte* Gemma-3-1B. The gap is closing.

**v2 lesson (banked): fp32 byte-plane LUT is shuffle-bound on X3/A715** — 14 instrs/word
on paper beat scalar's ~36, but 7 shuffle-class ops/word (4 tbl + 3 zip) vs v1's load-port
path measured SLOWER (6.45 vs 8.06). Correctness oracle ≠ speed proof; always measure on
the real core. v3 cuts shuffle ops to ~3/word (1 zip + 1 tbl2 + idx) and halves table bytes
— the int16 path is the right one, exactly where bitnet.cpp/T-MAC landed.

**v1 lesson: the branches were the single biggest defect** — 2.45× from deleting them,
far above the 1.2–1.5× prediction. ~1.9G data-dependent branches/token with an ~11%
minority class wreck the deep pipelines of X3/A715 (mispredict + disrupted dependent-load
chains). The x86 kernel never had them (branchless mask); the port introduced them.

**kc_times stage breakdown (v0): gemv = 98.2%** of compute; norm+fold 0.16%, gdn-other
0.75%, attn-other 0.65%, ffn-other 0.19%. rows8 is THE target; everything else is noise.

**The published "1.33/1.30/0.87 tok/s" were artifacts** (little-core cpuset when dozing /
Android-17 shell mid-cluster confinement / warm state) — controlled big-core v0 = 3.2–3.5.
**Reconstruction proof (v1-little run):** same S23, v1 branchless kernel pinned to little
cores 0-2 (taskset 07) = **1.83/1.81 tok/s** vs **8.06/7.77 on big cores** — a **4.4× swing
from affinity alone**, same phone, same binary. The published ~1.3 sits squarely in the
little-core regime. Exact historical conditions are unrecoverable (never recorded — that IS
the finding), but the mechanism is proven: cpuset/affinity + thermal state fully explain the
published numbers with zero engine difference. See findings §L2.

## 3. Why the original port was 6–20× behind llama.cpp (gap analysis)

1. **Scalar nibble extraction + data-dependent branches** (~11.3 Gops/token; ~1.9G
   branches/token, 11% minority class) where x86 uses branchless AVX2 `srlv`+`vpermps`.
   → v1 removes branches; v2/v3 vectorize.
2. **No SIMD gather on ARM.** The port header literally says "NEON SIMD acceleration of
   rows8() is a future step". x86's 8-lane `vpermps` gather = 1 instr; ARM scalar = 8
   dependent loads. → v2: `vld4q` deinterleave + 4×`vqtbl1q` + zip-recombine (8 rows/lookup);
   v3: `vqtbl2q` int16 (8 rows/lookup, half the table bytes).
3. **fp32 LUTs = 64B/pair** re-read per row-block (T-MAC/T-SAR: LUT traffic dominates).
   → v3 int16 = 32B/pair; future: Vec-LUT-style streamed tile-local lookup.
4. **fp32 accumulation everywhere** where bitnet.cpp accumulates int16 in registers and
   widens once per block. → v3.
5. Per-token OpenMP fork/join + barriers, `omp single` norms — real but small (≤2% here).

**Kindi's structural advantages** (why parity is physically plausible):
- Pair-code packing is already LUT-native (2 bits/weight + group scales) — same family as
  TL2/I1, no dequant step ever needed.
- **0.47 GB codes/token** (Bonsai-2B) vs 0.8 GB/token (Gemma-3-1B Q4_K_M) → lower memory
  floor than the llama.cpp model we race against. At phone DRAM ~25–40 GB/s the bandwidth
  ceiling is ~50–85 tok/s; ciot showed a good kernel streams within 7% of that.
- GDN delta-net layers (18/24) → less KV traffic at long context than full attention.
- Hadamard fold is 0.16% of time — the architecture is GEMV-shaped, exactly what LUT
  kernels accelerate best.

## 4. Roadmap (projections — measured numbers supersede)

| Step | Change | S23 decode (measured) | Status |
|---|---|---|---|
| v0 | scalar baseline | **3.29 / 3.17** | measured |
| v1 | branchless tables | **8.06 / 7.77 (2.45×)** | measured, **bit-exact** — safe default |
| v2 | NEON fp32 byte-plane LUT | 6.45 / 6.21 (1.96×) | measured, bit-exact but **shuffle-bound, slower than v1** |
| **v3** | **int8-act + int16 LUT + vqtbl2q** | **15.33 / 14.67 (4.66×)** | **measured, token-exact 40/40×4 — the speed winner** |
| v3.1 | v3 + words-outer streamed lookup | on-device FAIL (0/40); isolated-kernel qemu oracle bit-identical to v3 | **SHELVED — integration bug, NOT codegen**: NDK-18 AND Termux-21 builds both fail in the full engine on hardware, yet the isolated `rows8_i16_stream` passes single/multi-thread partial-span qemu oracles. Fault is in the `gemv`/full-engine wiring, not the kernel or compiler. Next: test `gemv`-in-isolation on qemu / instrument scratch reuse across calls. |
| v4 | prefill batched GEMM (T-MAC mpGEMM / smmla) | pp currently 19.3 vs llama.cpp 63.5 | future |

**Build selector:** default (no flag) = v1 branchless (bit-exact, safe). `-DKINDI_I16_ROWS8` = v3
(speed, token-exact on reference prompts, NMSE 1.6e-4 — run textjobs before production).
`-DKINDI_NEON_ROWS8` = v2 (not recommended). `-DKINDI_I16_STREAM` = v3.1 (DO NOT USE on-device yet).

**Reproduce the optimized build on a phone (Termux, from `kindi-android-test/`):**
```
export KINDI_DEVICE=cpu KINDI_PACK=$HOME/kindi-pack-2b PYTHONPATH=$PWD \
       KINDI_CPU_CACHE=$HOME/.cache/kindi OMP_NUM_THREADS=4 CC=clang
export KINDI_CPU_CFLAGS="-O3 -march=armv8.2-a -fopenmp -fno-math-errno -ffp-contract=off -DKINDI_I16_ROWS8"
rm -f $KINDI_CPU_CACHE/libkcpu_arm.so          # force rebuild with the v3 flag
taskset f8 python3 runner_kindi.py $KINDI_PACK v3-i16 out.json   # big cores; c0 on A32
```
`cpu.py` + `kcpu_arm.c` (+`kcpu.c` for x86 layout parity) must be deployed AS A SET (the `s01`
Model field changed the struct; mixing versions trips the `kc_sizeof_model` assert).
Correctness gate = `reference.json` 4×40 token-exact; speed = decode g64/g256 after warmup,
big-core affinity, screen awake, logged battery temp (see PROTOCOL.md).

**Where this lands vs llama.cpp (same S23):** llama.cpp Gemma-3-1B chat decode 22.1 steady /
llama-bench tg128 29.1. **Kindi v3 Bonsai-2B = 15.3 = 69% of llama.cpp steady** — up from 15%
(v0). Parity not yet reached; the remaining levers are v3.1 streaming (blocked on a full-engine
integration bug, not codegen), int8-LUT/sign-trick (bitnet.cpp TL1 style, halves table bytes again),
and prefill batching.

T-MAC's 4–5× over llama.cpp on ternary + ciot's bandwidth numbers say **parity with (or a
win over) llama.cpp on the same phone is physically reachable** — the model is smaller in
bytes and the kernel family is proven. The gate is engineering, not physics.

## 5. Borrowed TESTS (not answers) — per the scouting playbook

- **BitNetKernels**: equality check before timing → our 120-case bit-exact oracle (x86 +
  qemu-ARM) ran BEFORE any phone deployment; it caught a real byte-recombine bug in v2.
- **T-MAC**: NMSE vs fp32 ground truth on real shapes → v3 accuracy harness (1.6e-4).
- **bitnet.cpp**: int16-overflow discipline (their packs_epi32 saturation) → our explicit
  bound proof: 32 pairs × 254 = 8128 per chain, 16256 summed < 32767 ✓.
- **Vec-LUT**: cache-aware tiling → v3.1 design; T-SAR's >75%-LUT-traffic measurement is
  the motivation to measure our own T re-read traffic next.
- **Our own gate**: token-exact verify (reference.json) for v1/v2; prefix+textjobs agreement
  for v3 (gemma4-engine contract precedent, GEMMA4_MIN_PREFIX=12).

## 6. Sources
- github.com/microsoft/BitNet (README + include/bitnet-lut-kernels.h, fetched 2026-11-17)
- github.com/microsoft/T-MAC (README) + arXiv 2407.00088 (EuroSys'25)
- arXiv 2512.06443 (Vec-LUT) + github.com/OpenBitSys/vlut.cpp (MobiSys'26)
- arXiv 2511.13676 (T-SAR), arXiv 2511.11248 (T-MAN NPU)
- cintu07.github.io/posts/ciot-t2p-kernel (2026-10-17)
- github.com/Ehsan-Ib/BitNetKernels (M4 Pro LUT vs TL1)
- aclanthology.org/2025.acl-long.457 (bitnet.cpp paper, perf tables)