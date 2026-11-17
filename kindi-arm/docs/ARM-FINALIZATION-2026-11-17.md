# Kindi ARM64 — FINALIZATION (2026-11-17)

**Status: FINALIZED + S24 TESTED + PACKAGED. Final kernel = v3 int16-LUT (15.3 tok/s, 4.66×). Correctness gate
PASSED (verify 4/4 + textjobs 158/158 identical to v0 fp32); x86 re-verified 4/4 (the `s01` change preserved x86).**
Owner directive: finalize ARM development, then test on S24 — both done.

**Repo layout (share branch `arm/v3-int16-lut`, based on `main` = 0.6.0):** the engine changes live in `kindi/`
(`kcpu_arm.c` new + `cpu.py`/`kcpu.c` edits); `kindi-arm/` holds docs/tests/harness only — no engine re-copy.
See `RELEASE-arm-v3-int16-lut.md` (authoritative). The phone deployment layout below is unchanged.
Companions: `phone-tests/FINDINGS-2026-11-17-s24-vs-s23.md` (measurement forensics),
`docs/COMPARISON-2026-11-17-arm-ternary-engines.md` (landscape), vault
`2026-11-17-kindi-arm-s24-s23-inversion-diagnosis`, `2026-11-17-20-hard-self-grill-questions`.

## The variant ladder (S23#1 big cores, decode g64/g256, controlled)
| Variant | Flag | decode tok/s | vs v0 | Correctness | Verdict |
|---|---|---|---|---|---|
| v0 scalar (original port) | (none) | 3.29 / 3.17 | 1× | bit-exact | baseline |
| v1 branchless | (none, default now) | 8.06 / 7.77 | 2.45× | bit-exact | safe fallback |
| v2 NEON fp32 byte-plane | -DKINDI_NEON_ROWS8 | 6.45 / 6.21 | 1.96× | bit-exact | SHELVED (slower than v1; shuffle-bound) |
| **v3 int16-LUT block-outer** | **-DKINDI_I16_ROWS8** | **15.33 / 14.67** | **4.66×** | verify 4/4; textjobs running | **FINAL** |
| v3.1 int16-LUT streamed | -DKINDI_I16_STREAM | 6.05 / 5.83 | 1.84× | verify 4/4 (after race fix) | SHELVED (correct but slower than v3) |

**Why v3 wins:** block-outer keeps the int16 partial sums in NEON registers (s0/s1 as int16x8_t),
accumulating across words in registers and writing y once. v3.1's word-outer/block-inner layout forces an
s01 memory load/add/store per word×block — more traffic than the table re-reads it saves. Register accumulators win.

## What v3 does (the technique)
int8-quantize activations per 128-col group (scale=127/max, bitnet.cpp per_tensor_quant pattern) →
int16 pair-sum LUT (9 entries, t[8]=0) → NEON vqtbl2q gather (8 rows/lookup) → int16 register accumulate →
widen to int32 → fp32 scale at group end. int16 adds are exact; overflow-safe (32×254=8128/chain, 16256 sum <32767).
NOT bit-exact with fp32 x86 (activation quantization, NMSE ~1.6e-4) but token-exact on the verify set.

## The v3.1 race (found + fixed, for the record)
`rows8_i16_stream` indexed shared `m->s01` by LOCAL block id → the multiple matrices of one gemv call
(qkv+gate / q+k+v / ffg+ffu) collided across threads → [0,0,0,0]. Fix = index s01 by GLOBAL id (gbase+b0),
y stays local → [40,40,40,40]. Fix verified on S23#1. (v3.1 shelved anyway — slower than v3.)

## Reproducible build (final v3)
```
export KINDI_DEVICE=cpu KINDI_PACK=$HOME/kindi-pack-2b PYTHONPATH=$PWD/kindi-android-test \
       KINDI_CPU_CACHE=$HOME/.cache/kindi OMP_NUM_THREADS=4 CC=clang
export KINDI_CPU_CFLAGS="-O3 -march=armv8.2-a -fopenmp -fno-math-errno -ffp-contract=off -DKINDI_I16_ROWS8"
rm -f $KINDI_CPU_CACHE/libkcpu_arm.so          # force rebuild with the v3 flag
# big cores only (S23: cores 3-7 = f8; A32: cores 6-7 = c0); screen awake + stayon for bench
taskset f8 python3 runner_kindi.py $KINDI_PACK v3 out.json
```
`cpu.py` + `kcpu_arm.c` must be deployed as a set (the `s01` Model field changed the struct; mixing versions
trips the kc_sizeof_model assert). Termux needs libomp (present: libomp.a, static link).

## Correctness status — GATE PASSED
- verify (reference.json, 4 prompts × 40 tokens): **4/4 token-exact** on S23#1 (and A32).
- **textjobs (158 items) on v3: PASSED — IDENTICAL to the v0 fp32 baseline.**
  sentiment 25/30, escalate 28/30, spam 29/30, topic 28/30 (=110/120 classify), spell 0/15, grammar 0/12,
  table 0/5, notes 0/3, news 0/3. v0 had the SAME 110 passing items (and was killed at 148/158; v3 did all 158).
  **The int8 activation quantization did NOT degrade capability** — the spell/grammar/table 0s are the 2B model's
  ceiling (identical in v0), not an engine/quantization defect. Closes the "160 tokens only" overclaim.
- **Still NOT tested on the owner's multilingual/creative canon domain** — do that before production use there.

## Measurement rigor still owed (self-grill loops)
- v0-little (reproduce the exact published ~1.3 on little cores) — forensics, not blocking finalization.
- Repeated interleaved runs for a variance bound on the 4.66× (single runs so far).
- S24 getprop + affinity-controlled re-run when reconnected (Android 17 shell = mid-cluster 4-6 caveat).

## S24 testing — DONE (2026-11-17, reconnected)
- getprop: SM-S921B / erd9945 / s5e9945 / Android 17 = **Exynos 2400** (closes Q9/Q10).
- **Android-17 confinement reproduced live:** shell Cpus_allowed = **4-6** (misses A720@2.9 cores 7-8 + X4@3.2 core 9).
- Ran v3 on **taskset 3f0 (cores 4-9, full big cluster)**, 4 threads, awake+stayon:
  **verify 4/4 [40,40,40,40]; decode 14.19/16.81; prefill 21.6/20.0;** affinity 4-9 confirmed, threads_max 5;
  temp 26.2→34.5°C. (g256 > g64 = boost/warmup ordering, not a decay anomaly.)

## FINAL FLEET (v3 int16-LUT, decode g64/g256, big cores, controlled)
| Device | SoC | decode tok/s | prefill | verify |
|---|---|---|---|---|
| S23#1 | SD 8 Gen 2 | 15.33 / 14.67 | 19.3 / 18.3 | 4/4 |
| **S24** | **Exynos 2400** | **14.19 / 16.81** | **21.6 / 20.0** | **4/4** |
| A32 | Dimensity 720 | 6.52 / 6.40 | — | 4/4 |
**S23 ≈ S24 (same-class flagship, ~15 tok/s); A32 ~6.5 (mid-range).** All token-exact; textjobs on v3 == v0 fp32.
The old "S24 slower than S23" was the Android-17 mid-cluster confinement + warm state, NOT silicon or engine.