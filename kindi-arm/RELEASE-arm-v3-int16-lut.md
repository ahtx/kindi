# Kindi ARM64 — v3 int16-LUT optimization (4.66× decode, correctness-validated)

**Branch:** `arm/v3-int16-lut` (based on `main` = Kindi **0.6.0**)
**Date:** 2026-11-17
**Scope:** ARM64 (aarch64) CPU `rows8` ternary matmul kernel for Android Termux. x86-64 path preserved.

**Released by:** Tom Makela (TomYam & The Bots) — http://Kvasir.Asia

---

## TL;DR

The ARM64 ternary decode kernel was optimized from a scalar FP16 port (**3.3 tok/s**) to an
**int8-activation + int16 pair-sum-LUT NEON kernel (15.3 tok/s) — a 4.66× speedup** — while staying
**token-exact** and **matching the fp32 baseline on the full 158-item textjobs bench**. Verified on
Galaxy S23 (SD 8 Gen 2), S24 (Exynos 2400), and A32 (Dimensity 720); **x86 re-verified unchanged**.

Final kernel flag: **`-DKINDI_I16_ROWS8`**. Default (no flag) remains the bit-exact branchless v1.

## How this is laid out (no engine duplication)
The Kindi engine already lives in `kindi/` (this repo, `main`). The ARM contribution is **1 new file +
2 small edits** to that engine, plus a focused `kindi-arm/` folder for docs/tests/harness. We do **not**
re-copy the engine.

- `kindi/kcpu_arm.c` — **new** ARM64 kernel (all variants behind `#ifdef`).
- `kindi/cpu.py` — **modified**: auto-detect aarch64 (or `KINDI_ARM=1`) and build/load `libkcpu_arm.so`;
  on x86 it falls back to `kcpu.c` unchanged.
- `kindi/kcpu.c` — **modified**: +1 line (`s01` scratch field, struct-layout parity with `cpu.py`).
- `kindi-arm/` — docs, phone test records + raw artifacts, the bench harness, and this release note.

**Base-version note:** this targets the Kindi CPU engine, which is **byte-identical from v0.4.0 through
0.6.0** (`cpu.py`/`kcpu.c` verified identical). 0.6.0's changes are GPU/new-model only; the CPU ternary
path did not move, so the ARM changes apply cleanly to `main` and x86 behavior is preserved.

---

## What changed / what was fixed

### 1. The kernel optimization ladder (`kindi/kcpu_arm.c`)
| Variant | Compile flag | Decode tok/s (S23 big cores, g64/g256) | Correctness | Status |
|---|---|---|---|---|
| v0 scalar FP16 (original port) | *(none)* | 3.29 / 3.17 | bit-exact | baseline |
| v1 branchless | *(none — now default)* | 8.06 / 7.77 | bit-exact | safe fallback |
| v2 NEON fp32 byte-plane | `-DKINDI_NEON_ROWS8` | 6.45 / 6.21 | bit-exact | shelved (shuffle-bound, slower than v1) |
| **v3 int16-LUT block-outer** | **`-DKINDI_I16_ROWS8`** | **15.33 / 14.67** | token-exact; textjobs == v0 | **FINAL** |
| v3.1 int16-LUT streamed | `-DKINDI_I16_STREAM` | 6.05 / 5.83 | token-exact (after race fix) | shelved (correct but slower than v3) |

**How v3 works:** int8-quantize activations per 128-column group (scale = 127/max, the bitnet.cpp
`per_tensor_quant` pattern) → build an int16 pair-sum lookup table (9 entries, `t[8]=0`) → NEON
`vqtbl2q` gather (8 rows per lookup) → accumulate int16 partial sums **in registers** → widen to int32 →
apply the fp32 group scale once. int16 adds are exact and overflow-safe (max chain 32×254 = 8128; two
halves summed = 16256 < 32767). Block-outer ordering keeps partial sums in registers and writes `y` once.

**Why not v3.1 (streamed):** word-outer/block-inner forces an `s01` scratch load/add/store per word×block,
which costs more than the expert-table re-reads it saves. Register accumulators (v3) win. Measured, not assumed.

### 2. A real concurrency bug found + fixed (v3.1)
`rows8_i16_stream` indexed the shared `m->s01` scratch by **local** block id, so the multiple matrices of
one `gemv` call (qkv+gate / q+k+v / ffg+ffu) collided across OpenMP threads → verify went `[0,0,0,0]`.
**Fix:** index `s01` by **global** block id (`gbase + b0`), keep `y` local → `[40,40,40,40]`, parallelism
preserved (no barrier). Fix verified on-device. (v3.1 shelved anyway — slower than v3 — but the bug and fix
are recorded so nobody re-introduces it.)

### 3. Measurement fixes (why earlier phone numbers were wrong)
See `kindi-arm/phone-tests/FINDINGS-2026-11-17-s24-vs-s23.md`. Three artifacts were corrected:
- **Android CPU-affinity confinement:** Android 17 confines the adb shell to the **mid cluster** (S24 cores
  4–6), missing the fast prime cores. Escape with `taskset` (below). This — not silicon — made S24 look slow.
- **Chipset labels were swapped** in early notes. Verified via `getprop`: S23 = SM8550 (SD 8 Gen 2),
  S24 = s5e9945 (Exynos 2400), A32 = MT6853 (Dimensity 720).
- **Metric blending:** the old "verify tok/s" blended prompt+decode. This release reports **pure decode** and
  **pure prefill** separately, plus the token-exact verify as a correctness (not speed) check.

> The prior release's headline verify speeds (S23 ~1.3, S24 ~0.9 tok/s) were the *blended, confined* metric.
> The controlled pure-decode numbers here (S23 ~15, S24 ~14–17) are the corrected result. **S23 ≈ S24.**

---

## Tests done and verified

### Correctness
| Test | Result |
|---|---|
| `reference.json` verify (4 prompts × 40 tokens), ARM v3 | **4/4 PASS, 40/40 token-exact** on S23, S24, A32 |
| **textjobs, 158 items, on v3** | **identical to the v0 fp32 baseline** (table below) |
| **x86 re-verify with the `s01` change** | **4/4 token-exact** on the laptop (built `libkcpu.so`, x86 `kcpu.c`) — x86 preserved |
| ARM re-verify from the integrated layout | **4/4 token-exact** on S23 (source md5 matches the tested build) |

| textjobs task | v0 (fp32) | v3 (int16-LUT) |
|---|---|---|
| sentiment | 25/30 | 25/30 |
| escalate | 28/30 | 28/30 |
| spam | 29/30 | 29/30 |
| topic | 28/30 | 28/30 |
| spell / grammar / table / notes / news | 0 | 0 |

The 0-score tasks are the **2B model's capability ceiling** (identical in fp32 v0), not an engine or
quantization defect. **The int8 activation quantization did not degrade capability.** (v0 had been killed at
148/158; v3 completed all 158.) Raw result: `kindi-arm/phone-tests/artifacts/textjobs_v3.json`.

### Performance (controlled: big cores, 4 OMP threads, screen awake, steady-state)
| Device | SoC | decode g64/g256 tok/s | prefill tok/s | verify |
|---|---|---|---|---|
| S23 #1 | SD 8 Gen 2 | 15.33 / 14.67 | 19.3 / 18.3 | 4/4 |
| S24 | Exynos 2400 | 14.19 / 16.81 | 21.6 / 20.0 | 4/4 |
| A32 | Dimensity 720 | 6.52 / 6.40 | — | 4/4 |

Raw results: `kindi-arm/phone-tests/artifacts/kindi-bench-v3-s24.json`, `…-v31-fixed-full.json`.

---

## Clone and reproduce

### Prerequisites
- An ARM64 Android phone with **Termux** (F-Droid/GitHub, NOT Play Store), `clang`, `python`, `numpy`, and
  **libomp** (`pkg install libomp`; static `libomp.a` is used). See `kindi-arm/docs/ANDROID_ARM64_SETUP.md`.
- The **Bonsai-2B PTQ1_0 pack** (`kindi-pack-2b/`, with `kindi_pack.json`). Convert per the base Kindi docs.

### Deploy to the phone
The engine is `kindi/` in this repo. Push it (plus the harness) so they are siblings on the phone:
```
repo kindi/                     →  phone ~/kindi-android-test/kindi/
repo kindi-arm/runner_kindi.py  →  phone ~/kindi-android-test/runner_kindi.py
repo kindi-arm/run_variant.sh   →  phone ~/run_variant.sh
pack                            →  phone ~/kindi-pack-2b/
```
(Use `adb push` + `run-as com.termux`, or the USB/offline method in `ANDROID_ARM64_SETUP.md`.)

### Build + run the final v3 kernel (inside `run-as com.termux`)
```sh
export PREFIX=/data/data/com.termux/files/usr
export PATH=$PREFIX/bin:$PATH LD_LIBRARY_PATH=$PREFIX/lib
export HOME=/data/data/com.termux/files/home
export KINDI_DEVICE=cpu PYTHONPATH=$HOME/kindi-android-test
export KINDI_CPU_CACHE=$HOME/.cache/kindi OMP_NUM_THREADS=4 CC=clang
export KINDI_CPU_CFLAGS="-O3 -march=armv8.2-a -fopenmp -fno-math-errno -ffp-contract=off -DKINDI_I16_ROWS8"
rm -f "$KINDI_CPU_CACHE/libkcpu_arm.so"        # force rebuild with the v3 flag
cd "$HOME/kindi-android-test"
# BIG CORES ONLY + escape Android shell confinement (mask per device, see below):
/system/bin/taskset <MASK> python3 runner_kindi.py "$HOME/kindi-pack-2b" v3 out.json
```
Or use the wrapper: `sh run_variant.sh v3 "-DKINDI_I16_ROWS8 -ffp-contract=off" <MASK>`.

**Affinity mask per device** (big cores; Android confines the shell otherwise):
| Device | big cores | mask |
|---|---|---|
| S23 (SD 8 Gen 2) | 3–7 | `f8` |
| S24 (Exynos 2400) | 4–9 | `3f0` |
| A32 (Dimensity 720) | 6–7 | `c0` |

Keep the screen awake during a run (`svc power stayon true`, screen on) so the app-context cpuset stays on
big cores; log battery temp before/after. `runner_kindi.py` records thread count + `Cpus_allowed_list` at
start/end so you can prove the run was actually on the big cores with 4 threads.

### Run the correctness gates
```sh
# token-exact verify only (fast):
KINDI_BENCH_VERIFY_ONLY=1 sh run_variant.sh v3 "-DKINDI_I16_ROWS8 -ffp-contract=off" <MASK>
# full 158-item textjobs (resume-safe, ~2.5h on S23):
sh run_textjobs_v3.sh        # writes textjobs_v3.json; compare against the fp32 baseline
```
`kindi-arm/tools/phone_textjobs.py` is the 158-item runner; it uses **kindi-bench 0.1.1 data as-is** (the
repo's `kindi/bench.py` is unchanged). To compare against fp32, rebuild with no `-DKINDI_I16_ROWS8` (v1
default) and re-run; the task scores should match. On x86, `cpu.py` builds `kcpu.c` automatically — the same
verify passes there (that is the x86-preserved check).

### Files for reproduction
- `kindi/kcpu_arm.c` — the kernel (all variants behind `#ifdef`).
- `kindi/cpu.py` + `kindi/kcpu.c` — **deploy together with `kcpu_arm.c`** (the `s01` Model field changed the
  struct; mixing versions trips the `kc_sizeof_model` assert).
- `kindi-arm/runner_kindi.py` — controlled speed/verify harness (records affinity + threads; finds `kindi/`
  either as a sibling on the phone or at the repo root).
- `kindi-arm/run_variant.sh`, `kindi-arm/run_textjobs_v3.sh` — on-device wrappers.
- `kindi-arm/tools/phone_textjobs.py` — 158-item correctness runner.

---

## Honest caveats (read before relying on this)
- **Not bit-exact with fp32 x86.** v3 quantizes activations to int8 (NMSE ~1.6e-4). It is **token-exact** on the
  verify set and **matches v0 on all 158 textjobs**, but it is not a bit-for-bit fp32 reproduction. The default
  (no flag) v1 path IS bit-exact if you need that.
- **Not yet tested on multilingual / creative / long-form prose** (textjobs is technical/classification). Validate
  on your own domain before production story use.
- **Single-run speeds.** The 4.66× is from controlled single runs; a repeated-interleaved variance bound is still
  owed. Thermal state matters (a hot phone throttles); we logged temp and used steady-state requests.
- **Prefill is not batched** (the prefill/decode gap is larger than decode alone would suggest) — a known next lever.
- **`llama.cpp` cannot run Bonsai ternary `type-143`**, so there is no same-model llama.cpp baseline; comparison is
  against the Kindi x86 oracle and measured device throughput.
- **No compiled `.so` is shipped.** Rebuild from source with the flags above; the verified build's md5 is recorded
  in `kindi-arm/phone-tests/artifacts/MD5SUMS` as provenance.

## What did NOT change
- The x86-64 compute path (`kcpu.c` got only the struct-parity `s01` field; x86 verify re-confirmed 4/4).
- All other Kindi source; `kindi/bench.py`; kindi-bench itself.

## Companions
- `kindi-arm/docs/ARM-FINALIZATION-2026-11-17.md` — final state + fleet + build flags.
- `kindi-arm/phone-tests/FINDINGS-2026-11-17-s24-vs-s23.md` — the measurement forensics (affinity, chipsets, metric).
- `kindi-arm/docs/COMPARISON-2026-11-17-arm-ternary-engines.md` — where this sits vs MAPLE / tandem.cpp.
- `kindi-arm/RELEASE-v0.4.0-android-arm64.md` — the original ARM port (historical; this release supersedes its layout).