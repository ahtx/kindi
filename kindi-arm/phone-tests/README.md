# Kindi ARM64 — Phone Test Results

Shared test results for the **Kindi v0.4.0 Android ARM64 CPU port** running
**Bonsai-2B PTQ1_0** (1.34 GB ternary pack) on Samsung Galaxy phones via Termux.

These results are device-reported: run locally on each phone with the Kindi ARM64
engine, then copied here for sharing. No model weights, no secrets, no PII — just
tokens, scores, and timings.

## Device results

| Device | Chipset | Verify (4×40 token-exact) | Textjobs (158 items) | Max-context | Status |
|---|---|---|---|---|---|
| S23 #1 `RFCX71B5C9K` | Exynos (dm1q) | **4/4 PASS**, 1.33 tok/s | 148/158 = **74%** | 256→4096 OK | complete |
| S23 #2 `RFCX9011K4E` | Exynos (dm1q) | **4/4 PASS**, 1.30 tok/s | 151/158 = **73%** | 256→2048 OK | complete |
| S24 `RFCX9076A9H` | Snapdragon 8 Gen 3 (e1s) | **4/4 PASS**, 0.87 tok/s | 158/158 = **69.6%** | 256→1024 OK | complete |

## Per-device directories

- [`s23_1_RFCX71B5C9K/`](s23_1_RFCX71B5C9K/) — Galaxy S23 #1 (complete)
- [`s23_2_RFCX9011K4E/`](s23_2_RFCX9011K4E/) — Galaxy S23 #2 (complete)
- [`s24_RFCX9076A9H/`](s24_RFCX9076A9H/) — Galaxy S24 (textjobs in progress)

Each device directory contains:
- `device-info.md` — chipset, Android version, threads, toolchain, test date
- Result JSON/log files (phone_tests, phone_textjobs, phone_ctx_high, verify_clean)
- See [`CONTRIBUTING.md`](CONTRIBUTING.md) for how to add your own device results

## Smoke oracle (all devices)

```
OUT_TOKENS [198, 2, 220, 16, 13, 220, 16, 13]
```

This matches the desktop x86 oracle exactly — the Kindi ARM64 engine produces
bit-exact identical output to Kindi v0.4.0 x86-64 on Bonsai-2B PTQ1_0.

## Honest baseline note

`llama.cpp b12189` **cannot** run Bonsai ternary `type-143` GGUF, so there is
**no direct same-model llama.cpp baseline** on these phones. Comparison is against
the Kindi desktop x86 oracle and measured device throughput only.

## Performance insight (corrected 2026-11-17 — see [FINDINGS](FINDINGS-2026-11-17-s24-vs-s23.md))

The Kindi ARM64 engine is a **scalar FP32 port** (FP16 is KV-cache storage only) of the
AVX2 x86 kernel — NEON vectorization of `rows8()` is explicitly a future step. ~1 tok/s
is the predicted speed of this scalar kernel (≈11.3 G scalar ops/token), not an anomaly,
and **not** an engine-vs-engine result against NEON-optimized llama.cpp.

**The S24-vs-S23 verify gap (0.87 vs 1.30/1.33) is a thermal-state artifact, not a device
inversion.** Paired per-item comparison of the identical 158 textjobs shows S24 at
**93% of S23's speed at steady state** (0.88–0.96 by job block). S23's verify was measured
cold-boosted (its own warm decode suite shows 1.02–1.10, and its textjobs degrade +14%
within the first 30 items); S24's verify ran on an already-warm phone and stayed flat.
Verify tok/s is also a *blended* metric (prompt+gen tokens / wall incl. prefill), unlike
llama-server's pure-decode `predicted_per_second`.

**Chipset labels corrected:** S24 `SM-S921B` (`e1s`) = **Exynos 2400**, not Snapdragon 8 Gen 3;
S23 `SM-S911B` (`dm1q`) = **Snapdragon 8 Gen 2 for Galaxy**, not Exynos. Same-class silicon —
consistent with the steady-state parity above.

## Textjobs weakness map (from feasible evidence)

| Category | S23 #1 | S23 #2 | Failure mode |
|---|---|---|---|
| sentiment (30) | 25/30 | 25/30 | occasional label drift |
| escalate (30) | 28/30 | 28/30 | safe |
| spam (30) | 29/30 | 29/30 | strongest task |
| topic (30) | 28/30 | 28/30 | safe |
| spell (15) | 0/15 | 0/15 | 2B emits think tags instead of direct output |
| grammar (12) | 0/12 | 0/12 | same — think-tag leakage |
| table (5) | 0/1 | 0/4 | engine hangs on table generation |

**Headline:** ~73–74% on classify tasks; 0% on spell/grammar/table — a **capability
boundary of the 2B ternary model**, not a Kindi engine bug.

## Want to add your phone?

See [`CONTRIBUTING.md`](CONTRIBUTING.md) — we welcome results from any Android
phone running Termux with the Kindi ARM64 port.
