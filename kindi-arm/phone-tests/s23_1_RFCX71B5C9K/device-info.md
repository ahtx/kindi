# Device: Galaxy S23 #1 — `SM-S911B` (serial `RFCX71B5C9K`)

**Chipset:** Snapdragon 8 Gen 2 for Galaxy (`dm1q`, SM-S911B) — *label corrected 2026-11-17; was "Exynos" (no Exynos S23 exists). Verify: `getprop ro.board.platform`.*  
**Android:** verified on debuggable Termux (arm64-v8a)  
**Engine:** Kindi v0.4.0 ARM64 CPU port (`kcpu_arm.c` + `cpu.py`)  
**Model:** Bonsai-2B PTQ1_0 (1.34 GB ternary pack, type-143 GGUF)  
**Threads:** 4 (`OMP_NUM_THREADS=4`)  
**CLANG:** 21.1.8  

## Results included

| File | Description |
|---|---|
| `phone_tests.json` | Phase 1 suite (QA + speed + write + max-context ≤2048) |
| `phone_textjobs_full.json` | Full 158-item kbench textjobs (148/158 completed) |
| `phone_ctx_high.json` | Incremental max-context beyond 2048 (3072/4096 OK) |

## Key numbers

- **Verify `reference.json`:** 4/4 PASS, 40/40 token-exact, `ALL_PASS 1.33 tok/s` (ctx 1024)
- **Smoke oracle:** `OUT_TOKENS [198, 2, 220, 16, 13, 220, 16, 13]` — matches desktop x86 oracle exactly
- **Prefill tok/s:** 32→1.442, 64→1.431, 128→1.426, 256→1.394, 512→1.368, 1024→1.326
- **Decode tok/s:** g16→1.021, g32→0.988, g64→1.001, g128→0.980, g256→0.973
- **Write decode tok/s:** sentence 64@0.991, para 160@0.971
- **Max-context (≤2048):** 256/512/768/1024/1536/2048 all OK
- **Max-context (>2048):** 3072 OK (2200.5s), 4096 OK (3110.0s); stopped before 5120+
- **Textjobs (158 items):** 148/158 completed (killed on final table items), wall ~2.56 h, 110/148 pass = **74%**

## Test date

2026-11-11
