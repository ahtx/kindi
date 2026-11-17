# Device: Galaxy S23 #2 — `SM-S911B` (serial `RFCX9011K4E`)

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
| `phone_textjobs_full.json` | Full 158-item kbench textjobs (151/158 completed) |

## Key numbers

- **Verify `reference.json`:** 4/4 PASS, 40/40 token-exact, `ALL_PASS 1.30 tok/s` (ctx 1024)
- **Smoke oracle:** `OUT_TOKENS [198, 2, 220, 16, 13, 220, 16, 13]` — matches desktop x86 oracle exactly
- **Prefill tok/s:** 32→1.661, 64→1.633, 128→1.580, 256→1.543, 512→1.503, 1024→1.456
- **Decode tok/s:** g16→1.021, g32→1.032, g64→1.071, g128→1.097, g256→1.076
- **Write decode tok/s:** sentence 64@1.074, para 160@1.077
- **Max-context (≤2048):** 256/512/768/1024/1536/2048 all OK
- **Textjobs (158 items):** 151/158 completed (killed on final table items), wall ~2.57 h, 110/151 pass = **73%**

## Test date

2026-11-11
