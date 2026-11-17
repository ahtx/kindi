# Device: Galaxy S24 — `SM-S921B` (serial `RFCX9076A9H`)

**Chipset:** Exynos 2400 (`e1s`, SM-S921B) — *label corrected 2026-11-17; was "Snapdragon 8 Gen 3". The SD 8 Gen 3 S24 is SM-S921U/e1q. Verify: `getprop ro.board.platform`.*  
**Android:** 17, debuggable Termux (arm64-v8a)  
**Engine:** Kindi v0.4.0 ARM64 CPU port (`kcpu_arm.c` + `cpu.py`)  
**Model:** Bonsai-2B PTQ1_0 (1.34 GB ternary pack, type-143 GGUF)  
**Threads:** 4 (`OMP_NUM_THREADS=4`)  
**CLANG:** 21.1.8  

## Results included

| File | Description |
|---|---|
| `verify_clean.txt` | Smoke + clean verify (`reference.json`) |
| `phone_textjobs_full.json` | Full 158-item kbench textjobs (158/158 completed) |

## Key numbers

- **Verify `reference.json`:** 4/4 PASS, 40/40 token-exact, **0.87 tok/s** (ctx 1024)
- **Smoke oracle:** `OUT_TOKENS [198, 2, 220, 16, 13, 220, 16, 13]` — matches desktop x86 oracle exactly
- **Prefill tok/s:** ~1.09–1.10 (32–1024)
- **Decode tok/s:** ~0.82–0.91 (g16–256)
- **Write decode tok/s:** ~0.82–0.83 (sentence/paragraph)
- **Max-context (≤1024):** CTX 256/512/768/1024 all OK (226/453/679/922s)
- **Textjobs (158 items):** 158/158 completed, wall ~3.1 hours, **110/158 pass = 69.6%**

## Textjobs results by category

| Category | Pass | Total |
|---|---|---|
| sentiment | 25/30 | 30 |
| escalate | 28/30 | 30 |
| spam | 29/30 | 30 |
| topic | 28/30 | 30 |
| spell | 0/15 | 15 |
| grammar | 0/12 | 12 |
| table | 0/5 | 5 |
| notes | 0/3 | 3 |
| news | 0/3 | 3 |
| **TOTAL** | **110/158** | **158** |

## Test dates

- Provisioned/verified: 2026-11-11
- Textjobs completed: 2026-11-16
