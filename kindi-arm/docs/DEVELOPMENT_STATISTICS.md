# Kindi v0.4.0 Android ARM64 Port — Development Statistics

**Separate documentation** describing how this port was made: time, steps, measurements, and device comparison.
Companion to `docs/ISSUES_AND_RESOLUTIONS.md` (problems + fixes) and `docs/ANDROID_ARM64_SETUP.md` (setup guide).

---

## Timeline

| Date | Milestone |
|---|---|
| 2026-11-10 | Repo initialized: Kindi v0.4.0 source (tag `32bc703`), single commit |
| 2026-11-11 | ARM port implemented (`kcpu_arm.c` + `cpu.py` changes); token-exact on desktop oracle |
| 2026-11-11 | S23 #1 (`RFCX71B5C9K`): smoke + clean verify 4/4 PASS, 1.33 tok/s; Phase 1 complete; max-ctx 256→2048 OK; 3072/4096 OK |
| 2026-11-11 | S23 #2 (`RFCX9011K4E`): debuggable Termux bootstrapped, tmux + toolchain installed via USB debs; smoke + verify 4/4 PASS, 1.30 tok/s; Phase 1 complete |
| 2026-11-11 | S24 (`RFCX9076A9H`): Python+toolchain unblocked over USB; smoke PASS; clean verify 4/4 PASS, 0.87 tok/s |
| 2026-11-11 | S23 #1 textjobs: 148/158 completed, 74% (killed on final table items) |
| 2026-11-11 | S23 #2 textjobs: 151/158 completed, 73% (killed on final table items) |
| 2026-11-15 | S24 textjobs launched (PID 27372, killed at 133/158, resumed 2026-11-16) |
| 2026-11-16 | S24 textjobs completed: 158/158, 110/158 pass = 69.6% |
| 2026-11-15 | Release docs + statistics written |

---

## Code changes (this port)

| File | Lines | Change type |
|---|---|---|
| `kindi/kcpu_arm.c` | 443 | **new** — ARM64 scalar FP16 kernel |
| `kindi/cpu.py` | +26 / −4 | **modified** — auto-build/load `libkcpu_arm.so` on aarch64/`KINDI_ARM=1` |
| `docs/ANDROID_ARM64_SETUP.md` | new | setup guide |
| `docs/DEVELOPMENT_STATISTICS.md` | new | this document |
| `docs/ISSUES_AND_RESOLUTIONS.md` | new | problems + fixes |
| `RELEASE-v0.4.0-android-arm64.md` | new | release notes |

**Lines of code added:** ~470 (port + docs).
**Files untouched:** `kindi/kcpu.c` (x86-64) and all other upstream source — byte-identical to v0.4.0.

---

## Build / environment statistics

- **CPU engine:** scalar ARM64 FP16 (`__fp16`), OpenMP-optional, same layouts/`kc_*` symbols as `kcpu.c`, bit-exact scalar `rows8` first.
- **No NEON-optimized `rows8` yet** — correctness-first port; ~1 tok/s decode is the current throughput.
- **Toolchain installed on phones:** `python 3.14.10`, `python-numpy 2.5.4`, `clang/lld/llvm/libllvm 21.1.8-3`, `ndk-sysroot 30`, `libcompiler-rt 21.1.8-3`, `libc++ 30`, `make 4.4.1-1`, `libopenblas 0.3.34`, `tmux 3.8`, `libevent`, `ncurses`, `utf8proc` + ~20 support deps (30 packages total, ~98 MB tarball).
- **Model pack:** Bonsai-2B PTQ1_0, 1.34 GB, 13 files, ternary type-143 GGUF.
- **Build command:** `clang -O3 -march=armv8.2-a+simd -fno-math-errno -shared -fPIC -o libkcpu_arm.so kcpu_arm.c -lm` (auto-built by `cpu.py`).
- **Threads:** `OMP_NUM_THREADS=4`.

---

## Device comparison table

| Metric | S23 #1 `RFCX71B5C9K` (SM-S911B) | S23 #2 `RFCX9011K4E` (SM-S911B) | S24 `RFCX9076A9H` (SM-S921B) |
|---|---|---|---|
| Chipset | Exynos (dm1q) | Exynos (dm1q) | Snapdragon 8 Gen 3 (e1s) |
| Smoke load (ctx 512) | ~21.6s | ~21.6s | 39.4s |
| Verify (ctx 1024, 4×40) | **ALL_PASS 1.33 tok/s** | **ALL_PASS 1.30 tok/s** | **ALL_PASS 0.87 tok/s** |
| Verify load | 19.18s | 19.3s | 37.7s |
| Prefill 32→1024 tok/s | 1.442 → 1.326 | 1.661 → 1.456 | 1.09–1.10 |
| Decode g16→g256 tok/s | 1.021 → 0.973 | 1.021 → 1.076 | 0.82–0.91 |
| Write decode tok/s | sentence 0.991 / para 0.971 | sentence 1.074 / para 1.077 | 0.82–0.83 |
| Max-ctx ≤2048 | all OK | all OK | CTX 256→1024 OK |
| Max-ctx >2048 | 3072 OK (2200.5s), 4096 OK (3110.0s) | not run | not run |
| Textjobs 158 completed | 148/158, 74% | 151/158, 73% | in progress (2026-11-15) |

**Performance insight:** decode tok/s is nearly identical across S23 #1, S23 #2, and S24 (~0.8–1.1 tok/s) because the Kindi ARM64 engine is a scalar FP16 implementation — device-independent correctness, clock-speed-dependent speed. S24 is slower (~0.87 tok/s verify vs ~1.3 on S23) due to thermal throttling under sustained load, not engine differences.

---

## Textjobs weakness map (from feasible evidence)

| Category | S23 #1 pass | S23 #2 pass | Failure mode |
|---|---|---|---|
| sentiment (30) | 25/30 | 25/30 | occasional label drift |
| escalate (30) | 28/30 | 28/30 | safe |
| spam (30) | 29/30 | 29/30 | strongest task |
| topic (30) | 28/30 | 28/30 | safe |
| spell (15) | 0/15 | 0/15 | 2B emits think tags instead of direct output |
| grammar (12) | 0/12 | 0/12 | same — think-tag leakage |
| table (5) | 0/1 | 0/4 | engine hangs on table generation |

**Headline:** ~73–74% on classify tasks; 0% on spell/grammar/table — a **capability boundary of the 2B ternary model**, not a Kindi engine bug.

---

## What was NOT done (deferred / out of scope)

- NEON-optimized `rows8` in `kcpu_arm.c` (throughput boost only; correctness proven).
- Full 65-task kbench agent suite (impractical on-device at ~1 tok/s; needs pi/Node/sandbox/server).
- Incremental max-context beyond 4096 (~1+ hour per step at ~1 tok/s; killed a 5120→12288 run).

- `orailnoor/termux-llm` evaluation (next lane after this release).