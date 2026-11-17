# Android ARM64 Setup — Kindi v0.4.0 on Termux (USB / Offline)

**Target:** Galaxy S23 (`SM-S911B`) and Galaxy S24 (`SM-S921B`) running Termux.
**Engine:** Kindi v0.4.0 ARM64 CPU port (`kindi/kcpu_arm.c` + `kindi/cpu.py`).
**Model:** Bonsai-2B PTQ1_0 (1.34 GB ternary pack).
**No internet required on the phone** — all packages are downloaded on the host and pushed over USB.

---

## Prerequisites (host)

| Item | Path |
|---|---|
| Debuggable Termux APK (arm64-v8a) | `/tmp/opencode/phone2-setup/apk/termux-app_v0.118.3+github-debug_arm64-v8a.apk` |
| tmux deb set (install first) | `/tmp/opencode/phone2-setup/debs/` |
| Full toolchain/runtime deb bundle (30 pkgs) | `/tmp/opencode/debs-full.tar` (98 MB) |
| Kindi source + ARM port | `~/Projects/kindi-android-test/` |
| Model pack (Bonsai-2B PTQ1_0) | `~/.kindi/pack-2b/` |
| Test helpers | `/tmp/opencode/phone_tests.py`, `/tmp/opencode/phone_textjobs.py`, `/tmp/opencode/phone_ctx_step.py` |
| ADB | `adb` on host, phone authorized |

---

## Step 0 — Confirm ADB + phone (host)

```bash
export SERIAL=RFCX9076A9H   # S24 example; use RFCX71B5C9K / RFCX9011K4E for S23
adb devices -l
adb -s "$SERIAL" get-state
```

**USER CHECKPOINT (popup, 60s):** If `adb devices` shows `unauthorized`, tap **Allow** on the phone's "Allow USB debugging?" dialog (optionally "Always allow"). Re-run `adb devices -l` until it shows `device`.

---

## Step 1 — Install + bootstrap Termux (start from Termux)

```bash
adb -s "$SERIAL" install -r /tmp/opencode/phone2-setup/apk/termux-app_v0.118.3+github-debug_arm64-v8a.apk
adb -s "$SERIAL" shell monkey -p com.termux -c android.intent.category.LAUNCHER 1
```

**USER CHECKPOINT (popup, 120s):** On the phone, approve any **Install / Update / Install unknown app / Play Protect** prompts. **Open Termux once** and wait for first-run bootstrap to finish until you see a shell prompt. Wait up to 120s.

**Verify:**
```bash
adb -s "$SERIAL" shell 'pm path com.termux; run-as com.termux sh -c "echo RUNAS_OK; ls /data/data/com.termux/files/usr/bin/sh"'
```
Expect `package:...`, `RUNAS_OK`, and path to `sh`.

**IF FAIL:** ensure this exact debug APK installed (release Termux won't allow `run-as`) → reinstall APK → open Termux once → retry.

---

## Step 2 — Install packages (tmux first, then full toolchain) — offline debs

**Push (host):**
```bash
adb -s "$SERIAL" shell 'rm -rf /data/local/tmp/tmux-debs && mkdir -p /data/local/tmp/tmux-debs'
adb -s "$SERIAL" push /tmp/opencode/phone2-setup/debs/. /data/local/tmp/tmux-debs/
adb -s "$SERIAL" push /tmp/opencode/debs-full.tar /data/local/tmp/debs-full.tar
adb -s "$SERIAL" shell 'rm -rf /data/local/tmp/debs-full && mkdir -p /data/local/tmp/debs-full && tar xf /data/local/tmp/debs-full.tar -C /data/local/tmp/debs-full'
```

**Install (in Termux):**
```bash
adb -s "$SERIAL" shell 'run-as com.termux sh -c "export PREFIX=/data/data/com.termux/files/usr; export PATH=\$PREFIX/bin:\$PATH; mkdir -p \$HOME/tmux-debs \$HOME/debs-full; cp /data/local/tmp/tmux-debs/*.deb \$HOME/tmux-debs/; cp /data/local/tmp/debs-full/*.deb \$HOME/debs-full/; cd \$HOME/tmux-debs && \$PREFIX/bin/dpkg -i --force-confnew --force-overwrite ./*.deb; cd \$HOME/debs-full && \$PREFIX/bin/dpkg -i --force-confnew --force-overwrite ./*.deb"'
```

**Verify (required):**
```bash
adb -s "$SERIAL" shell 'run-as com.termux sh -c "export PREFIX=/data/data/com.termux/files/usr PATH=\$PREFIX/bin:\$PATH LD_LIBRARY_PATH=\$PREFIX/lib; python3 --version; python3 -c \"import numpy; print(numpy.__version__)\"; clang --version | head -1; tmux -V"'
```
Expect `Python 3.14.x`, `2.5.x`, `clang version 21.x`, `tmux 3.8`.

**IF FAIL:** check debs staged (`ls \$HOME/debs-full \| wc -l`), check `PATH` includes `$PREFIX/bin` → re-run dpkg (safe to retry) → `dpkg --configure -a` inside Termux if needed. Do **not** use `pkg/apt` on the phone (repo fetch unreliable / clock skew breaks TLS).

**USER CHECKPOINT (popup, 60s):** if Android/Termux shows any permission/storage prompt during install, approve it; wait up to 60s and retry VERIFY.

---

## Step 3 — Push Kindi code + model pack

```bash
# Code (tar stream into Termux home — avoids /sdcard scoped-storage issues)
adb -s "$SERIAL" exec-out "tar cf - -C \"$HOME/Projects\" kindi-android-test" | \
  adb -s "$SERIAL" exec-in 'run-as com.termux tar xf - -C /data/data/com.termux/files/home'

# Helper scripts
adb -s "$SERIAL" push /tmp/opencode/phone_tests.py /tmp/opencode/phone_textjobs.py /tmp/opencode/phone_ctx_step.py /data/local/tmp/
adb -s "$SERIAL" shell 'run-as com.termux sh -c "cp /data/local/tmp/phone_tests.py /data/local/tmp/phone_textjobs.py /data/local/tmp/phone_ctx_step.py /data/data/com.termux/files/home/kindi-android-test/"'

# Model pack 1.34 GB (reuse if already on /sdcard)
adb -s "$SERIAL" shell 'rm -f /data/local/tmp/kindi-pack-2b.tar && tar cf /data/local/tmp/kindi-pack-2b.tar -C /sdcard kindi-pack-2b'
adb -s "$SERIAL" shell 'run-as com.termux sh -c "rm -rf \$HOME/kindi-pack-2b; tar xf /data/local/tmp/kindi-pack-2b.tar -C \$HOME; echo PACK_EXTRACTED"'
```

**Verify:**
```bash
adb -s "$SERIAL" shell 'run-as com.termux sh -c "ls \$HOME/kindi-android-test/kindi/kcpu_arm.c \$HOME/kindi-android-test/smoke.py \$HOME/kindi-android-test/kindi/reference.json \$HOME/kindi-pack-2b/kindi_pack.json"'
```

**IF FAIL:** check host paths exist → if `/sdcard` read blocked, push pack directly to `/data/local/tmp` via tar stream and extract as Termux user → retry VERIFY.

---

## Step 4 — Run smoke + clean verify (acceptance)

```bash
adb -s "$SERIAL" shell 'run-as com.termux sh -c "\
export PREFIX=/data/data/com.termux/files/usr PATH=\$PREFIX/bin:\$PATH LD_LIBRARY_PATH=\$PREFIX/lib HOME=/data/data/com.termux/files/home; \
export KINDI_ARM=1 KINDI_PACK=\$HOME/kindi-pack-2b PYTHONPATH=\$HOME/kindi-android-test KINDI_CPU_CACHE=\$HOME/.cache/kindi; \
export CC=\$PREFIX/bin/clang KINDI_CPU_CFLAGS=\"-O3 -march=armv8.2-a+simd -fno-math-errno\" OMP_NUM_THREADS=4; \
cd \$HOME/kindi-android-test && python3 smoke.py 2>&1 | tee smoke_${SERIAL}.txt"'

adb -s "$SERIAL" shell 'run-as com.termux sh -c "\
export PREFIX=/data/data/com.termux/files/usr PATH=\$PREFIX/bin:\$PATH LD_LIBRARY_PATH=\$PREFIX/lib HOME=/data/data/com.termux/files/home; \
export KINDI_ARM=1 KINDI_PACK=\$HOME/kindi-pack-2b PYTHONPATH=\$HOME/kindi-android-test KINDI_CPU_CACHE=\$HOME/.cache/kindi; \
export CC=\$PREFIX/bin/clang KINDI_CPU_CFLAGS=\"-O3 -march=armv8.2-a+simd -fno-math-errno\" OMP_NUM_THREADS=4; \
cd \$HOME/kindi-android-test && python3 run_verify.py 2>&1 | tee verify_clean_${SERIAL}.txt"'
```

**Pass criteria:** smoke shows `OUT_TOKENS [198, 2, 220, 16, 13, 220, 16, 13]` + `SMOKE_OK`; verify shows `prompt 1-4 PASS 40/40` and `ALL_PASS ... tok/s`.

**IF FAIL:** check `PYTHONPATH`, `KINDI_ARM=1`, `KINDI_PACK` → re-run `python3 --version`/`numpy` → check `clang --version` → check `~/.cache/kindi` writable → retry.

**USER CHECKPOINT (long run, optional 120s):** if phone throttles/kills long runs, keep Termux in foreground or allow battery = Unrestricted; keep screen awake; wait up to 120s for user to adjust, then retry/re-run.

---

## Step 5 — Optional: comprehensive tests (long-running)

```bash
# Phase 1 suite (QA, speed, write, ctx up to 2048)
python3 phone_tests.py --pack $HOME/kindi-pack-2b --ctx-max 2048 --threads 4 --out $HOME/kindi-android-test/phone_tests_${SERIAL}.json

# Incremental max-context (>2048)
python3 phone_ctx_step.py --pack $HOME/kindi-pack-2b --ctx-max 8192 --threads 4 --start 3072 --out $HOME/kindi-android-test/phone_ctx_high_${SERIAL}.json

# Full 158 textjobs (very long at ~1 tok/s)
python3 phone_textjobs.py --pack $HOME/kindi-pack-2b --ctx 2048 --threads 4 --out $HOME/kindi-android-test/phone_textjobs_full_${SERIAL}.json
```

Run long jobs with `nohup ... &` and capture logs/JSON. `phone_textjobs.py` checkpoints after every item and auto-resumes.

---

## Known environment notes

- `pkg/apt` on-device is unreliable (clock skew broke TLS/repo validity on S23 bootstrap) → **USB-pushed deb install is the fast path**.
- `/sdcard` scoped storage / `:` filenames → stage via `/data/local/tmp` + `run-as` tar/cp.
- `run-as` requires **debuggable Termux** → use the provided debug APK and open once to bootstrap.
- `dpkg-deb`/PATH issues → always export `PATH=$PREFIX/bin:$PATH` before `dpkg -i`.
- `llama.cpp b12189` cannot run Bonsai ternary `type-143` GGUF — no same-model llama.cpp baseline; Gemma 3 1B Q4_K_M NA on Android/llama.cpp is a separate issue.