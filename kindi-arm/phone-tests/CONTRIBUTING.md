# Contributing Phone Test Results

We welcome test results from any Android phone running Termux with the Kindi
ARM64 CPU port. Here's how to share yours.

## 1. Run the tests on your phone

Follow [`docs/ANDROID_ARM64_SETUP.md`](../../docs/ANDROID_ARM64_SETUP.md) to provision
Termux with Python + clang + the Kindi ARM64 port, then run:

```bash
# 1. Smoke + verify (acceptance oracle)
export PREFIX=/data/data/com.termux/files/usr
export PATH=$PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$PREFIX/lib
export HOME=/data/data/com.termux/files/home
export KINDI_ARM=1
export KINDI_PACK=$HOME/kindi-pack-2b
export PYTHONPATH=$HOME/kindi-android-test
export KINDI_CPU_CACHE=$HOME/.cache/kindi
export CC=$PREFIX/bin/clang
export KINDI_CPU_CFLAGS="-O3 -march=armv8.2-a+simd -fno-math-errno"
export OMP_NUM_THREADS=4
cd $HOME/kindi-android-test
python3 phone_tests.py | tee smoke_<your-serial>.txt
python3 run_verify.py | tee verify_clean_<your-serial>.txt

# 2. Phase 1 suite (QA, speed, write, max-context ≤2048)
python3 phone_tests.py --pack $HOME/kindi-pack-2b --ctx-max 2048 --threads 4 --out phone_tests_<your-serial>.json

# 3. Full 158-item textjobs (long-running, ~2–3 h at ~1 tok/s)
python3 phone_textjobs.py --pack $HOME/kindi-pack-2b --ctx 2048 --threads 4 --out phone_textjobs_full_<your-serial>.json

# 4. Incremental max-context beyond 2048 (optional, slow — ~1+ hour per step)
python3 phone_ctx_step.py --pack $HOME/kindi-pack-2b --ctx-max 8192 --threads 4 --start 3072 --out phone_ctx_high_<your-serial>.json
```

## 2. Create your device directory

Copy your result files into a new directory under `kindi-arm/phone-tests/`:

```
kindi-arm/phone-tests/
└── <model>_<your-serial>/
    ├── device-info.md        # chipset, Android version, threads, toolchain, test date
    ├── smoke_<serial>.txt    # smoke output (OUT_TOKENS oracle)
    ├── verify_clean_<serial>.txt  # reference.json 4/4 PASS + tok/s
    ├── phone_tests_<serial>.json  # phase 1 suite results
    ├── phone_textjobs_full_<serial>.json  # textjobs results
    └── phone_ctx_high_<serial>.json  # max-context >2048 (optional)
```

Use the naming pattern `<model>_<serial>` (e.g. `pixel8_GQ8888/`).

## 3. Write `device-info.md`

Copy the template from any existing device directory and fill in:
- Phone model + serial (serials are just device IDs, not secrets)
- Chipset + Android version
- Termux build (debuggable) + threads + clang version
- Test date

## 4. Submit

Push your new directory to this repo (or open a PR / sync request against
`kvasirasia/kindi-TY` branch `add/kindi-arm-android-port`), and add a row to the
results table in [`README.md`](README.md).

## What we do NOT need

- Model weights or GGUF files (too large, obtain separately)
- API keys, credentials, or any personal data
- Screenshots or photos

## What these results show

- **Correctness:** smoke oracle tokens + `reference.json` 4/4 token-exact verify
- **Speed:** prefill/decode/write tok/s per device
- **Capacity:** max-context that the engine can load and generate within
- **Task behavior:** textjobs pass/fail by category (classify strong; spell/grammar/table weak on 2B)
