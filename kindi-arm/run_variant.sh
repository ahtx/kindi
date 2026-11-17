#!/system/bin/sh
# run_variant.sh <label> [extra-cflags] [affinity-mask] — inside run-as com.termux.
# Fresh .so build from pushed kcpu_arm.c + cpu.py, recorded flags, controlled affinity.
V="$1"
EXTRA="$2"
MASK="${3:-f8}"
export PREFIX=/data/data/com.termux/files/usr
export PATH=$PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$PREFIX/lib
export HOME=/data/data/com.termux/files/home
export KINDI_DEVICE=cpu
export PYTHONPATH=$HOME/kindi-android-test
export KINDI_CPU_CACHE=$HOME/.cache/kindi
export OMP_NUM_THREADS=4
export CC=clang
export KINDI_CPU_CFLAGS="-O3 -march=armv8.2-a -fopenmp -fno-math-errno $EXTRA"
rm -f "$KINDI_CPU_CACHE/libkcpu_arm.so"
cd "$HOME/kindi-android-test" || exit 1
/system/bin/taskset "$MASK" python3 runner_kindi.py "$HOME/kindi-pack-2b" "$V" "$HOME/kindi-bench-$V.json"
rc=$?
echo "=== provenance ==="
md5sum "$KINDI_CPU_CACHE/libkcpu_arm.so" "$HOME/kindi-android-test/kindi/kcpu_arm.c" "$HOME/kindi-android-test/kindi/cpu.py"
exit $rc