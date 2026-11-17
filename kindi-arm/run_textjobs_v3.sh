#!/system/bin/sh
# run_textjobs_v3.sh — textjobs correctness gate on the FINAL v3 kernel (KINDI_I16_ROWS8)
H=/data/data/com.termux/files/home
export PREFIX=/data/data/com.termux/files/usr
export PATH=$PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$PREFIX/lib
export HOME=$H
export KINDI_DEVICE=cpu
export PYTHONPATH=$H/kindi-android-test
export KINDI_CPU_CACHE=$H/.cache/kindi
export OMP_NUM_THREADS=4
export CC=clang
export KINDI_CPU_CFLAGS="-O3 -march=armv8.2-a -fopenmp -fno-math-errno -ffp-contract=off -DKINDI_I16_ROWS8"
rm -f "$KINDI_CPU_CACHE/libkcpu_arm.so"
cd "$H/kindi-android-test" || exit 1
/system/bin/taskset f8 python3 phone_textjobs.py --pack "$H/kindi-pack-2b" --out "$H/textjobs_v3.json" --threads 4 --ctx 2048
echo "TEXTCOMPUTE rc=$?"
md5sum "$KINDI_CPU_CACHE/libkcpu_arm.so"