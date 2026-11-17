#!/usr/bin/env bash
# Post-Kindi-test queue runner for ARM64 Android devices.
#
# Waits for in-context/phase-1 runs to finish on each device, then launches
# the full 158-item kbench textjobs suite sequentially per device, in parallel
# across devices. Results are checkpointed after every item (see phone_textjobs.py).
#
# Device serials are placeholders — replace with your actual adb serials.
# Usage:
#   ./kindi_post_queue.sh
#
set -u
P1="<SERIAL_S23_1>"      # e.g. RFCX71B5C9K
P2="<SERIAL_S23_2>"      # e.g. RFCX9011K4E
P3="<SERIAL_S24>"        # e.g. RFCX9076A9H
LOG="/tmp/opencode/kindi_post_queue.log"
echo "[$(date -Is)] queue start" >> "$LOG"

wait_pid() {
  local serial="$1" pid="$2" name="$3"
  echo "[$(date -Is)] waiting $name serial=$serial pid=$pid" >> "$LOG"
  while adb -s "$serial" shell "ps -A 2>&1 | grep -w $pid" >/dev/null 2>&1; do
    sleep 30
  done
  echo "[$(date -Is)] finished $name" >> "$LOG"
}

run_textjobs() {
  local serial="$1" out="$2" log="$3"
  echo "[$(date -Is)] starting full textjobs on $serial -> $out" >> "$LOG"
  adb -s "$serial" shell 'run-as com.termux sh -c "export PREFIX=/data/data/com.termux/files/usr; export PATH=/data/data/com.termux/files/usr/bin:\$PATH; export LD_LIBRARY_PATH=/data/data/com.termux/files/usr/lib; export HOME=/data/data/com.termux/files/home; export KINDI_ARM=1; export KINDI_PACK=/data/data/com.termux/files/home/kindi-pack-2b; export PYTHONPATH=/data/data/com.termux/files/home/kindi-android-test; export KINDI_CPU_CACHE=/data/data/com.termux/files/home/.cache/kindi; export CC=/data/data/com.termux/files/usr/bin/clang; export KINDI_CPU_CFLAGS=\"-O3 -march=armv8.2-a+simd -fno-math-errno\"; export OMP_NUM_THREADS=4; cd /data/data/com.termux/files/home/kindi-android-test; /data/data/com.termux/files/usr/bin/python3 tools/phone_textjobs.py --pack /data/data/com.termux/files/home/kindi-pack-2b --ctx 2048 --threads 4 --out '"$out"' > '"$log"' 2>&1"' >> "$LOG" 2>&1
  echo "[$(date -Is)] textjobs done on $serial rc=$?" >> "$LOG"
}

# wait for current runs (replace PIDs with your actual in-flight PIDs)
# wait_pid "$P1" 11951 "P1_ctx_high"
# wait_pid "$P2" 31097 "P2_phase1"

# run full 158 textjobs sequentially per device in parallel across devices
# run_textjobs "$P1" "/data/data/com.termux/files/home/kindi-android-test/phone_textjobs_full_p1.json" "/data/data/com.termux/files/home/kindi-android-test/phone_textjobs_full_p1.log" &
# J1=$!
# run_textjobs "$P2" "/data/data/com.termux/files/home/kindi-android-test/phone_textjobs_full_p2.json" "/data/data/com.termux/files/home/kindi-android-test/phone_textjobs_full_p2.log" &
# J2=$!
# run_textjobs "$P3" "/data/data/com.termux/files/home/kindi-android-test/phone_textjobs_full_p3.json" "/data/data/com.termux/files/home/kindi-android-test/phone_textjobs_full_p3.log" &
# J3=$!
# wait $J1 $J2 $J3
# echo "[$(date -Is)] queue complete" >> "$LOG"
