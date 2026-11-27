#!/bin/bash
# Kindi: Qwen3.8-Flash-Next (native install; docs/FLASHNEXT.md). install.sh calls this for --model flashnext or a plan
# with a flashnext server; it also runs on its own.
#
#   bash tools/install_flashnext.sh [--dir DIR] [--plan PLAN.json] [--port 8102] [--no-mtp] [--no-service]
#                                   [--skip-download] [--skip-verify] [--dry-run] [--yes] [--force]
#
# What it does:
#   1. checks the host: Linux x86-64 with AVX2, gcc, python3 (3.10 or newer) with venv, the NVIDIA driver, the RAM
#      (the experts, 50 GB, stay in host RAM: 64 GB or more; with less the kernel reads them from disk) and the disk;
#   2. the cards: the plan's Flash-Next server (--plan), else kindi plan --model flashnext (the fastest card that holds
#      the dense layers, sm_61 or later, 8 GB free); the engine puts the most used experts in the VRAM left on each
#      card (sized from the free VRAM at start, kindi/qwen4exp-experts.json);
#   3. a Python environment in DIR/venv: PyTorch (CUDA 12.4 wheels), numpy, Jinja2, FastAPI, uvicorn;
#   4. downloads the model, ISTA-DASLab's Qwen3.8-Flash-Next GSQ-RCO IQ3_S GGUF (2 shards, 83.6 GB), with resume, and
#      checks each shard's sha256 against the values in this script;
#   5. the MTP head (on unless --no-mtp or KINDI_MTP=0): its 31 tensors fetched by byte range from Qwen's official
#      release (5.2 GB, not the whole repository), converted here with Kindi's importer (kindi/qwen4exp_mtp_import.py)
#      and the compact form (tools/qwen4exp_mtp_compact.py); the fetched tensors are deleted after. The head is under
#      the Qwen Community License 1.0 (the notice is in the converted file and in docs/FLASHNEXT.md);
#   6. kindi convert (the pack: the shards' sizes and fast hashes), then the gate: kindi verify, teacher-forced
#      against the black-box reference (8 prompts, 174 tokens; 173 of 174 with the one miss a near-tie);
#   7. a systemd user service, kindi-flashnext.service (Kindi's OpenAI server, kindi serve, on --port), and starts it.
# --dry-run prints the plan and changes nothing.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
DIR=${KINDI_FN_DIR:-$HOME/.kindi/flashnext}
PORT=8102 PLAN="" MTP=1 SERVICE=1 DOWNLOAD=1 VERIFY=1 DRY=0 YES=0 FORCE=0 PORT_SET=0
[ "${KINDI_MTP:-1}" = 0 ] && MTP=0
while [ $# -gt 0 ]; do
    case "$1" in
        --dir) DIR=$2; shift 2;;
        --plan) PLAN=$2; shift 2;;
        --port) PORT=$2; PORT_SET=1; shift 2;;
        --no-mtp) MTP=0; shift;;
        --no-service) SERVICE=0; shift;;
        --skip-download) DOWNLOAD=0; shift;;
        --skip-verify) VERIFY=0; shift;;
        --dry-run) DRY=1; shift;;
        --yes|-y) YES=1; shift;;
        --force) FORCE=1; shift;;
        -h|--help) sed -n 2,28p "$0"; exit 0;;
        *) echo "unknown option: $1" >&2; exit 2;;
    esac
done
REPO=https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/ed59f92082b1e93c0e96d60a8b11aab089b52f09/IQ3_S
SHARDS=("Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf" "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf")
SHAS=(4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3 316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113)
SIZES=(54817524224 28800138432)
NEED_GB=$((84 + 8))             # the shards, and the MTP head's fetched tensors while they are converted
say() { echo "kindi-flashnext: $*"; }
die() { echo "kindi-flashnext: ERROR: $*" >&2; exit 1; }
run() { if [ $DRY = 1 ]; then echo "  [dry-run] $*"; else "$@"; fi; }
# ---- 1. host checks
[ "$(uname -s)" = Linux ] && [ "$(uname -m)" = x86_64 ] || die "Linux x86-64 only"
grep -qw avx2 /proc/cpuinfo || die "the CPU has no AVX2 (the CPU expert kernels need it)"
command -v gcc >/dev/null || die "gcc is missing (apt install build-essential): the CPU expert kernels are built with it"
command -v python3 >/dev/null || die "python3 is missing"
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || die "Python 3.10 or newer is needed (found $(python3 -V 2>&1))"
python3 -c 'import venv, ensurepip' 2>/dev/null || die "the Python venv module is missing (apt install python3-venv)"
command -v nvidia-smi >/dev/null || die "no NVIDIA driver (nvidia-smi)"
RAM_GB=$(awk '/MemTotal/ {printf "%d", $2 / 1048576}' /proc/meminfo)
CORES=$(nproc)
mkdir -p "$DIR" 2>/dev/null || true
FREE_GB=$(df -BG --output=avail "$DIR" 2>/dev/null | tail -1 | tr -dc 0-9)
say "RAM ${RAM_GB} GB, ${CORES} CPU threads; ${DIR}: ${FREE_GB:-?} GB free"
if [ "$RAM_GB" -lt 64 ]; then
    say "warning: ${RAM_GB} GB of RAM: the experts that do not fit the cards (most of the 50 GB) are then read from the"
    say "         disk at each token, which is much slower than from RAM (64 GB or more is recommended)"
fi
[ $DOWNLOAD = 0 ] || [ "${FREE_GB:-0}" -ge $NEED_GB ] || die "need about ${NEED_GB} GB free in ${DIR}, have ${FREE_GB:-?} GB"
# ---- 2. the cards (the plan)
if [ -z "$PLAN" ]; then
    PLAN=$DIR/plan.json
    if [ $DRY = 1 ]; then
        PYTHONPATH="$HERE" python3 -m kindi.place --model flashnext || die "kindi plan: no card here runs Flash-Next (see above)"
    else
        PYTHONPATH="$HERE" python3 -m kindi.place --model flashnext --name kindi-flashnext --port-base "$PORT" --out "$PLAN" \
            || die "kindi plan: no card here runs Flash-Next (see above; a card of compute capability 5.0 or later with 8 GB free)"
    fi
fi
PLAN_ENV="CUDA_DEVICE_ORDER=PCI_BUS_ID KINDI_FN_CARDS=0"
if [ -f "$PLAN" ]; then
    if ! PYTHONPATH="$HERE" python3 -m kindi.place --check "$PLAN" >/dev/null; then
        PYTHONPATH="$HERE" python3 -m kindi.place --check "$PLAN" | sed 's/^/    /' || true
        [ $DRY = 1 ] && say "[dry-run] the plan does not fit the cards' free memory now; a real run stops here" \
            || die "the plan $PLAN does not fit the cards here (kindi plan again)"
    fi
    read -r PLAN_ENV PLAN_PORT < <(python3 -c '
import json, sys
p = json.load(open(sys.argv[1]))
s = [x for x in p["servers"] if x["engine"] == "flashnext"]
if s:
    print("%s %s" % (" ".join("%s=%s" % kv for kv in sorted(s[0]["env"].items())).replace(" ", "|"), s[0]["port"]))
' "$PLAN")
    [ -n "${PLAN_ENV:-}" ] || die "the plan $PLAN has no Flash-Next server"
    PLAN_ENV=${PLAN_ENV//|/ }
    [ $PORT_SET = 1 ] || PORT=$PLAN_PORT
fi
say "cards: $PLAN_ENV; port $PORT; MTP head $([ $MTP = 1 ] && echo on || echo off)"
# ---- 3. the Python environment
VENV=$DIR/venv
if [ ! -x "$VENV/bin/python" ] || [ $FORCE = 1 ]; then
    run python3 -m venv "$VENV"
    run "$VENV/bin/pip" install -q --upgrade pip
    # cards below sm_75 need a CUDA 12.x build (the CUDA 13 builds dropped them); FAKE_COMPUTE_CAP is for dry-run tests
    selection=()
    visible=$(printf '%s\n' "$PLAN_ENV" | sed -n 's/.*CUDA_VISIBLE_DEVICES=\([^ ]*\).*/\1/p')
    [ -z "$visible" ] || selection=(-i "$visible")
    CC=${FAKE_COMPUTE_CAP:-$(nvidia-smi "${selection[@]}" --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null |
        awk '$1 ~ /^[0-9]+\.[0-9]+$/ && (min == "" || $1+0 < min+0) {min=$1} END {print min}')}
    CCN=$(echo "${CC:-0}" | awk -F. '{print $1*10+$2}')
    if [ "$CCN" -gt 0 ] && [ "$CCN" -lt 75 ]; then
        TORCH_SPEC="torch==2.9.1"; TORCH_IDX=https://download.pytorch.org/whl/cu126
    else
        TORCH_SPEC="torch==2.6.0"; TORCH_IDX=https://download.pytorch.org/whl/cu124
    fi
    say "compute capability ${CC:-unknown}: $TORCH_SPEC from $TORCH_IDX"
    run "$VENV/bin/pip" install -q "$TORCH_SPEC" --index-url "$TORCH_IDX"
    run "$VENV/bin/pip" install -q "numpy<3" "Jinja2>=3.1" "fastapi>=0.100" "uvicorn>=0.23"
fi
PY=$VENV/bin/python
# ---- 4. the model, with sha256 checks
mkdir -p "$DIR/model" 2>/dev/null || true
fetch() {   # URL FILE SHA256 SIZE
    if [ -f "$2.sha256-ok" ] && [ "$(stat -c %s "$2" 2>/dev/null || echo 0)" = "$4" ]; then say "ok   $(basename "$2") (checked before)"; return; fi
    run curl -fL --retry 10 --retry-delay 5 -C - -o "$2" "$1"
    if [ $DRY = 0 ]; then
        say "checking the sha256 of $(basename "$2") ..."
        [ "$(sha256sum "$2" | cut -d' ' -f1)" = "$3" ] || die "sha256 mismatch: $2 (delete it and run again)"
        touch "$2.sha256-ok"
    fi
    say "got  $(basename "$2")"
}
if [ $DOWNLOAD = 1 ]; then
    if [ $YES = 0 ] && [ $DRY = 0 ] && [ ! -f "$DIR/model/${SHARDS[1]}.sha256-ok" ]; then
        read -r -p "Download about 84 GB into ${DIR}/model? [y/N] " ok
        [ "$ok" = y ] || [ "$ok" = Y ] || die "stopped"
    fi
    for i in 0 1; do fetch "$REPO/${SHARDS[$i]}" "$DIR/model/${SHARDS[$i]}" "${SHAS[$i]}" "${SIZES[$i]}"; done
fi
MODEL=$DIR/model/${SHARDS[0]}
# ---- 5. the MTP head: fetched by byte range from Qwen's release and converted here
if [ $MTP = 1 ] && [ ! -f "$DIR/model/qwen4exp-mtp-q2.gguf" ]; then
    say "the MTP head: 31 tensors (5.2 GB) by byte range from Qwen/Qwen3.8-Flash-Next (Qwen Community License 1.0)"
    run "$PY" "$HERE/tools/qwen4exp_mtp_fetch.py" "$DIR/mtp-src"
    run env PYTHONPATH="$HERE" "$PY" -m kindi.qwen4exp_mtp_import "$DIR/mtp-src" "$DIR/model/qwen4exp-mtp.gguf"
    run env PYTHONPATH="$HERE" "$PY" "$HERE/tools/qwen4exp_mtp_compact.py" "$DIR/model/qwen4exp-mtp.gguf" "$DIR/model/qwen4exp-mtp-q2.gguf"
    run rm -rf "$DIR/mtp-src"
fi
# ---- 6. the pack and the gate
if [ ! -f "$DIR/pack/kindi_pack.json" ] || [ $FORCE = 1 ]; then
    run env PYTHONPATH="$HERE" "$PY" -m kindi --pack "$DIR/pack" convert "$MODEL" --out "$DIR/pack" --force
fi
# the engine's settings (the engine owner's defaults, 11-16): the mailbox pass and the adaptive card cache; the
# experts' pages locked in RAM (KINDI_QXG_PIN, about 50 GB) only with RAM to spare; every physical core for the CPU
# experts; the context from the main card's free memory (the expert plan keeps room for it)
CORES_LIST=$(lscpu -p=CPU,CORE,SOCKET 2>/dev/null | grep -v '^#' | sort -t, -k3,3n -k2,2n -u | awk -F, '!seen[$3","$2]++ {print $1}' | paste -sd, - || true)
[ -n "$CORES_LIST" ] || CORES_LIST="0-$((CORES - 1))"
MAIN=$(echo "$PLAN_ENV" | tr ' ' '\n' | sed -n 's/^CUDA_VISIBLE_DEVICES=\([^,]*\).*/\1/p')
FREE_MIB=$(nvidia-smi ${MAIN:+-i "$MAIN"} --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc 0-9 || true)
if [ "${FREE_MIB:-0}" -ge 11000 ]; then LMAX=16384; else LMAX=8192; fi
ENG_ENV="KINDI_QXG_PASS=1 KINDI_QXG_ADAPT=1 KINDI_QXG_LMAX=$LMAX KINDI_FN_CPUS=$CORES_LIST"
if [ "$RAM_GB" -ge 96 ]; then ENG_ENV="$ENG_ENV KINDI_QXG_PIN=1"; fi
if [ -f "$HOME/.config/systemd/user/kindi-dsv41.service" ]; then
    # a host that also serves DeepSeek: no nibble copies and no locked pages (they take the page cache DeepSeek uses)
    ENG_ENV="$ENG_ENV KINDI_QX_NIB=0"; ENG_ENV=${ENG_ENV/ KINDI_QXG_PIN=1/}
    say "kindi-dsv41.service is on this host: Flash-Next keeps off its page cache (KINDI_QX_NIB=0, no locked pages)"
fi
say "engine: $ENG_ENV"
SVC_ENV="$PLAN_ENV $ENG_ENV KINDI_PACK=$DIR/pack KINDI_STATE_DIR=$DIR/state KINDI_CPU_CACHE=$DIR/cpu KINDI_MODEL_NAME=qwen38-flash-next PYTHONUNBUFFERED=1"
[ $MTP = 1 ] || SVC_ENV="$SVC_ENV KINDI_MTP=0"
if [ $VERIFY = 1 ]; then
    say "the gate: kindi verify (8 prompts, 174 tokens, teacher-forced; minutes)"
    run env $SVC_ENV PYTHONPATH="$HERE" "$PY" -m kindi verify || die "kindi verify did not pass: no service is installed"
fi
# ---- 7. the service
UNIT=$HOME/.config/systemd/user/kindi-flashnext.service
if [ $SERVICE = 1 ] && [ -f "$UNIT" ] && [ $FORCE = 0 ]; then
    say "a kindi-flashnext.service exists already ($UNIT); it is left as it is (use --force to replace it)"
    SERVICE=0
fi
if [ $SERVICE = 1 ]; then
    say "writing $UNIT (port $PORT)"
    if [ $DRY = 0 ]; then
        mkdir -p "$(dirname "$UNIT")"
        cat > "$UNIT" <<EOF
[Unit]
Description=Kindi Qwen3.8-Flash-Next server (OpenAI API on 127.0.0.1:$PORT)
After=network-online.target
[Service]
WorkingDirectory=$HERE
Environment=$SVC_ENV PYTHONPATH=$HERE
ExecStart=$PY -m kindi serve --host 127.0.0.1 --port $PORT
Restart=on-failure
TimeoutStartSec=1800
[Install]
WantedBy=default.target
EOF
        systemctl --user daemon-reload
        systemctl --user enable --now kindi-flashnext.service
    fi
    say "started; the first start reads the model (minutes on a cold disk). check: curl -s 127.0.0.1:$PORT/health"
fi
say "done"
