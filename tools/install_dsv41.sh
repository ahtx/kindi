#!/bin/bash
# Kindi: DeepSeek-V4.1-Flash (native host install; the model is too large for the container image)
#
#   bash tools/install_dsv41.sh [--dir DIR] [--port 8101] [--ctx 65536] [--profile auto|ram|nvme] [--no-service]
#                               [--skip-download] [--dry-run] [--yes] [--force] [--plan PLAN.json]
#   --force: replace an existing kindi-dsv41.service (by default it is never overwritten)
#   --plan: the cards come from this plan file (kindi plan; install.sh writes it): the main card, the expert cards
#           and the hot-expert budget of the DeepSeek server in it, and its port. Without --plan the script makes a
#           one-server plan: card 0 main, every other card an expert card, the hot experts on all spare VRAM
#           (the main card's too). docs/dev/F6-PLACEMENT.md
#
# What it does:
#   1. checks the host: Linux x86-64 with AVX2, an NVIDIA card (sm_50 or newer, 12 GB+ recommended) with nvcc,
#      python3 + numpy, gcc, the free disk space, and the RAM;
#   2. picks the profile: "ram" (the whole model fits the page cache: 496 GiB of RAM or more, as the kernel
#      reports it in MemTotal) or "nvme"
#      (less RAM: the routed experts are streamed from a fast NVMe drive into a RAM expert cache,
#      KINDI_DSV_EXPERTS=stream; a spinning disk is refused);
#   3. downloads the UD-Q4_K_XL GGUF (9 shards, about 494 GB) and the DSpark drafter (3 checkpoint shards), with
#      resume, and checks every file against the published sha256 (tests/data/dsv41/SHA256SUMS-UD-Q4_K_XL.txt);
#   4. converts the drafter (python -m kindi.dsv41_dspark convert) and plans the hot experts for the cards
#      (tools/dsv41_hotplan.py, the shared MoE placement of kindi/moe.py);
#   5. writes a systemd user service, kindi-dsv41.service (OpenAI API on --port), and starts it.
# --dry-run prints the plan and changes nothing.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
DIR=${KINDI_DSV_DIR:-$HOME/kindi-models/deepseek-v4.1-flash}
PORT=8101 CTX=65536 PROFILE=auto SERVICE=1 DOWNLOAD=1 DRY=0 YES=0 FORCE=0 PLAN="" PORT_SET=0
while [ $# -gt 0 ]; do
    case "$1" in
        --dir) DIR=$2; shift 2;;
        --port) PORT=$2; PORT_SET=1; shift 2;;
        --plan) PLAN=$2; shift 2;;
        --ctx) CTX=$2; shift 2;;
        --profile) PROFILE=$2; shift 2;;
        --no-service) SERVICE=0; shift;;
        --skip-download) DOWNLOAD=0; shift;;
        --dry-run) DRY=1; shift;;
        --yes|-y) YES=1; shift;;
        --force) FORCE=1; shift;;
        -h|--help) sed -n 2,25p "$0"; exit 0;;
        *) echo "unknown option: $1" >&2; exit 2;;
    esac
done
GGUF_BASE=https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/resolve/main/UD-Q4_K_XL
DSPARK_BASE=https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main
SUMS=$HERE/tests/data/dsv41/SHA256SUMS-UD-Q4_K_XL.txt
DSPARK_SUMS=$HERE/tests/data/dsv41/SHA256SUMS-dspark.txt
NEED_GB=530                     # 494 GB GGUF + 20 GB drafter shards + 8 GB converted drafter + margin
# RAM the ram profile needs, in GiB of MemTotal (measured 2026-11-15 on the DeepSeek test host, ctx 81920):
#   467 = the 9 GGUF shards (466.3 GiB), all held in the page cache
#  + 11 = the server's own memory (RssAnon 10.4 GiB)
#  + 18 = kernel, slab and the rest of the OS (16.3 GiB of slab on that host), with a small margin
RAM_NEED_GB=496
say() { echo "kindi-dsv41: $*"; }
die() { echo "kindi-dsv41: ERROR: $*" >&2; exit 1; }
run() { if [ $DRY = 1 ]; then echo "  [dry-run] $*"; else "$@"; fi; }

# ---- 1. host checks
[ "$(uname -s)" = Linux ] && [ "$(uname -m)" = x86_64 ] || die "Linux x86-64 only"
grep -qw avx2 /proc/cpuinfo || die "the CPU has no AVX2 (the expert kernels need it)"
command -v python3 >/dev/null || die "python3 is missing"
python3 -c "import numpy" 2>/dev/null || die "numpy is missing: python3 -m pip install --user numpy"
command -v gcc >/dev/null || die "gcc is missing (apt install build-essential)"
command -v nvidia-smi >/dev/null || die "no NVIDIA driver (nvidia-smi); DeepSeek-V4.1-Flash needs an NVIDIA card"
NVCC=${KINDI_NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}
[ -x "$NVCC" ] || die "nvcc not found (install the CUDA toolkit, or set KINDI_NVCC)"
CARDS=$(nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader)
say "cards:"; echo "$CARDS" | sed 's/^/    /'
VRAM0=$(nvidia-smi -i 0 --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' ')
[ "${VRAM0:-0}" -ge 11000 ] || say "warning: card 0 has ${VRAM0} MiB; 12 GB or more is recommended (dense layers + drafter)"
RAM_GB=$(awk '/MemTotal/ {printf "%d", $2 / 1048576}' /proc/meminfo)
mkdir -p "$DIR" 2>/dev/null || true
FREE_GB=$(df -BG --output=avail "$DIR" 2>/dev/null | tail -1 | tr -dc 0-9)
DEV=$(df --output=source "$DIR" | tail -1)
ROTA=$(lsblk -no ROTA "$(lsblk -no PKNAME "$DEV" 2>/dev/null | head -1 | sed 's#^#/dev/#')" 2>/dev/null | head -1 || echo "?")
[ -z "$ROTA" ] && ROTA=$(lsblk -no ROTA "$DEV" 2>/dev/null | head -1)
say "RAM ${RAM_GB} GiB; ${DIR}: ${FREE_GB:-?} GB free on ${DEV} ($( [ "${ROTA:-1}" = 0 ] && echo SSD/NVMe || echo "spinning disk or unknown"))"

# ---- 2. profile
if [ "$PROFILE" = auto ]; then
    if [ "$RAM_GB" -ge $RAM_NEED_GB ]; then PROFILE=ram; else PROFILE=nvme; fi
    say "${RAM_GB} GiB of RAM found, the ram profile needs ${RAM_NEED_GB} GiB: profile ${PROFILE} (--profile ram|nvme overrides)"
elif [ "$PROFILE" = ram ] && [ "$RAM_GB" -lt $RAM_NEED_GB ]; then
    say "warning: ${RAM_GB} GiB of RAM found, the ram profile needs ${RAM_NEED_GB} GiB; part of the model will be read from ${DEV} again and again"
fi
case "$PROFILE" in
    ram)  say "profile ram: the model stays in the page cache (NUMA-placed); expect about 18 tok/s write, 100 tok/s read on a 2-socket DDR4 host";;
    nvme) [ "${ROTA:-1}" = 0 ] || die "profile nvme needs the model on an SSD/NVMe drive; ${DEV} is a spinning disk (or unknown). Use --dir on an NVMe drive, or a host with ${RAM_NEED_GB} GiB+ of RAM"
          say "profile nvme: the experts are read from ${DEV}; speed follows the drive (see docs/DEEPSEEK.md)";;
    *) die "--profile must be auto, ram or nvme";;
esac
[ $DOWNLOAD = 0 ] || [ "${FREE_GB:-0}" -ge $NEED_GB ] || die "need about ${NEED_GB} GB free in ${DIR}, have ${FREE_GB:-?} GB"
if [ $YES = 0 ] && [ $DRY = 0 ] && [ $DOWNLOAD = 1 ]; then
    read -r -p "Download about 514 GB into ${DIR}? [y/N] " ok
    [ "$ok" = y ] || [ "$ok" = Y ] || die "stopped"
fi

# ---- 3. downloads, with resume and sha256 checks
fetch() {   # URL FILE SHA256
    if [ -f "$2" ] && [ "$(sha256sum "$2" | cut -d' ' -f1)" = "$3" ]; then say "ok   $(basename "$2")"; return; fi
    run curl -fL --retry 10 --retry-delay 5 -C - -o "$2" "$1"
    [ $DRY = 1 ] || [ "$(sha256sum "$2" | cut -d' ' -f1)" = "$3" ] || die "sha256 mismatch: $2 (delete it and run again)"
    say "got  $(basename "$2")"
}
if [ $DOWNLOAD = 1 ]; then
    mkdir -p "$DIR/gguf" "$DIR/dspark"
    while read -r sha name _; do
        [ -n "$name" ] && fetch "$GGUF_BASE/$name" "$DIR/gguf/$name" "$sha"
    done < "$SUMS"
    run curl -fsSL -o "$DIR/dspark/config.json" "$DSPARK_BASE/config.json"
    run curl -fsSL -o "$DIR/dspark/model.safetensors.index.json" "$DSPARK_BASE/model.safetensors.index.json"
    while read -r sha name; do
        [ -n "$name" ] && fetch "$DSPARK_BASE/$name" "$DIR/dspark/$name" "$sha"
    done < "$DSPARK_SUMS"
fi
# the chat encoding of the model release (MIT): the server formats prompts and parses tool calls with it
[ $DRY = 1 ] || mkdir -p "$DIR/dspark"
fetch "$DSPARK_BASE/encoding/encoding.py" "$DIR/dspark/encoding.py" 13231c592e958fb39e2af6001dae69d90e7a36b265702b9b390236fb387599e2
MODEL=$DIR/gguf/DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00001-of-00009.gguf

# ---- 4. drafter + hot experts
if [ ! -f "$DIR/dspark/kindi-dspark.bin" ]; then
    run env PYTHONPATH="$HERE" python3 -m kindi.dsv41_dspark convert "$DIR/dspark"
fi
# the cards: the plan's DeepSeek server (main card first in CUDA_VISIBLE_DEVICES, the expert cards, the hot-expert
# budget over all of them: the main card's spare VRAM too). Without --plan: card 0 main, every other card an expert
# card, as before 0.7.0, but with the main card's spare VRAM in the budget (0.6.0 set KINDI_HOT_MAX=0)
if [ -z "$PLAN" ]; then
    PLAN=$DIR/plan.json
    run env PYTHONPATH="$HERE" python3 -m kindi.place --place "dsv41:0+" --out "$PLAN" || die "kindi plan: card 0 cannot be DeepSeek's main card (see above)"
fi
if [ $DRY = 1 ] && [ ! -f "$PLAN" ]; then
    PLAN_ENV="KINDI_DSV_GPU=2"; HOT_GB=0; say "[dry-run] the plan is not written, so the cards are not shown"
else
    [ -f "$PLAN" ] || die "no plan file $PLAN"
    if ! PYTHONPATH="$HERE" python3 -m kindi.place --check "$PLAN" >/dev/null; then
        PYTHONPATH="$HERE" python3 -m kindi.place --check "$PLAN" | sed 's/^/    /' || true
        # --dry-run while a DeepSeek server holds the cards (an upgrade check): their free memory is the server's
        [ $DRY = 1 ] && say "[dry-run] the plan does not fit the cards' free memory now (is a server using them?); a real run stops here" \
            || die "the plan $PLAN does not fit the cards here (kindi plan again)"
    fi
    read -r PLAN_ENV HOT_GB PLAN_PORT < <(python3 -c '
import json, sys
p = json.load(open(sys.argv[1]))
s = [x for x in p["servers"] if x["engine"] == "dsv41"]
if s:
    print("%s %s %s" % (",".join("%s=%s" % kv for kv in sorted(s[0]["env"].items())), s[0]["hot_gb_total"], s[0]["port"]))
' "$PLAN")
    [ -n "${PLAN_ENV:-}" ] || die "the plan $PLAN has no DeepSeek server"
    PLAN_ENV=${PLAN_ENV//,KINDI_/ KINDI_}
    [ $PORT_SET = 1 ] || PORT=$PLAN_PORT
    say "cards from the plan $PLAN: $PLAN_ENV; hot experts on ${HOT_GB} GB of VRAM"
fi
if [ -f "$HERE/tests/data/dsv41/hot-counts.txt" ]; then
    run python3 "$HERE/tools/dsv41_hotplan.py" "$DIR/hot.txt" --card-gb "$HOT_GB" --list "$HERE/tests/data/dsv41/hot-counts.txt"
fi

# ---- 5. service
ENVS="$PLAN_ENV KINDI_SPLIT=1 KINDI_SPLIT_PF=0 PYTHONUNBUFFERED=1"
[ -f "$DIR/hot.txt" ] && ENVS="$ENVS KINDI_HOT_LIST=$DIR/hot.txt"
NODES=$(ls -d /sys/devices/system/node/node[0-9]* 2>/dev/null | wc -l)
PRE=""
if [ "$PROFILE" = ram ] && [ "$NODES" -ge 2 ]; then
    PRE="ExecStartPre=/bin/sh -c 'cd $HERE && python3 -m kindi.dsv41 numa $MODEL --rowsplit'"
    ENVS="$ENVS KINDI_MOE_SHARE0=0.5 KINDI_DSV_SHARE0=0.667"
fi
if [ "$PROFILE" = nvme ]; then
    # the expert cache (xstore.c): 70 % of the available RAM less 16 GB (the dense weights, about 9 GB, and the page
    # cache of the 203 GB engram table, which is read at random, a few rows per token); at least 8 GB (one layer of
    # experts is 7 GB). The most used experts that the cards do not hold are pinned (half the cache, KINDI_XS_PIN).
    AVAIL_GB=$(awk '/MemAvailable/ {printf "%d", $2 / 1048576}' /proc/meminfo)
    XS_GB=$(( AVAIL_GB * 70 / 100 - 16 ))
    [ "$XS_GB" -ge 8 ] || XS_GB=8
    ENVS="$ENVS KINDI_DSV_EXPERTS=stream KINDI_XS_GB=$XS_GB KINDI_XS_DROP=1"
    say "nvme: an expert cache of ${XS_GB} GB (of ${AVAIL_GB} GB available); the experts are read with direct I/O"
fi
UNIT=$HOME/.config/systemd/user/kindi-dsv41.service
if [ $SERVICE = 1 ] && [ -f "$UNIT" ] && [ $FORCE = 0 ]; then
    say "a kindi-dsv41.service exists already ($UNIT); it is left as it is (use --force to replace it)"
    SERVICE=0
fi
if [ $SERVICE = 1 ]; then
    say "writing $UNIT (port $PORT, ctx $CTX, profile $PROFILE)"
    if [ $DRY = 0 ]; then
        mkdir -p "$(dirname "$UNIT")"
        cat > "$UNIT" <<EOF
[Unit]
Description=Kindi DeepSeek-V4.1-Flash server (OpenAI API on :$PORT, profile $PROFILE)
After=network-online.target

[Service]
WorkingDirectory=$HERE
Environment=$ENVS
$PRE
ExecStart=/usr/bin/python3 -m kindi.dsv41_server $MODEL --host 127.0.0.1 --port $PORT --ctx $CTX --tmax 512 --dspark $DIR/dspark --encoding $DIR/dspark/encoding.py --cache-dir $DIR/cache --cache-gb 40
Restart=on-failure
TimeoutStartSec=3600

[Install]
WantedBy=default.target
EOF
        systemctl --user daemon-reload
        systemctl --user enable --now kindi-dsv41.service
    fi
    say "started; the first start compiles the kernels and reads the model (minutes to an hour on a cold disk)."
    say "check: curl -s 127.0.0.1:$PORT/health ; gate: python3 -m kindi.dsv41 verify $MODEL (needs the gate data)"
fi
say "done (profile $PROFILE)"
