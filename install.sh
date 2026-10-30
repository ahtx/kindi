#!/usr/bin/env bash
# Kindi installer: one command from a bare Linux machine to a verified, running Kindi server.
#
#   curl -fsSL https://amirhusain.com/kindi/install.sh | bash             # install (or upgrade / repair)
#   curl -fsSL https://amirhusain.com/kindi/install.sh | bash -s -- --dry-run
#   bash install.sh --uninstall [--purge]
#
# What it does, in order (every step prints what it is about to do and why):
#   1. checks the machine: Linux x86-64, RAM, disk, every NVIDIA and AMD GPU (or none: the CPU engine);
#   2. checks Docker and GPU access: the NVIDIA container toolkit, or the AMD compute driver (/dev/kfd); offers to
#      install what is missing (sudo only for that). AMD GPUs need no ROCm on the host: ROCm is in the image;
#   3. lists every setup that Kindi supports on this machine, the recommended one first, and asks which one
#      (--auto takes the recommended one without a question);
#   4. builds the Kindi image from the release source (CUDA, or ROCm for AMD GPUs);
#   5. downloads the model GGUF from Prism ML on Hugging Face and checks its sha256;
#   6. converts it (kindi convert) and checks the output against llama.cpp's reference (kindi verify);
#   7. starts the server as a service (Docker, restart unless-stopped) on 127.0.0.1 and prints how to use it.
# Run it again to upgrade or repair; the downloads and the converted model are kept. It never asks for or stores a
# password or a key. Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only.
set -euo pipefail

KINDI_VERSION="0.3.0"
SOURCE_URL="https://github.com/ahtx/kindi/archive/refs/tags/v${KINDI_VERSION}.tar.gz"
HF="https://huggingface.co/prism-ml"
# model: GGUF file, sha256 (= the Hugging Face file hash), size in bytes
declare -A GGUF SHA SIZE
GGUF[2b]="Ternary-Bonsai-2-2B-PTQ1_0.gguf";  SHA[2b]="36061ad406df8a4903a6db005be350c50fabfaf717f9ef0acf4b76adcd9929fa"; SIZE[2b]=429183456
GGUF[4b]="Ternary-Bonsai-2-4B-PTQ1_0.gguf";  SHA[4b]="56711263ec9a9ffeaa74a3a0e58dece8127cf9268404746c721113ca76994d60"; SIZE[4b]=949549920
GGUF[9b]="Ternary-Bonsai-2-9B-PTQ1_0.gguf";  SHA[9b]="3561df7ac7d26a71ffa991ac2db740b10841c3a350aa2676ccd3ff5d428b4e8b"; SIZE[9b]=1997486976
GGUF[27b]="Ternary-Bonsai-2-27B-PTQ1_0.gguf"; SHA[27b]="53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3"; SIZE[27b]=5946648928
# NVIDIA Nemotron 3.5 Bolt E4B (Gemma 4), from unsloth's GGUF repository
GGUF[nemotron]="NVIDIA-Nemotron-3.5-Bolt-E4B-UD-Q3_K_XL.gguf"; SHA[nemotron]="3f8032b9d2c18536c4ac51b97959a563cc27434de99823640c98147edc62ce9b"; SIZE[nemotron]=4219133120
declare -A URL
for p in 2b 4b 9b 27b; do URL[$p]="$HF/Ternary-Bonsai-2-${p^^}-gguf/resolve/main/${GGUF[$p]}"; done
URL[nemotron]="https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF/resolve/main/${GGUF[nemotron]}"

DIR="${KINDI_HOME:-$HOME/.kindi}"
PORT=8093
NAME=kindi
MODEL=auto
CTX=auto
GPU=auto
SOURCE=""
DRY=0
YES=0
MODE=install
PURGE=0
THREADS=""
MEMORY=""
AUTO=0            # --auto: take the recommended setup and answer yes (never for kernel changes)
CHOICE=""         # --config N: setup N of the list

usage() {
  cat <<EOF
Kindi ${KINDI_VERSION} installer

  install.sh [options]          install, upgrade or repair Kindi
  install.sh --uninstall        remove the Kindi service and image (keeps the models in ${DIR})
  install.sh --uninstall --purge   also remove ${DIR}

Options:
  --dry-run        print every step and command, change nothing
  --yes, -y        answer yes to every question (for scripts); takes the recommended setup
  --auto           choose the recommended setup and continue without questions (kernel changes are only
                   described, never made)
  --config N       take setup N of the list that the installer prints (see --dry-run); N,M runs several setups
                   side by side, one server each (for example a 27B on each of two cards); NxK runs setup N on
                   K cards of that kind. Each server gets its own container (kindi, kindi-2, ...) and port.
  --model M        2b | 4b | 9b | 27b | nemotron (default: the recommended model for the card)
  --ctx N          context length in tokens (default: the profile's; the engine fits it to the free memory)
  --gpu G          N (one NVIDIA card), N,M (two cards), amd or amdN (an AMD card), all, or cpu (default: choose)
  --port P         server port on 127.0.0.1 (default ${PORT})
  --dir D          data folder (default ${DIR})
  --name N         container name (default ${NAME})
  --threads N      CPU engine: the number of threads (default: the cores of one socket)
  --memory M       the container memory limit, for example 8g (default: the profile's)
  --source S       Kindi source: a folder or a .tar.gz URL/path (default: the ${KINDI_VERSION} release)
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --yes|-y) YES=1 ;;
    --auto) AUTO=1; YES=1 ;;
    --config) CHOICE="$2"; shift ;;
    --uninstall) MODE=uninstall ;;
    --purge) PURGE=1 ;;
    --model) MODEL="$2"; shift ;;
    --ctx) CTX="$2"; shift ;;
    --gpu) GPU="$2"; shift ;;
    --port) PORT="$2"; shift ;;
    --dir) DIR="$2"; shift ;;
    --name) NAME="$2"; shift ;;
    --source) SOURCE="$2"; shift ;;
    --threads) THREADS="$2"; shift ;;
    --memory) MEMORY="$2"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1 (see --help)"; exit 2 ;;
  esac
  shift
done

# ------------------------------------------------------------------------------------------------ helpers
B=$(printf '\033[1m'); G=$(printf '\033[32m'); Y=$(printf '\033[33m'); R=$(printf '\033[31m'); N=$(printf '\033[0m')
[ -t 1 ] || { B=""; G=""; Y=""; R=""; N=""; }
STEP=0
step() { STEP=$((STEP + 1)); echo; echo "${B}[$STEP] $1${N}"; [ -n "${2:-}" ] && echo "    $2"; return 0; }
info() { echo "    $*"; }
ok() { echo "    ${G}ok${N}  $*"; }
warn() { echo "    ${Y}note${N} $*"; }
die() { echo; echo "${R}stopped:${N} $*"; exit 1; }
exec 3>&1         # run() prints to the terminal also when the command's output goes to /dev/null
run() {   # print, then run unless --dry-run
  echo "    \$ $*" >&3
  [ "$DRY" = 1 ] || "$@"
}
ask() {   # ask "question" -> 0 for yes
  [ "$YES" = 1 ] && { info "$1 yes (--yes)"; return 0; }
  [ "$DRY" = 1 ] && { info "$1 (dry run: no)"; return 1; }
  local a=""
  if [ -r /dev/tty ]; then read -r -p "    $1 [y/N] " a < /dev/tty || true; else info "$1 (no terminal: no)"; fi
  [ "$a" = y ] || [ "$a" = Y ] || [ "$a" = yes ]
}
SUDO=""
need_sudo() {
  [ "$(id -u)" = 0 ] && { SUDO=""; return 0; }
  command -v sudo >/dev/null || die "this step needs root, and sudo is not installed"
  SUDO="sudo"
}
human() { awk -v b="$1" 'BEGIN { split("B KB MB GB TB", u); i = 1; while (b >= 1024 && i < 5) { b /= 1024; i++ } printf "%.1f %s", b, u[i] }'; }

DOCKER="docker"
STATE="$DIR/install.json"

# ------------------------------------------------------------------------------------------------ uninstall
if [ "$MODE" = uninstall ]; then
  step "Remove the Kindi servers" "Stops and removes this installer's containers. The models in ${DIR} stay unless --purge."
  command -v docker >/dev/null || die "Docker is not installed: nothing to remove"
  docker info >/dev/null 2>&1 || DOCKER="sudo docker"
  names="$NAME"; imgs="kindi:${KINDI_VERSION}"   # only what this installer made (other Kindi containers and images stay)
  if [ -f "$STATE" ]; then
    names=$(sed -n 's/.*"container": "\([^"]*\)".*/\1/p' "$STATE" | tr '\n' ' ')
    imgs=$(sed -n 's/.*"image": "\([^"]*\)".*/\1/p' "$STATE" | sort -u | tr '\n' ' ')
  fi
  for c in $names; do
    if $DOCKER ps -a --format '{{.Names}}' | grep -x "$c" >/dev/null; then run $DOCKER rm -f "$c" >/dev/null; ok "container ${c} removed"
    else info "no container named ${c}"; fi
  done
  for img in $imgs; do
    if $DOCKER image inspect "$img" >/dev/null 2>&1 && [ -z "$($DOCKER ps -a --filter ancestor="$img" --format '{{.Names}}')" ] \
       && ask "Remove the image ${img}?"; then if run $DOCKER rmi "$img" >/dev/null 2>&1; then ok "image removed"; else warn "could not remove the image ${img}"; fi; fi
  done
  if [ "$PURGE" = 1 ] && [ -d "$DIR" ]; then
    info "${DIR} holds $(du -sh "$DIR" 2>/dev/null | cut -f1): the models, the converted packs and the cache."
    if ask "Delete ${DIR}?"; then run rm -rf "$DIR"; ok "deleted"; fi
  fi
  echo; echo "${G}Kindi is removed.${N}"
  exit 0
fi

echo "${B}Kindi ${KINDI_VERSION} installer${N}$([ "$DRY" = 1 ] && echo " (dry run: nothing changes)")"
PREV=""
[ -f "$STATE" ] && PREV=$(cat "$STATE") && info "found an earlier install (${STATE}): this run upgrades or repairs it"

# ------------------------------------------------------------------------------------------------ 1. machine
step "Check the machine" "Kindi runs on 64-bit Linux: on an NVIDIA GPU from the Quadro K420 (Kepler) up, on an AMD RDNA3 GPU, or on the CPU."
[ "$(uname -s)" = Linux ] || die "Kindi needs Linux (found $(uname -s))"
[ "$(uname -m)" = x86_64 ] || die "Kindi needs an x86-64 CPU (found $(uname -m))"
OS_ID=$(. /etc/os-release 2>/dev/null && echo "${ID:-unknown}") || OS_ID=unknown
RAM_MB=$(awk '/MemTotal/ {print int($2 / 1024)}' /proc/meminfo)
AVAIL_MB=$(awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo)
CORES=$(nproc)
[ "$DRY" = 1 ] || mkdir -p "$DIR" 2>/dev/null || die "cannot create ${DIR}"
d="$DIR"; while [ ! -d "$d" ]; do d=$(dirname "$d"); done      # the nearest folder that exists (a dry run makes none)
FREE_DISK=$(df -Pk "$d" | awk 'NR == 2 {print $4 * 1024}')
ok "Linux ($OS_ID), ${CORES} CPU threads, $((RAM_MB / 1024)) GB RAM, $(human "$FREE_DISK") free for ${DIR}"

# GPUs: index, name, memory (MiB), compute capability
GIDX=(); GNAME=(); GMEM=(); GCC=()
DRIVER=""
if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
  DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sed -n 1p)
  if out=$(nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader,nounits 2>/dev/null); then :; else
    out=$(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits | sed 's/$/, ?/'); fi
  while IFS=',' read -r i n m c; do
    n=$(echo "$n" | sed 's/^ *//; s/ *$//'); c=$(echo "$c" | tr -d ' ')
    if [ "$c" = "?" ] || [ -z "$c" ]; then   # older drivers (Kepler's 470) do not report it: from the name
      case "$n" in
        *K6000*|*K5200*|*K40*|*"GTX TITAN"*|*"GTX 780"*) c=3.5 ;;                      # GK110
        *K80*) c=3.7 ;;                                                                # GK210
        *K420*|*K600*|*K2000*|*K4000*|*K4200*|*K5000*|*"GTX 6"*|*"GTX 7"[0-4]*|*"GT 7"*) c=3.0 ;;
        *K620*|*K1200*|*K2200*|*"GTX 75"*|*"GTX 9"*|*M[0-9]*) c=5.0 ;;
        *"GTX 10"*|*"Quadro P"*|*"TITAN X"*|*Tesla\ P*) c=6.1 ;;
        *"RTX 20"*|*"GTX 16"*|*"Quadro RTX"*|*T4*) c=7.5 ;;
        *"RTX 30"*|*A[0-9]*) c=8.6 ;;
        *"RTX 40"*|*L4*) c=8.9 ;;
        *) c=6.1 ;;
      esac
    fi
    GIDX+=("$(echo "$i" | tr -d ' ')"); GNAME+=("$n"); GMEM+=("$(echo "$m" | tr -d ' ')"); GCC+=("$c")
  done <<< "$out"
fi
NG=${#GIDX[@]}
if [ "$NG" = 0 ]; then
  info "no NVIDIA GPU found (or no NVIDIA driver)"
else
  for k in $(seq 0 $((NG - 1))); do ok "GPU ${GIDX[$k]}: ${GNAME[$k]}, ${GMEM[$k]} MiB, sm_${GCC[$k]/./}"; done
  info "NVIDIA driver ${DRIVER}"
  case "${GCC[0]}" in 3.*) [ "${DRIVER%%.*}" -le 470 ] || warn "Kepler cards need the NVIDIA 470 driver branch (found ${DRIVER})" ;; esac
fi

# AMD GPUs: the ROCm compute driver (amdgpu, part of the Linux kernel) lists each GPU in the KFD topology
ANAME=(); AGFX=(); AMEM=(); ABUS=()
for p in /sys/class/kfd/kfd/topology/nodes/*/properties; do
  [ -r "$p" ] || continue
  g=$(awk '$1 == "gfx_target_version" {print $2}' "$p"); [ "${g:-0}" -gt 0 ] || continue
  gfx=$(printf 'gfx%d%d%x' $((g / 10000)) $((g / 100 % 100)) $((g % 100)))
  r=$(awk '$1 == "drm_render_minor" {print $2}' "$p"); dv="/sys/class/drm/renderD${r}/device"
  bus=$(basename "$(readlink -f "$dv")")
  vram=$(( $(cat "$dv/mem_info_vram_total" 2>/dev/null || echo 0) / 1048576 ))
  gtt=$(( $(cat "$dv/mem_info_gtt_total" 2>/dev/null || echo 0) / 1048576 ))
  name=$(lspci -s "$bus" 2>/dev/null | sed 's/^[^ ]* [^:]*: //; s/ (rev [0-9a-f]*)$//; s/Advanced Micro Devices, Inc. //; s/\[AMD\/ATI\] //')
  [ "$gfx" = gfx1103 ] && name="Radeon 780M/760M, ${name:-Phoenix}"
  case "$gfx" in
    gfx110[0-3]) ;;
    *) warn "AMD GPU ${bus} (${name:-$gfx}, ${gfx}): Kindi supports AMD RDNA3 GPUs (gfx1100-gfx1103) only"; continue ;;
  esac
  # an APU (the 780M) has a small carve-out and uses system memory (GTT): count the larger one
  m=$vram; [ "$gtt" -gt "$m" ] && [ "$vram" -lt 8192 ] && m=$gtt
  ANAME+=("${name:-AMD $gfx}"); AGFX+=("$gfx"); AMEM+=("$m"); ABUS+=("$bus")
done
NA=${#AGFX[@]}
for k in $(seq 0 $((NA - 1))); do
  ok "AMD GPU ${k}: ${ANAME[$k]}, ${AGFX[$k]}, ${AMEM[$k]} MiB usable$([ "${AGFX[$k]}" != gfx1103 ] && echo " (not yet measured with Kindi; the Radeon 780M is)")"
done
if [ "$NA" = 0 ] && lspci -n 2>/dev/null | grep -E ' 03(00|80): 1002:' >/dev/null; then
  warn "an AMD GPU is in this machine, but its compute driver is not active (no /sys/class/kfd): Kindi cannot use it"
  info "the amdgpu kernel driver ships with Linux; a kernel 6.2 or newer and the package linux-firmware are needed"
fi

# GeForce RTX 30 cards (GA102/GA103/GA104/GA106) run FP16/BF16 matrix math at half rate. The third-party gpu-unlock
# module (https://github.com/Zagitta/gpu-unlock, GPL-2.0) lifts the limit at boot. Kindi does not include it: the
# installer only says whether each such card is unlocked and, if asked, runs that project's own installer.
UNLOCK_REPO="https://github.com/Zagitta/gpu-unlock"; UNLOCK_REV="af28cbb"     # the version that was reviewed
UNLOCK_IDS=" 2200 2203 2204 2205 2206 2207 2208 220a 220d 2216 222b 222f 2414 2480 2482 2483 2484 2486 2487 2488 2489
 248a 248c 248d 248e 249c 249d 249f 24a0 24a4 24ac 24ad 24af 24bf 24c0 24c7 24c8 24c9 24dc 24dd 24df 24e0 2501 2503
 2504 2505 2507 2508 2509 252f 2544 "
unlock_targets() { cat /sys/module/gpu_unlock/parameters/targets 2>/dev/null || true; }
check_unlock() {
  [ "$NG" -gt 0 ] || return 0
  local locked=() name bus dev t
  t=$(unlock_targets)
  while IFS=',' read -r i bus dev name; do
    bus=$(echo "$bus" | tr -d ' ' | tr 'A-F' 'a-f'); bus=${bus: -12}           # 00000000:08:00.0 -> 0000:08:00.0
    dev=$(echo "$dev" | tr -d ' ' | tr 'A-F' 'a-f'); dev=${dev:2:4}             # 0x220610de -> 2206
    case "$UNLOCK_IDS" in *" $dev "*|*$'\n'"$dev "*) ;; *) continue ;; esac
    if [ "$t" = all ] || { [ -n "$t" ] && case ",$t," in *",$bus,"*) true ;; *) false ;; esac; }; then
      ok "GPU $(echo "$i" | tr -d ' '): unlocked by gpu-unlock (FP16/BF16 matrix math at full rate)"
    else
      locked+=("GPU $(echo "$i" | tr -d ' ') ($(echo "$name" | sed 's/^ *//'), $bus)")
    fi
  done < <(nvidia-smi --query-gpu=index,pci.bus_id,pci.device_id,name --format=csv,noheader 2>/dev/null)
  [ "${#locked[@]}" -gt 0 ] || return 0
  for g in "${locked[@]}"; do warn "$g is not unlocked."; done
  info "On GeForce RTX 30 cards NVIDIA limits FP16/BF16 matrix math to half rate. gpu-unlock, a third-party"
  info "kernel module (${UNLOCK_REPO}), removes this limit at boot. Its users report 56-81% more FP16/BF16"
  info "matrix throughput on these cards, so they can run up to about 70% faster on matrix-heavy work (prompt"
  info "reading). Token writing is limited by memory speed and changes little. Kindi does not include gpu-unlock."
  info "It adds a kernel module to the initramfs and needs a kernel option and a reboot. To undo it, remove"
  info "gpu_unlock.targets=all from the kernel command line and reboot."
  if [ ! -f /etc/default/grub ] || ! command -v update-grub >/dev/null || ! command -v apt-get >/dev/null; then
    info "To install it: git clone ${UNLOCK_REPO}; cd gpu-unlock; make install; then add gpu_unlock.targets=all"
    info "to the kernel command line and reboot (the project's README covers GRUB, systemd-boot and UKI)."
    return 0
  fi
  # never answered by --yes: this installs a kernel module
  local a=""
  [ "$AUTO" = 1 ] && { info "gpu-unlock not installed (--auto never changes the kernel; run without --auto to install it)"; return 0; }
  [ "$DRY" = 1 ] && { info "Install gpu-unlock? (dry run: no)"; return 0; }
  if [ -r /dev/tty ]; then read -r -p "    Download and install gpu-unlock ${UNLOCK_REV} now (you reboot afterwards)? [y/N] " a < /dev/tty || true; fi
  [ "$a" = y ] || [ "$a" = Y ] || [ "$a" = yes ] || { info "gpu-unlock not installed"; return 0; }
  need_sudo
  local src="$DIR/gpu-unlock"
  command -v git >/dev/null || run $SUDO apt-get install -y git
  [ -d "$src/.git" ] || run git clone --quiet "$UNLOCK_REPO" "$src" || { warn "could not download gpu-unlock"; return 0; }
  run git -C "$src" checkout --quiet "$UNLOCK_REV" || { warn "gpu-unlock ${UNLOCK_REV} not found"; return 0; }
  ( cd "$src" && run make install ) || { warn "gpu-unlock did not install (see the output above); nothing else was changed"; return 0; }
  if grep -q 'gpu_unlock.targets=' /etc/default/grub; then
    ok "the kernel option is already in /etc/default/grub"
  else
    run $SUDO cp /etc/default/grub "/etc/default/grub.kindi-$(date +%Y%m%d%H%M%S)"
    run $SUDO sed -i 's/^GRUB_CMDLINE_LINUX_DEFAULT="\(.*\)"/GRUB_CMDLINE_LINUX_DEFAULT="\1 gpu_unlock.targets=all"/' /etc/default/grub
    run $SUDO update-grub
  fi
  ok "gpu-unlock installed; reboot to unlock the card(s), then run this installer again to check"
}
check_unlock

# ------------------------------------------------------------------------------------------------ 2. docker
step "Check Docker and GPU access" "Kindi runs in a Docker container with CUDA 10.2 (NVIDIA) or ROCm 6.3 (AMD) inside, so the host needs only the driver."
if ! command -v docker >/dev/null; then
  warn "Docker is not installed"
  case "$OS_ID" in
    ubuntu|debian|linuxmint|pop)
      if ask "Install Docker (apt package docker.io) and add ${USER} to the docker group?"; then
        need_sudo; run $SUDO apt-get update -qq; run $SUDO apt-get install -y -qq docker.io
        run $SUDO usermod -aG docker "$USER"; DOCKER="$SUDO docker"
      else die "Kindi needs Docker: https://docs.docker.com/engine/install/"; fi ;;
    *) die "install Docker for ${OS_ID} (https://docs.docker.com/engine/install/), then run this again" ;;
  esac
fi
if [ "$DRY" = 0 ] && ! docker info >/dev/null 2>&1; then
  if sudo -n docker info >/dev/null 2>&1 || { need_sudo; $SUDO docker info >/dev/null 2>&1; }; then
    DOCKER="sudo docker"; warn "your user cannot use Docker yet (log out and in after 'usermod -aG docker'); using sudo docker"
  else die "Docker does not answer (is the daemon running? 'sudo systemctl start docker')"; fi
fi
ok "Docker $(docker --version 2>/dev/null | awk '{print $3}' | tr -d ,)"

GPU_MODE=none
if [ "$NG" -gt 0 ] && [ "$GPU" != cpu ]; then
  if $DOCKER info 2>/dev/null | grep -i 'runtimes.*nvidia' >/dev/null || command -v nvidia-ctk >/dev/null || command -v nvidia-container-toolkit >/dev/null; then
    GPU_MODE=toolkit; ok "NVIDIA container toolkit found: the container gets the GPUs with --gpus"
  else
    warn "the NVIDIA container toolkit is not installed"
    if [ "$OS_ID" = ubuntu ] || [ "$OS_ID" = debian ]; then
      if ask "Install the NVIDIA container toolkit (NVIDIA's apt repository)?"; then
        need_sudo
        run sh -c "curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | $SUDO gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
        run sh -c "curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | $SUDO tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null"
        run $SUDO apt-get update -qq; run $SUDO apt-get install -y -qq nvidia-container-toolkit
        run $SUDO nvidia-ctk runtime configure --runtime=docker; run $SUDO systemctl restart docker
        GPU_MODE=toolkit
      fi
    fi
    if [ "$GPU_MODE" = none ]; then
      GPU_MODE=passthrough; info "Kindi passes the GPU devices and the driver libraries to the container itself instead"
    fi
  fi
fi
if [ "$NA" -gt 0 ] && [ "$GPU" != cpu ]; then
  # ROCm itself is in the image: the host needs the amdgpu driver's /dev/kfd and /dev/dri render nodes only
  if [ -c /dev/kfd ] && ls /dev/dri/renderD* >/dev/null 2>&1; then
    ok "AMD compute devices /dev/kfd and /dev/dri found: the container gets them with --device (no ROCm install on the host)"
  else
    warn "/dev/kfd or /dev/dri/renderD* is missing: the AMD GPU cannot be used (is the amdgpu driver loaded? 'lsmod | grep amdgpu')"
    NA=0
  fi
fi

# The Radeon 780M (gfx1103) resets under long GPU work when the driver preempts a running kernel (CWSR). The kernel
# option amdgpu.cwsr_enable=0 turns that preemption off; with it, the long-prompt tests that reset the GPU passed.
check_cwsr() {
  local v; v=$(cat /sys/module/amdgpu/parameters/cwsr_enable 2>/dev/null || echo "?")
  [ "$v" = 0 ] && { ok "amdgpu.cwsr_enable=0 is set (the 780M needs it for long prompts)"; return 0; }
  warn "the kernel option amdgpu.cwsr_enable=0 is not set (cwsr_enable=${v})."
  info "Without it, the Radeon 780M can reset during long prompts (8K tokens and more), and the server restarts."
  info "The option turns off one form of GPU preemption. To undo it, remove it from the kernel command line and reboot."
  if [ ! -f /etc/default/grub ] || ! command -v update-grub >/dev/null; then
    info "To set it: add amdgpu.cwsr_enable=0 to the kernel command line and reboot."; return 0; fi
  [ "$AUTO" = 1 ] && { info "not changed (--auto never changes the kernel; run without --auto to set it)"; return 0; }
  [ "$DRY" = 1 ] && { info "Add amdgpu.cwsr_enable=0 to GRUB? (dry run: no)"; return 0; }
  local a=""   # never answered by --yes: this changes the kernel command line
  if [ -r /dev/tty ]; then read -r -p "    Add amdgpu.cwsr_enable=0 to /etc/default/grub now (you reboot afterwards)? [y/N] " a < /dev/tty || true; fi
  [ "$a" = y ] || [ "$a" = Y ] || [ "$a" = yes ] || { info "not changed"; return 0; }
  need_sudo
  if grep -q 'amdgpu.cwsr_enable=' /etc/default/grub; then ok "the option is already in /etc/default/grub (reboot to use it)"; return 0; fi
  run $SUDO cp /etc/default/grub "/etc/default/grub.kindi-$(date +%Y%m%d%H%M%S)"
  run $SUDO sed -i 's/^GRUB_CMDLINE_LINUX_DEFAULT="\(.*\)"/GRUB_CMDLINE_LINUX_DEFAULT=" amdgpu.cwsr_enable=0"/' /etc/default/grub
  run $SUDO update-grub
  ok "set; reboot to use it (Kindi works before the reboot, but long prompts can reset the GPU)"
}

# ------------------------------------------------------------------------------------------------ 3. setup
step "Choose the setup" "Every setup that Kindi supports on this machine, the recommended one first (docs/CARDS.md). One server per card, or several."
pick_card() {     # the NVIDIA GPU with the most memory (first on a tie)
  local best=0
  for k in $(seq 0 $((NG - 1))); do [ "${GMEM[$k]}" -gt "${GMEM[$best]}" ] && best=$k; done
  echo $best
}
card_k() { local k; for k in $(seq 0 $((NG - 1))); do [ "${GIDX[$k]}" = "$1" ] && { echo "$k"; return 0; }; done; return 1; }
card_class() {    # NVIDIA card position -> tc | pascal8 | maxwell | kepler
  local V=${GMEM[$1]} c=${GCC[$1]/./}
  if [ "$c" -ge 80 ] && [ "$V" -ge 9500 ]; then echo tc
  elif [ "$c" -ge 61 ] && [ "$V" -ge 7500 ]; then echo pascal8
  elif [ "$c" -lt 50 ]; then echo kepler
  else echo maxwell; fi
}
card_models() {   # NVIDIA card position -> the models that run on it, the best first
  local V=${GMEM[$1]}
  case "$(card_class "$1")" in
    tc|pascal8) echo "27b nemotron 9b 4b 2b" ;;
    kepler) if [ "$V" -ge 11000 ]; then echo "27b 9b 4b 2b"; elif [ "$V" -ge 3800 ]; then echo "9b 4b 2b"   # no Nemotron on Kepler
            elif [ "$V" -ge 1900 ]; then echo "4b 2b"; else echo "2b"; fi ;;
    *) if [ "$V" -ge 3800 ]; then echo "9b nemotron 4b 2b"; elif [ "$V" -ge 3300 ]; then echo "4b nemotron 2b"
       elif [ "$V" -ge 1900 ]; then echo "4b 2b"; else echo "2b"; fi ;;
  esac
}
pair_ok() {       # two NVIDIA card positions -> 0 if they run a model split over both (tensor parallel)
  local a=$1 b=$2
  [ "${GCC[$a]/./}" -ge 50 ] && [ "${GCC[$a]/./}" -lt 61 ] && [ "${GCC[$b]/./}" -ge 50 ] && [ "${GCC[$b]/./}" -lt 61 ] \
    && [ "${GMEM[$a]}" -ge 3800 ] && [ "${GMEM[$b]}" -ge 3800 ]
}
amd_models() {    # AMD GPU position -> the models that run on it, the best first
  local V=${AMEM[$1]} m=""
  [ "$V" -ge 12000 ] && [ "$RAM_MB" -ge 24000 ] && m="27b"
  [ "$V" -ge 6000 ] && m="$m nemotron"
  echo "$m 2b"
}
model_label() {
  case "$1" in
    27b) echo "Ternary-Bonsai-2 27B" ;;
    nemotron) echo "NVIDIA Nemotron 3.5 Bolt E4B" ;;
    *) echo "Ternary-Bonsai-2 ${1^^}" ;;
  esac
}
# A setup is one or more servers. Each server is "backend|model|cards", where cards is a card list ("0,1": one model
# split over both), "kind:0 3 5" (any free card of this kind: each pick takes the next one) or "" (the CPU).
short() { if [ "$1" = nemotron ]; then echo Nemotron; else echo "${1^^}"; fi; }
CFG=()            # the servers of each setup, separated by ";"
CFG_N=()          # the label of each setup
add_cfg() { CFG+=("$1"); CFG_N+=("$2"); }
want() { [ "$MODEL" = auto ] || [ "$MODEL" = "$1" ]; }
SINGLE=()         # the single-server setups (for the all-cards setup)
PAIR=""          # two NVIDIA cards that run one model together
if [ "$GPU" != cpu ]; then
  if [ "$NG" -gt 0 ]; then
    if [[ "$GPU" == *,* ]]; then pa=$(card_k "${GPU%%,*}") || die "no NVIDIA GPU ${GPU%%,*}"; pb=$(card_k "${GPU##*,}") || die "no NVIDIA GPU ${GPU##*,}"
    elif [ "$NG" -ge 2 ] && { [ "$GPU" = auto ] || [ "$GPU" = all ]; }; then pa=0; pb=1
    else pa=""; pb=""; fi
    PAIR=""
    if [ -n "$pa" ] && pair_ok "$pa" "$pb"; then
      PAIR="${GIDX[$pa]},${GIDX[$pb]}"
      want 27b && add_cfg "cuda|27b|$PAIR" "$(model_label 27b) on GPUs ${PAIR}, split over both cards"
      want nemotron && add_cfg "cuda|nemotron|$PAIR" "$(model_label nemotron) on GPUs ${PAIR}, split over both cards"
    elif [[ "$GPU" == *,* ]]; then die "GPUs ${GPU} cannot run one model together (two Maxwell cards of 4 GB, such as two K1200s, can)"
    fi
    if [[ "$GPU" != *,* ]] && [[ "$GPU" != amd* ]]; then
      # one card: the card with the most memory first, then each other kind of card. A setup on a kind of card
      # takes any free card of that kind, so "1,1" runs setup 1 on two such cards.
      # (with one card, grep finds nothing and exits 1: pipefail would stop the installer here)
      if [ "$GPU" = auto ] || [ "$GPU" = all ]; then best=$(pick_card); order="$best $(seq 0 $((NG - 1)) | { grep -vx "$best" || true; } | tr '\n' ' ')"
      else order=$(card_k "$GPU") || die "no NVIDIA GPU ${GPU}"; fi
      seen="|"
      for k in $order; do
        kind="${GNAME[$k]}/${GMEM[$k]}"; case "$seen" in *"|$kind|"*) continue ;; esac; seen="$seen$kind|"
        same=""; for j in $order; do [ "${GNAME[$j]}/${GMEM[$j]}" = "$kind" ] && same="$same ${GIDX[$j]}"; done; same=${same# }
        n=$(echo "$same" | wc -w)
        where="GPU ${GIDX[$k]} (${GNAME[$k]})"; [ "$n" -gt 1 ] && where="one ${GNAME[$k]} (GPUs ${same// /,}: ${n} cards)"
        first=1
        for m in $(card_models "$k"); do
          want "$m" || continue
          add_cfg "cuda|$m|kind:$same" "$(model_label "$m") on ${where}"
          [ "$first" = 1 ] && SINGLE+=("cuda|$m|kind:$same|$n") && first=0
        done
      done
    fi
  fi
  if [ "$NA" -gt 0 ] && { [ "$GPU" = auto ] || [ "$GPU" = all ] || [[ "$GPU" == amd* ]]; }; then
    k=${GPU#amd}; case "$k" in ''|auto|all) k=0 ;; esac
    [ "$k" -lt "$NA" ] || die "no AMD GPU ${k}"
    first=1
    for m in $(amd_models "$k"); do
      want "$m" || continue
      add_cfg "rocm|$m|a$k" "$(model_label "$m") on AMD GPU ${k} (${ANAME[$k]})"
      [ "$first" = 1 ] && SINGLE+=("rocm|$m|a$k|1") && first=0
    done
  fi
fi
if [ "$MODEL" = auto ] || [ "$MODEL" = 2b ]; then
  add_cfg "cpu|2b|" "$(model_label 2b) on the CPU ($CORES threads; slow, no GPU needed)"
fi
[ "${#CFG[@]}" -gt 0 ] || die "model '${MODEL}' does not run on the selected hardware (run without --model to see what does)"
# the all-cards setup: the best model on every card (a pair that runs one model together counts as one); first
# when the machine has more than one GPU that is not in such a pair
if [ "$GPU" = auto ] || [ "$GPU" = all ]; then
  srv=""; lab=""; cnt=0
  if [ -n "$PAIR" ]; then srv="${CFG[0]}"; IFS='|' read -r b m c <<< "${CFG[0]}"; lab="$(short "$m") on GPUs ${PAIR} (split)"; cnt=1; fi
  for s in ${SINGLE[@]+"${SINGLE[@]}"}; do
    IFS='|' read -r b m c n <<< "$s"
    for x in ${c#kind:}; do
      [ -n "$PAIR" ] && case ",$PAIR," in *",$x,"*) continue ;; esac
      [ "$b" = rocm ] && x=$c
      srv="${srv:+$srv;}$b|$m|$x"; cnt=$((cnt + 1))
      lab="${lab:+$lab + }$(short "$m") on $([ "$b" = rocm ] && echo "AMD GPU ${x#a}" || echo "GPU $x")"
      [ "$b" = rocm ] && break
    done
  done
  if [ "$cnt" -ge 2 ]; then CFG=("$srv" "${CFG[@]}"); CFG_N=("every GPU, one server each: ${lab}" "${CFG_N[@]}"); fi
fi

SEL=1
show_list() { for i in "${!CFG[@]}"; do info "$([ "$i" = 0 ] && echo "${G}" )$((i + 1))) ${CFG_N[$i]}$([ "$i" = 0 ] && echo " (recommended)${N}")"; done; }
if [ -n "$CHOICE" ]; then SEL=$CHOICE; show_list
elif [ "${#CFG[@]}" -gt 1 ] && [ "$AUTO" = 0 ] && [ "$YES" = 0 ] && [ "$DRY" = 0 ] && [ -r /dev/tty ]; then
  show_list
  info "One number for one server, or several for several servers: \"3,8\" runs setups 3 and 8 side by side,"
  info "\"5x2\" runs setup 5 on two cards of that kind."
  read -r -p "    Choose [1]: " SEL < /dev/tty || true; SEL=${SEL:-1}
else
  show_list
  [ "${#CFG[@]}" -gt 1 ] && info "taking setup 1$([ "$AUTO" = 1 ] && echo " (--auto)")$([ "$YES" = 1 ] && echo " (--yes)"); --config N (or N,M or NxK) chooses others"
fi

# the servers: give each its cards, a container name and a port
SV_B=(); SV_M=(); SV_C=(); SV_NAME=(); SV_PORT=(); USED="|"
port_free() { ! { ss -ltnH 2>/dev/null | awk '{print $4}' | grep -E "[:.]$1\$" >/dev/null; }; }
port_ok() {       # free, or held by one of this installer's servers (it is replaced)
  port_free "$1" && return 0
  local h; h=$($DOCKER ps --format '{{.Names}} {{.Ports}}' 2>/dev/null | awk -v p=":$1->" 'index($0, p) {print $1}')
  [ -n "$h" ] && case " $OURS $NAME " in *" $h "*) return 0 ;; esac
  return 1
}
OURS=""; [ -f "$STATE" ] && OURS=$(sed -n 's/.*"container": "\([^"]*\)".*/\1/p' "$STATE" | tr '\n' ' ')
next_port=$PORT
for part in ${SEL//,/ }; do
  num=${part%%x*}; times=1; [[ "$part" == *x* ]] && times=${part##*x}
  [[ "$num" =~ ^[0-9]+$ ]] && [[ "$times" =~ ^[0-9]+$ ]] && [ "$num" -ge 1 ] && [ "$num" -le "${#CFG[@]}" ] && [ "$times" -ge 1 ] \
    || die "no setup '${part}' (choose 1 to ${#CFG[@]}; N,M for several; NxK for K copies)"
  for t in $(seq 1 "$times"); do
    IFS=';' read -r -a servers <<< "${CFG[$((num - 1))]}"
    for s in "${servers[@]}"; do
      IFS='|' read -r b m c <<< "$s"
      if [[ "$c" == kind:* ]]; then
        got=""; for x in ${c#kind:}; do case "$USED" in *"|$x|"*) ;; *) got=$x; break ;; esac; done
        [ -n "$got" ] || die "setup ${num}: every card of this kind already has a server (${c#kind:})"
        c=$got
      elif [ "$b" = cpu ]; then
        case "$USED" in *"|cpu|"*) die "only one CPU server" ;; esac; c=""; USED="${USED}cpu|"
      fi
      for x in ${c//,/ }; do case "$USED" in *"|$x|"*) die "setup ${num}: GPU ${x#a} already has a server" ;; esac; USED="$USED$x|"; done
      i=${#SV_B[@]}
      if [ "$i" = 0 ]; then nm=$NAME; else nm="${NAME}-$((i + 1))"; fi
      while ! port_ok "$next_port"; do next_port=$((next_port + 1)); done
      SV_B+=("$b"); SV_M+=("$m"); SV_C+=("${c#a}"); SV_NAME+=("$nm"); SV_PORT+=("$next_port"); next_port=$((next_port + 1))
    done
  done
done
NS=${#SV_B[@]}

# the engine settings of server i -> PROFILE, C (context), MEM, CPUS, ENV (the -e settings), XPACKS (extra packs)
profile_of() {
  local i=$1 B=${SV_B[$1]} M=${SV_M[$1]} CARDS=${SV_C[$1]} k V
  ENV=(); XPACKS=(); MEM=""; CPUS=""
  if [ "$B" = cpu ]; then
    PROFILE=cpu
    th=$(lscpu -p=CORE,SOCKET 2>/dev/null | grep -v '^#' | awk -F, '$2 == 0' | sort -u | wc -l); [ "$th" -gt 0 ] || th=$CORES
    [ -n "$THREADS" ] && th=$THREADS
    ENV+=(-e KINDI_DEVICE=cpu -e OMP_NUM_THREADS="$th" -e OMP_PROC_BIND=close -e OMP_PLACES=cores)
    C=${CTX/auto/24576}; MEM=8g; CPUS="$th"
  elif [ "$B" = rocm ]; then
    PROFILE="rocm-${AGFX[$CARDS]}"; C=${CTX/auto/24576}
    ENV+=(-e HIP_VISIBLE_DEVICES="$CARDS" -e ROCR_VISIBLE_DEVICES="$CARDS" -e KINDI_REPEAT_STOP=1)
    # the 780M (gfx1103) runs ROCm's gfx1102 code; system DMA off, as on the measured machine
    [ "${AGFX[$CARDS]}" = gfx1103 ] && ENV+=(-e HSA_OVERRIDE_GFX_VERSION=11.0.2 -e HSA_ENABLE_SDMA=0)
  else
    k=$(card_k "${CARDS%%,*}"); V=${GMEM[$k]}
    if [[ "$CARDS" == *,* ]]; then
      PROFILE=k1200x2; C=${CTX/auto/24576}; MEM=12g
      ENV+=(-e KINDI_CARDS=2 -e KINDI_REPEAT_STOP=1 -e KINDI_CHECKPOINT_BUDGET_MB=4096)
    else
      PROFILE=$(card_class "$k")
      case "$PROFILE" in
        tc) C=${CTX/auto/131072}; MEM=14g
            ENV+=(-e KINDI_CARDS=1 -e KINDI_REPEAT_STOP=1 -e KINDI_CHECKPOINT_BUDGET_MB=1280 -e KINDI_PARK_MAX=1
                  -e KINDI_DENSE_SPACING=2048 -e KINDI_PREFIX_MAX=2 -e KINDI_MAX_TOKENS=65536)
            if [ "$M" = 27b ]; then   # the profile of the RTX 3080 cluster servers (docs/CARDS.md)
              ENV+=(-e KINDI_TC=1 -e KINDI_FD=1 -e KINDI_ATT_TC=1 -e KINDI_CKPT_VRAM=1
                    -e KINDI_SPEC=draft2b -e KINDI_SPEC_K=3 -e KINDI_SPEC_PACK=/data/pack-2b -e KINDI_SPEC_WINDOW=8192
                    -e KINDI_TC_TMAX=4 -e KINDI_MEM_RESERVE_MB=160)
              XPACKS+=(2b)
            fi ;;
        pascal8) C=${CTX/auto/65536}; MEM=14g; ENV+=(-e KINDI_CARDS=1 -e KINDI_REPEAT_STOP=1 -e KINDI_MAX_TOKENS=65536) ;;
        kepler) if [ "$V" -ge 1500 ]; then C=${CTX/auto/65536}; else C=${CTX/auto/24576}; fi
                # the 4B on a 2 GB card: the context that the engine fits on the Quadro K420 2 GB (849 MiB free)
                [ "$M" = 4b ] && [ "$V" -lt 2500 ] && C=${CTX/auto/23808}
                # the 27B on a 12 GB Kepler card (Quadro K6000): the one-card engine, as on the GTX 1080
                [ "$M" = 27b ] && { MEM=14g; ENV+=(-e KINDI_CARDS=1 -e KINDI_REPEAT_STOP=1 -e KINDI_MAX_TOKENS=65536); } ;;
        *) C=${CTX/auto/24576} ;;
      esac
    fi
    ENV+=(-e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_VISIBLE_DEVICES="$CARDS")   # the card numbers of nvidia-smi
    # blocking sync frees a CPU core per busy server but costs ~2.5% decode (GTX 1080): only the tc profile (many
    # servers per host on the cluster) keeps it; one server on its own card waits in a loop
    [ "$PROFILE" = tc ] || ENV+=(-e KINDI_BLOCKING_SYNC=0)
  fi
  [ "$M" = nemotron ] && C=${CTX/auto/24576}
  [ -n "$MEMORY" ] && MEM="$MEMORY"
  return 0
}
model_name() { if [ "$1" = nemotron ]; then echo "nemotron-bolt-e4b"; else echo "bonsai2-$1"; fi; }

PACKS=(); BACKENDS=" "; ram_need=0; cwsr=0
for i in $(seq 0 $((NS - 1))); do
  profile_of "$i"; M=${SV_M[$i]}
  [ -n "${GGUF[$M]:-}" ] || die "unknown model '${M}' (2b, 4b, 9b, 27b or nemotron)"
  for p in "$M" ${XPACKS[@]+"${XPACKS[@]}"}; do case " ${PACKS[*]+${PACKS[*]}} " in *" $p "*) ;; *) PACKS+=("$p") ;; esac; done
  case "$BACKENDS" in *" ${SV_B[$i]} "*) ;; *) BACKENDS="$BACKENDS${SV_B[$i]} " ;; esac
  [[ "$MEM" =~ ^[0-9]+g$ ]] && ram_need=$((ram_need + ${MEM%g} * 1024))
  [ "${SV_B[$i]}" = rocm ] && [ "${AGFX[${SV_C[$i]}]}" = gfx1103 ] && cwsr=1
  ok "server $((i + 1)): $(model_label "$M") ($(model_name "$M")) on $([ "${SV_B[$i]}" = cpu ] && echo "the CPU" || echo "$([ "${SV_B[$i]}" = rocm ] && echo "AMD ")GPU(s) ${SV_C[$i]}"), context ${C}, profile ${PROFILE}; container ${SV_NAME[$i]} on port ${SV_PORT[$i]}"
done
info "(each engine fits its context to the free memory and reports what it got)"
[ "$cwsr" = 1 ] && check_cwsr
if [ "$NS" -gt 1 ] && [ "$ram_need" -gt "$RAM_MB" ]; then
  warn "the ${NS} servers may use up to $((ram_need / 1024)) GB of RAM together (this machine: $((RAM_MB / 1024)) GB); they start one after another"
fi
for i in $(seq 0 $((NS - 1))); do
  if [ "${SV_M[$i]}" = 27b ] && [[ "${SV_C[$i]}" == *,* ]] && [ "$AVAIL_MB" -lt 5000 ] && [ -z "$OURS" ] && [ "${KINDI_HOST_RAM_CHECK:-1}" = 1 ]; then
    die "the 27B on two cards needs about 5 GB of free RAM while it loads (free now: $((AVAIL_MB / 1024)) GB of $((RAM_MB / 1024)) GB).
  Stop the other model servers and large programs, then run the installer again (KINDI_HOST_RAM_CHECK=0 skips this check)."
  elif [ "${SV_M[$i]}" = 27b ] && [[ "${SV_C[$i]}" != *,* ]] && [ "$AVAIL_MB" -lt 14000 ]; then
    warn "a 27B needs about 14 GB of free RAM for its first load (free now: $((AVAIL_MB / 1024)) GB of $((RAM_MB / 1024)) GB)"
    break
  fi
done
need=0; for p in ${PACKS[@]+"${PACKS[@]}"}; do need=$((need + ${SIZE[$p]} * 3)); done
[ "$FREE_DISK" -gt "$need" ] || warn "the models, their packs and their caches need about $(human $need) of disk (free: $(human "$FREE_DISK"))"

# the docker run arguments that give a container its GPU
gpu_args() {
  GARGS=()
  if [ "$1" = rocm ]; then
    GARGS=(--device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined)
    for g in video render; do gid=$(getent group "$g" | cut -d: -f3); [ -n "$gid" ] && GARGS+=(--group-add "$gid"); done
  elif [ "$1" = cpu ]; then :
  elif [ "$GPU_MODE" = toolkit ]; then GARGS=(--gpus all)
  elif [ "$GPU_MODE" = passthrough ]; then
    for d in /dev/nvidia[0-9]* /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools; do [ -e "$d" ] && GARGS+=(--device "$d"); done
    L=$(ldconfig -p | awk '/libcuda\.so\.1 .*x86-64/ && !f {print $NF; f = 1}')
    [ -n "$L" ] || die "the NVIDIA driver library libcuda.so.1 is not on this host"
    GARGS+=(-v "$(readlink -f "$L"):/usr/lib/x86_64-linux-gnu/libcuda.so.1:ro")
    for lib in libnvidia-ptxjitcompiler.so.1 libnvidia-nvvm.so.4; do
      p=$(ldconfig -p | awk -v l="$lib" '$1 == l && /x86-64/ && !f {print $NF; f = 1}')
      [ -n "$p" ] && GARGS+=(-v "$(readlink -f "$p"):/usr/lib/x86_64-linux-gnu/${lib}:ro")
    done
  fi
  return 0
}
image_of() { if [ "$1" = rocm ]; then echo "kindi-rocm:${KINDI_VERSION}"; else echo "kindi:${KINDI_VERSION}"; fi; }
base_image_of() { if [ "$1" = rocm ]; then echo "kindi-base-rocm:${KINDI_VERSION}"; else echo "kindi-base:${KINDI_VERSION}"; fi; }
UIDGID="$(id -u):$(id -g)"

# ------------------------------------------------------------------------------------------------ 4. image
src_id=""
for b in $BACKENDS; do
  IMAGE=$(image_of "$b")
  BASE_IMAGE=$(base_image_of "$b")
  if [ "$b" = rocm ]; then
    DOCKERFILE=docker/Dockerfile.rocm
    BASE_DOCKERFILE=docker/base/Dockerfile.rocm
    step "Build the Kindi image (${IMAGE})" "The image holds ROCm 6.3, CuPy 14.2 (built for ROCm) and Kindi. The first build downloads about 6 GB and takes 30-45 minutes."
  else
    DOCKERFILE=docker/Dockerfile
    BASE_DOCKERFILE=docker/base/Dockerfile
    step "Build the Kindi image (${IMAGE})" "The image holds CUDA 10.2, CuPy 9.6 and Kindi. The first build downloads about 3 GB and takes 10-15 minutes."
  fi
  if [ -n "$SOURCE" ] && [ -d "$SOURCE" ]; then SRC="$SOURCE"
  else
    SRC="$DIR/src-${KINDI_VERSION}"
    tarball="${SOURCE:-$SOURCE_URL}"
    if [ ! -f "$SRC/$DOCKERFILE" ]; then
      run mkdir -p "$SRC"
      if [ -f "$tarball" ]; then run tar -xzf "$tarball" -C "$SRC" --strip-components=1
      else run sh -c "curl -fsSL '$tarball' | tar -xz -C '$SRC' --strip-components=1"; fi
    fi
  fi
  # the base image holds the dependencies only: build it once, then layer the code image on top of it
  if $DOCKER image inspect "$BASE_IMAGE" >/dev/null 2>&1; then ok "${BASE_IMAGE} is present"
  else
    run $DOCKER build -q -t "$BASE_IMAGE" -f "$SRC/$BASE_DOCKERFILE" "$SRC" >/dev/null
    ok "built ${BASE_IMAGE}"
  fi
  [ "$DRY" = 1 ] || src_id=$(cd "$SRC" && find kindi docker -type f | sort | xargs cat | sha256sum | cut -c1-12)
  have=$($DOCKER image inspect -f '{{index .Config.Labels "kindi.source"}}' "$IMAGE" 2>/dev/null || true)
  if [ -n "$have" ] && [ "$have" = "$src_id" ]; then ok "${IMAGE} is up to date (source ${src_id})"
  else
    [ -n "$have" ] && info "the image is from other source (${have}): upgrading"
    run $DOCKER build -q --build-arg "BASE_IMAGE=${BASE_IMAGE}" --label "kindi.source=${src_id}" -t "$IMAGE" -f "$SRC/$DOCKERFILE" "$SRC" >/dev/null
    ok "built ${IMAGE}"
  fi
  # every module must load before anything runs on a GPU (a broken file otherwise shows only at the server start)
  if [ "$DRY" = 0 ]; then
    if ! out=$($DOCKER run --rm --entrypoint python3 "$IMAGE" -c "import kindi.server, kindi.verify, kindi.convert" 2>&1); then
      echo "$out" | tail -5 | sed 's/^/    /'; die "the Kindi source in ${IMAGE} does not load (see above)"
    fi
    ok "the Kindi modules load"
  fi
done

# ------------------------------------------------------------------------------------------------ 5. download
step "Download the model$([ "${#PACKS[@]}" -gt 1 ] && echo s)" "From Hugging Face (Prism ML; unsloth for Nemotron). The sha256 of each file must match the value in this script."
[ "$DRY" = 1 ] || mkdir -p "$DIR/models"
for p in ${PACKS[@]+"${PACKS[@]}"}; do
  f="$DIR/models/${GGUF[$p]}"
  if [ -f "$f" ] && [ "$(stat -c %s "$f")" = "${SIZE[$p]}" ] && [ -f "$f.sha256-ok" ]; then ok "${GGUF[$p]} is here (checked before)"; continue; fi
  info "${GGUF[$p]}: $(human "${SIZE[$p]}")"
  if [ -t 2 ]; then prog=--progress-bar; else prog=-sS; fi     # no progress bar in a log file
  run curl -fL --retry 5 --retry-delay 5 -C - "$prog" -o "$f.part" "${URL[$p]}"
  if [ "$DRY" = 0 ]; then
    got=$(sha256sum "$f.part" | cut -d' ' -f1)
    [ "$got" = "${SHA[$p]}" ] || { rm -f "$f.part"; die "${GGUF[$p]}: sha256 ${got} is not the expected ${SHA[$p]} (deleted; run again)"; }
    mv "$f.part" "$f"; touch "$f.sha256-ok"
  fi
  ok "${GGUF[$p]}: sha256 ${SHA[$p]:0:16}... matches"
done

# ------------------------------------------------------------------------------------------------ 6. convert + verify
step "Convert and verify" "kindi convert builds Kindi's weight pack; kindi verify checks the output against llama.cpp, token for token, once for each model on each kind of card."
state_of() {      # the cache folder of server i: one per model, and one per card when a model runs more than once
  local i=$1 j n=0
  for j in $(seq 0 $((NS - 1))); do [ "${SV_M[$j]}" = "${SV_M[$i]}" ] && n=$((n + 1)); done
  if [ "$n" -gt 1 ]; then echo "state-${SV_M[$i]}-${SV_B[$i]}${SV_C[$i]//,/-}"; else echo "state-${SV_M[$i]}"; fi
}
base_of() {
  local sd; sd=$(state_of "$1")
  BASE=(-u "$UIDGID" -v "$DIR:/data" -e HOME=/tmp -e "CUPY_CACHE_DIR=/data/${sd}/cupy" -e "CUDA_CACHE_PATH=/data/${sd}/cudacache"
        -e "KINDI_STATE_DIR=/data/${sd}" -e KINDI_CPU_CACHE=/data/cpu)
}
for p in ${PACKS[@]+"${PACKS[@]}"}; do
  pk="$DIR/pack-$p"
  if [ -f "$pk/kindi_pack.json" ] && grep -q "${SHA[$p]}" "$pk/kindi_pack.json"; then ok "pack-$p is up to date"; continue; fi
  base_of 0
  run $DOCKER run --rm ${BASE[@]+"${BASE[@]}"} "$(image_of "${SV_B[0]}")" convert "/data/models/${GGUF[$p]}" --out "/data/pack-$p" --force
  ok "pack-$p converted"
done
# this installer's servers hold the GPUs: stop them for the checks (the chosen ones start again below)
for c in $OURS $(printf '%s ' "${SV_NAME[@]}"); do
  if $DOCKER ps --format '{{.Names}}' | grep -x "$c" >/dev/null; then info "stopping '${c}' for the check"; run $DOCKER stop "$c" >/dev/null; fi
done
SPEED=(); declare -A SPK
for i in $(seq 0 $((NS - 1))); do
  profile_of "$i"; M=${SV_M[$i]}; b=${SV_B[$i]}
  if [ "$b" = rocm ]; then kind="amd/${AGFX[${SV_C[$i]}]}"; elif [ "$b" = cpu ]; then kind=cpu
  else k=$(card_k "${SV_C[$i]%%,*}"); kind="${GNAME[$k]}/${GMEM[$k]}/${SV_C[$i]//[0-9]/}"; fi
  key="$M/$PROFILE/$kind"
  if [ -n "${SPK[$key]:-}" ]; then SPEED+=("${SPK[$key]}"); ok "server $((i + 1)): checked above (the same model on the same kind of card)"; continue; fi
  [ "$M" = 27b ] && [ "$PROFILE" != k1200x2 ] && [ "$PROFILE" != kepler ] && info "the first 27B start writes its 5-trit weights into the pack (5.6 GB, up to 15 minutes)"
  base_of "$i"; gpu_args "$b"; IMAGE=$(image_of "$b")
  VERIFY_OUT="$DIR/verify-$((i + 1)).log"; sp="?"
  if [ "$DRY" = 1 ]; then info "\$ $DOCKER run --rm ${BASE[*]} ${GARGS[*]:-} ${ENV[*]:-} -e KINDI_PACK=/data/pack-$M $IMAGE verify"
  else
    VX=(); [ -n "$CPUS" ] && VX=(--cpus "$CPUS")
    if $DOCKER run --rm ${VX[@]+"${VX[@]}"} ${BASE[@]+"${BASE[@]}"} ${GARGS[@]+"${GARGS[@]}"} ${ENV[@]+"${ENV[@]}"} -e "KINDI_PACK=/data/pack-$M" -e KINDI_HEAD=exact "$IMAGE" verify > "$VERIFY_OUT" 2>&1; then
      grep -E "PASS|FAIL" "$VERIFY_OUT" | sed 's/^\[[^]]*\] /    /' | sed -n 1,6p
      sp=$(grep -o '[0-9.]* tokens/s over the run' "$VERIFY_OUT" | awk '{print $1}')
      if [ "$M" = nemotron ]; then ok "server $((i + 1)): agrees with llama.cpp on the first 12 tokens or more of each prompt (${sp} tokens/s over the check)"
      else ok "server $((i + 1)): token-exact with llama.cpp (${sp} tokens/s over the check, prompt tokens included)"; fi
    else
      tail -15 "$VERIFY_OUT" | sed 's/^/    /'
      die "kindi verify did not pass for server $((i + 1)) (log: ${VERIFY_OUT}). No server is started."
    fi
  fi
  SPEED+=("$sp"); SPK[$key]=$sp
done

# ------------------------------------------------------------------------------------------------ 7. service
step "Start the server$([ "$NS" -gt 1 ] && echo s)" "Docker containers with restart=unless-stopped: they start again after a reboot. Only this machine can reach them (127.0.0.1)."
for c in $OURS; do      # an earlier install's servers that this setup does not keep
  case " ${SV_NAME[*]} " in *" $c "*) ;; *)
    if $DOCKER ps -a --format '{{.Names}}' | grep -x "$c" >/dev/null; then info "removing the earlier server '${c}'"; run $DOCKER rm -f "$c" >/dev/null; fi ;; esac
done
GOT_CTX=(); GOT_VER=$KINDI_VERSION; GOT_SHA=""; JSON=""
for i in $(seq 0 $((NS - 1))); do
  profile_of "$i"; M=${SV_M[$i]}; b=${SV_B[$i]}; nm=${SV_NAME[$i]}; pt=${SV_PORT[$i]}
  base_of "$i"; gpu_args "$b"; IMAGE=$(image_of "$b")
  $DOCKER ps -a --format '{{.Names}}' | grep -x "$nm" >/dev/null && run $DOCKER rm -f "$nm" >/dev/null
  SARGS=(-d --name "$nm" --restart unless-stopped ${BASE[@]+"${BASE[@]}"} ${GARGS[@]+"${GARGS[@]}"} ${ENV[@]+"${ENV[@]}"} -p "127.0.0.1:${pt}:8093"
         -e "KINDI_PACK=/data/pack-$M" -e "KINDI_CTX=$C" -e KINDI_HEAD=exact -e "KINDI_MODEL_NAME=$(model_name "$M")" -e KINDI_HOST=0.0.0.0)
  [ -n "$MEM" ] && SARGS+=(--memory "$MEM")
  [ -n "$CPUS" ] && SARGS+=(--cpus "$CPUS")
  run $DOCKER run ${SARGS[@]+"${SARGS[@]}"} "$IMAGE" serve >/dev/null
  HEALTH=""
  if [ "$DRY" = 0 ]; then
    printf "    waiting for %s" "$nm"
    for t in $(seq 1 450); do
      HEALTH=$(curl -fs -m 2 "http://127.0.0.1:${pt}/health" 2>/dev/null || true)
      echo "$HEALTH" | grep -q '"ok":true' && break
      # stopped, or restarting after a crash (restart=unless-stopped keeps a crashing container in 'docker ps')
      st=$($DOCKER inspect -f '{{.State.Running}} {{.State.Restarting}} {{.RestartCount}}' "$nm" 2>/dev/null || echo "false false 0")
      if [ "${st%% *}" != true ] || [ "${st##* }" != 0 ]; then
        echo; $DOCKER logs --tail 20 "$nm" 2>&1 | sed 's/^/    /'; die "the server ${nm} stopped (see above)"
      fi
      [ $((t % 10)) = 0 ] && printf "."
      sleep 2
    done
    echo
    echo "$HEALTH" | grep -q '"ok":true' || die "the server ${nm} did not answer in 15 minutes ($DOCKER logs ${nm})"
  fi
  gc=$(echo "$HEALTH" | sed -n 's/.*"ctx":\([0-9]*\).*/\1/p'); GOT_CTX+=("${gc:-$C}")
  v=$(echo "$HEALTH" | sed -n 's/.*"kindi": *"\([^"]*\)".*/\1/p'); [ -n "$v" ] && GOT_VER=$v
  v=$(echo "$HEALTH" | sed -n 's/.*"commit": *"\([0-9a-f]*\)".*/\1/p'); [ -n "$v" ] && GOT_SHA=$v
  ok "${nm} answers on http://127.0.0.1:${pt}"
  JSON="${JSON:+$JSON,
}  {\"container\": \"${nm}\", \"image\": \"${IMAGE}\", \"port\": ${pt}, \"backend\": \"${b}\", \"profile\": \"${PROFILE}\", \"model\": \"${M}\", \"ctx\": ${GOT_CTX[$i]}, \"gpus\": \"${SV_C[$i]}\"}"
done

[ "$DRY" = 1 ] || cat > "$STATE" <<EOF
{"version": "${KINDI_VERSION}", "source": "${src_id}", "gpu_access": "${GPU_MODE}", "installed": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
 "servers": [
${JSON}
 ]}
EOF

# ------------------------------------------------------------------------------------------------ done
echo
echo "${G}${B}Kindi is running$([ "$NS" -gt 1 ] && echo ": ${NS} servers").${N}"
echo
echo "  Kindi      ${GOT_VER}${GOT_SHA:+ (commit ${GOT_SHA})}"
for i in $(seq 0 $((NS - 1))); do
  echo "  ${SV_NAME[$i]}  http://127.0.0.1:${SV_PORT[$i]}/v1  $(model_name "${SV_M[$i]}") on $([ "${SV_B[$i]}" = cpu ] && echo "the CPU" || echo "$([ "${SV_B[$i]}" = rocm ] && echo "AMD ")GPU ${SV_C[$i]}"), ${GOT_CTX[$i]:-?} tokens, ${SPEED[$i]:-?} tokens/s in the check"
done
URL="http://127.0.0.1:${SV_PORT[0]}/v1"; MODEL_NAME=$(model_name "${SV_M[0]}")
cat <<EOF

  Each server is an OpenAI-compatible endpoint; status: /health on its port. Logs: ${DOCKER} logs -f ${SV_NAME[0]}
  The containers restart unless stopped.

Try it:
  curl ${URL}/chat/completions -H 'Content-Type: application/json' \\
    -d '{"messages": [{"role": "user", "content": "Hello!"}]}'

OpenAI clients:
  export OPENAI_BASE_URL=${URL}  OPENAI_API_KEY=none

pi (~/.pi/agent/models.json; one provider for each server):
  {"providers": {
$(for i in $(seq 0 $((NS - 1))); do mn=$(model_name "${SV_M[$i]}"); pv=${SV_NAME[$i]}
  printf '    "%s": {"baseUrl": "http://127.0.0.1:%s/v1", "api": "openai-completions", "apiKey": "none",\n' "$pv" "${SV_PORT[$i]}"
  printf '      "compat": {"supportsDeveloperRole": false, "supportsReasoningEffort": false,\n'
  printf '                 "thinkingFormat": "qwen-chat-template", "thinkingTokenBudgetField": "thinking_budget_tokens"},\n'
  printf '      "models": [{"id": "%s", "name": "%s", "reasoning": true, "input": ["text"],\n' "$mn" "$mn"
  printf '                  "contextWindow": %s, "maxTokens": 4096}]}%s\n' "${GOT_CTX[$i]:-24576}" "$([ "$i" -lt $((NS - 1)) ] && echo ,)"
done)
  }}
  (the thinking format sends chat_template_kwargs.enable_thinking, which both Kindi model families read)

Run this installer again to upgrade, repair or change the setup. Remove Kindi: bash install.sh --uninstall [--purge]
EOF
