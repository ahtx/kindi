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

KINDI_VERSION="0.6.0"
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
# Nemotron 3.5 Lightning 30B-A3B, Nota AI's ternary version (GGUF by boro64; licence OpenMDW 1.1): the Lightning
# MoE engine. A 10 GB card holds all of it; on 6 to 8 GB cards the least used experts stay in host memory.
GGUF[lightning]="Ternary-Nemotron-3.5-Lightning-30B-A3B-GGUF.gguf"; SHA[lightning]="38854de649212dc2761006b75afbc41412e7531f763c3ff965bd31f969bb14f3"; SIZE[lightning]=9472656320
URL[lightning]="https://huggingface.co/boro64/Ternary-Nemotron-3.5-Lightning-30B-A3B-GGUF/resolve/main/${GGUF[lightning]}"
# AH-Nanbeige4.3-33B IQ3_XXS (a separate model; the Kindi Nanbeige backend, NVIDIA sm_80 and later)
GGUF[nanbeige]="AH-Nanbeige4.3-33B-IQ3_XXS.gguf"; SHA[nanbeige]="9d4f18e8a22573c94b18a455147e807a8b9d2cd4b30302f2a4639c2db0a6c239"; SIZE[nanbeige]=13108685952
NB_CARD_URL="https://huggingface.co/ahtx/AH-Nanbeige4.3-33B-IQ3_XXS-GGUF"
URL[nanbeige]="${NB_CARD_URL}/resolve/main/${GGUF[nanbeige]}"
# MTP head for the 2B (docs/MTP.md, tensor-core cards only).
# The files are on Hugging Face (https://huggingface.co/ahtx/kindi-mtp-heads); KINDI_MTP_URL overrides the location.
MTP_URL="${KINDI_MTP_URL:-https://huggingface.co/ahtx/kindi-mtp-heads/resolve/main}"
MTPF="kindi-mtp-2b-v1.npz"; MTPSHA="1b5c0c790a594af368c3646370c527c319cdb3bb0592dd3e523e752f721a5520"; MTPSIZE=60960552
mtp_ready() { [ -f "$DIR/mtp/$MTPF" ] && [ -f "$DIR/mtp/$MTPF.sha256-ok" ]; }
# The DSpine block drafter of the 27B on tensor-core cards (Tencent, Apache-2.0: https://huggingface.co/tencent/Qwen3.8-27B-DSpine-b16),
# pinned to a revision and sha256; KINDI_DSPINE_URL overrides the location, --no-dspine leaves it out (docs/DSPINE.md)
DSP_URL="${KINDI_DSPINE_URL:-https://huggingface.co/tencent/Qwen3.8-27B-DSpine-b16/resolve/5f54174d301dbca65d9aa4448dc47e56732d9a12}"
DSPF=(config.json LICENSE model.safetensors)
DSPSHA=(c2aa9e3ff22766ea7df0fa5b5f5f7d2035ae722671c6b7eb58b65578705e6815 bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a
        ac90f1df842ea66b23e4224f3e86624add547526f689ffe9e9c6b7a5937116b1)
DSPSIZE=4051845726                                   # the three files
dspine_ready() { [ "${NO_DSPINE:-0}" = 0 ] && [ -f "$DIR/dspine/model.safetensors.sha256-ok" ] && [ -f "$DIR/dspine/config.json" ]; }

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
LAN=0             # --lan: serve the local network (0.0.0.0), with an API key
APIKEY=""         # --api-key K: the key clients must send; "none" = no key (not recommended with --lan)
BRIDGE=ask        # Kindi Bridge (optional): ask | yes (--with-bridge) | no (--no-bridge); --yes/--auto alone never adds it
BRIDGE_SOURCE=""  # --bridge-source: a Kindi Bridge folder (default: bridge/ in the Kindi source)
PI_VERSION="1.3.0"   # the pi coding agent that Bridge is tested with

usage() {
  cat <<EOF
Kindi ${KINDI_VERSION} installer

  install.sh [options]          install, upgrade or repair Kindi
  install.sh --uninstall        remove the Kindi service and image (keeps the models in ${DIR})
  install.sh --uninstall --purge   also remove ${DIR}

Options:
  --dry-run        print every step and command, change nothing
  --no-dspine      the 27B on tensor-core cards drafts with the 2B, not Tencent's DSpine (4 GB, Apache-2.0)
  --yes, -y        answer yes to every question (for scripts); takes the recommended setup
  --auto           choose the recommended setup and continue without questions (kernel changes are only
                   described, never made)
  --config N       take setup N of the list that the installer prints (see --dry-run); N,M runs several setups
                   side by side, one server each (for example a 27B on each of two cards); NxK runs setup N on
                   K cards of that kind. Each server gets its own container (kindi, kindi-2, ...) and port.
  --model M        2b | 4b | 9b | 27b | lightning | nemotron | nanbeige (default: the recommended model for the
                   card; lightning = Nemotron 3.5 Lightning 30B-A3B, a mixture of experts, on NVIDIA cards with 6 GB
                   or more)
  --ctx N          context length in tokens (default: the profile's; the engine fits it to the free memory)
  --gpu G          N (one NVIDIA card), N,M (two cards), amd or amdN (an AMD card), all, or cpu (default: choose)
  --port P         server port on 127.0.0.1 (default ${PORT})
  --lan            also serve the local network (other computers can use Kindi). Off by default. The installer
                   makes an API key (kept in ${DIR}/api-key) that every client must send, unless --api-key none
  --api-key K      the API key clients must send (Authorization: Bearer K); also without --lan
  --dir D          data folder (default ${DIR})
  --name N         container name (default ${NAME})
  --threads N      CPU engine: the number of threads (default: the cores of one socket)
  --memory M       the container memory limit, for example 8g (default: the profile's)
  --source S       Kindi source: a folder or a .tar.gz URL/path (default: the ${KINDI_VERSION} release)
  --with-bridge    also install Kindi Bridge, the terminal workspace for agents on Kindi (optional; the installer
                   asks when it has a terminal)
  --no-bridge      do not ask about Kindi Bridge
  --bridge-source D   install Kindi Bridge from folder D (default: bridge/ in the Kindi source)
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --no-dspine) NO_DSPINE=1 ;;
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
    --lan) LAN=1 ;;
    --api-key) APIKEY="$2"; shift ;;
    --with-bridge) BRIDGE=yes ;;
    --no-bridge) BRIDGE=no ;;
    --bridge-source) BRIDGE_SOURCE="$2"; BRIDGE=yes; shift ;;
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

# ------------------------------------------------------------------------------------------------ Kindi Bridge
# Kindi Bridge is optional. It runs on the host (not in a container): a Python venv in $DIR/bridge, a launcher in
# ~/.local/bin, its state (agents, schedules, knowledge) in ~/.local/state/kindi-bridge, and its own pi folder there,
# so the user's own pi setup does not change. $DIR/bridge.json records what was installed, for --uninstall.
BRIDGE_HOME="$DIR/bridge"
BRIDGE_BIN="$HOME/.local/bin/kindi-bridge"
BRIDGE_STATE="${XDG_STATE_HOME:-$HOME/.local/state}/kindi-bridge"
BRIDGE_UNIT="$HOME/.config/systemd/user/kindi-bridge.service"
BRIDGE_RECORD="$DIR/bridge.json"

ver_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" = "$2" ]; }   # ver_ge 3.11.2 3.10 -> true

bridge_install() {   # $1 = the port of the first Kindi server
  local src="${BRIDGE_SOURCE:-${SRC:-}/bridge}" py pyv tmv knowledge=0 unit=0
  step "Install Kindi Bridge" "A terminal workspace for agents on Kindi: conversations, scheduled prompts, completion checks, local knowledge folders and live Kindi telemetry. It runs on this computer, not in a container."
  [ -f "$src/pyproject.toml" ] || { warn "no Kindi Bridge source in ${src} (use --bridge-source DIR): Bridge is not installed"; return 0; }
  py=$(command -v python3 || true)
  pyv=$([ -n "$py" ] && "$py" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo 0)
  tmv=$(tmux -V 2>/dev/null | sed -n 's/^tmux \([0-9.]*\).*/\1/p'); tmv=${tmv:-0}
  local missing=""
  ver_ge "$pyv" 3.10 || missing="$missing python3 (3.10 or later)"
  [ -n "$py" ] && ! "$py" -c 'import venv, ensurepip' 2>/dev/null && missing="$missing python3-venv"
  ver_ge "$tmv" 3.2 || missing="$missing tmux (3.2 or later)"
  if [ -n "$missing" ]; then
    info "Bridge needs:${missing}."
    if command -v apt-get >/dev/null && ask "Install python3, python3-venv and tmux with apt-get (needs sudo)?"; then
      need_sudo; run $SUDO apt-get update -qq; run $SUDO apt-get install -y -qq python3 python3-venv tmux >/dev/null
      py=$(command -v python3)
    else warn "Bridge is not installed. Install${missing}, then run this installer again with --with-bridge."; return 0; fi
  fi
  ok "python $("$py" -c 'import platform; print(platform.python_version())' 2>/dev/null || echo "?"), $(tmux -V 2>/dev/null || echo tmux)"
  local pibin; pibin=$(command -v pi || ls "$HOME"/.nvm/versions/node/*/bin/pi 2>/dev/null | tail -1 || true)
  [ -z "$pibin" ] && [ -x "$HOME/.local/bin/pi" ] && pibin="$HOME/.local/bin/pi"
  if [ -n "$pibin" ]; then
    if "$pibin" --version >/dev/null 2>&1; then ok "pi coding agent found ($("$pibin" --version 2>/dev/null | head -1))"
    else warn "${pibin} does not start: pi ${PI_VERSION} needs Node.js 22.19 or later (this Node.js: $(node -v 2>/dev/null || echo none)). Bridge installs, but its agents start only after a Node.js upgrade."; fi
  elif command -v npm >/dev/null && ver_ge "$(node -v 2>/dev/null | tr -d v)" 22.19; then
    info "Bridge's agents use the pi coding agent (an npm package, ${PI_VERSION}). It installs into ~/.local, without sudo."
    if ask "Install pi ${PI_VERSION} with npm?"; then
      run npm install -g -s --prefix "$HOME/.local" "@earendil-works/pi-coding-agent@${PI_VERSION}" >/dev/null; ok "pi installed in ~/.local/bin"
    else warn "without pi, Bridge opens but cannot start agents. Later: npm install -g --prefix ~/.local @earendil-works/pi-coding-agent@${PI_VERSION}"; fi
  else warn "Bridge's agents need the pi coding agent, and pi ${PI_VERSION} needs Node.js 22.19 or later (https://nodejs.org; this Node.js: $(node -v 2>/dev/null || echo none)). Install it, then run this installer again with --with-bridge."
  fi
  # pi searches files with ripgrep and fd (optional; without them pi prints a warning and works more slowly)
  local tools=""
  command -v rg >/dev/null || tools="$tools ripgrep"
  command -v fd >/dev/null || command -v fdfind >/dev/null || tools="$tools fd-find"
  if [ -n "$tools" ] && command -v apt-get >/dev/null; then
    info "pi searches files with ripgrep and fd. Missing:${tools}."
    if ask "Install${tools} with apt-get (needs sudo)?"; then
      need_sudo; run $SUDO apt-get install -y -qq $tools >/dev/null || warn "could not install${tools}"
    fi
  elif [ -n "$tools" ]; then warn "pi searches files with ripgrep and fd. Install${tools} for faster file search."; fi
  if ! command -v fd >/dev/null && command -v fdfind >/dev/null; then   # Debian and Ubuntu name the fd program fdfind
    run mkdir -p "$HOME/.local/bin"; run ln -sf "$(command -v fdfind)" "$HOME/.local/bin/fd"; ok "fd -> fdfind in ~/.local/bin"
  fi
  info "Document search (PDF, Word, PowerPoint and Excel files, with local embeddings) adds about 300 MB."
  ask "Add document search to Bridge?" && knowledge=1
  run mkdir -p "$BRIDGE_HOME" "$(dirname "$BRIDGE_BIN")" "$BRIDGE_STATE/pi-agent"
  [ -x "$BRIDGE_HOME/venv/bin/python" ] || run "$py" -m venv "$BRIDGE_HOME/venv"
  # build from a copy: pip writes build files next to the source, and the source can be read-only
  run rm -rf "$BRIDGE_HOME/src"; run mkdir -p "$BRIDGE_HOME/src"
  run sh -c "cd '$src' && tar -c --exclude=./.git --exclude=./.venv --exclude='*.egg-info' --exclude=__pycache__ . | tar -x -C '$BRIDGE_HOME/src'"
  run "$BRIDGE_HOME/venv/bin/pip" install -q --upgrade "$BRIDGE_HOME/src$([ "$knowledge" = 1 ] && echo '[knowledge]')" >/dev/null
  [ "$DRY" = 1 ] || cat > "$BRIDGE_BIN" <<EOF
#!/bin/sh
# Kindi Bridge launcher (written by the Kindi installer). Bridge keeps its own pi folder, so ~/.pi does not change.
export PI_CODING_AGENT_DIR="\${PI_CODING_AGENT_DIR:-${BRIDGE_STATE}/pi-agent}"
export PATH="\$HOME/.local/bin:\$PATH"
exec "${BRIDGE_HOME}/venv/bin/kindi-bridge" "\$@"
EOF
  [ "$DRY" = 1 ] || chmod 755 "$BRIDGE_BIN"
  ok "launcher ${BRIDGE_BIN}"
  # connect Bridge's agents to the first Kindi server (Bridge's own code writes the pi provider, the same form as below)
  if [ "$DRY" = 0 ]; then
    if out=$(PI_CODING_AGENT_DIR="$BRIDGE_STATE/pi-agent" BRIDGE_STATE="$BRIDGE_STATE" "$BRIDGE_HOME/venv/bin/python" -c \
        "import sys; from kindi_bridge import kindi; from kindi_bridge.store import Store; r = kindi.use(Store(), sys.argv[1], 'kindi'); print(r['model'], r['ctx'])" \
        "http://127.0.0.1:$1" 2>&1); then ok "agents use ${out% *} on port $1 (${out##* } tokens of context)"
    else warn "could not connect Bridge to port $1: ${out##*$'\n'} (later: Kindi > Connect to an address in Bridge)"; fi
  fi
  if command -v systemctl >/dev/null && systemctl --user show-environment >/dev/null 2>&1; then
    info "Bridge's scheduler sends scheduled prompts. It starts with Bridge; as a user service it also runs when Bridge is closed."
    if ask "Run the Bridge scheduler as a user service (starts at login)?"; then
      unit=1
      [ "$DRY" = 1 ] || { mkdir -p "$(dirname "$BRIDGE_UNIT")"; cat > "$BRIDGE_UNIT" <<EOF
[Unit]
Description=Kindi Bridge scheduler and knowledge indexer
After=default.target

[Service]
Type=simple
ExecStart=${BRIDGE_BIN} serve
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF
      }
      run systemctl --user daemon-reload; run systemctl --user enable --now kindi-bridge.service >/dev/null 2>&1 || warn "could not start the user service"
    fi
  fi
  [ "$DRY" = 1 ] || printf '{"home": "%s", "bin": "%s", "state": "%s", "unit": %s, "knowledge": %s}\n' \
      "$BRIDGE_HOME" "$BRIDGE_BIN" "$BRIDGE_STATE" "$([ "$unit" = 1 ] && echo "\"$BRIDGE_UNIT\"" || echo null)" "$knowledge" > "$BRIDGE_RECORD"
  BRIDGE_DONE=1
}

bridge_uninstall() {
  [ -f "$BRIDGE_RECORD" ] || [ -x "$BRIDGE_BIN" ] || return 0
  step "Remove Kindi Bridge" "Removes its venv, launcher and user service. Its state (agents, schedules, knowledge) in ${BRIDGE_STATE} stays unless --purge."
  # close Bridge first: its own tmux server (the workbench, its agents and their panes), then its scheduler, found by
  # the PID that Bridge records (never by name). The conversations are saved in the state folder.
  if command -v tmux >/dev/null && tmux -L kindi-bridge has-session 2>/dev/null; then
    info "Bridge is open: its window and its agents close now (the conversations are saved)."
    run tmux -L kindi-bridge kill-server >/dev/null 2>&1 || true; ok "Bridge's tmux server stopped"
  fi
  local spid=""
  [ -f "$BRIDGE_STATE/bridge.sqlite3" ] && command -v python3 >/dev/null && spid=$(python3 -c \
      "import sqlite3,sys; r=sqlite3.connect(sys.argv[1]).execute(\"SELECT value FROM settings WHERE key='service_pid'\").fetchone(); print(r[0] if r else '')" \
      "$BRIDGE_STATE/bridge.sqlite3" 2>/dev/null || true)
  if [ -n "$spid" ] && tr '\0' ' ' < "/proc/$spid/cmdline" 2>/dev/null | grep -q 'kindi_bridge serve\|kindi-bridge serve'; then
    run kill "$spid" 2>/dev/null || true; ok "Bridge scheduler (PID ${spid}) stopped"
  fi
  if [ -f "$BRIDGE_UNIT" ]; then
    run systemctl --user disable --now kindi-bridge.service >/dev/null 2>&1 || true
    run rm -f "$BRIDGE_UNIT"; run systemctl --user daemon-reload >/dev/null 2>&1 || true; ok "user service removed"
  fi
  if [ -f "$BRIDGE_BIN" ] && grep -q 'written by the Kindi installer' "$BRIDGE_BIN" 2>/dev/null; then run rm -f "$BRIDGE_BIN"; ok "launcher removed"; fi
  [ -d "$BRIDGE_HOME" ] && { run rm -rf "$BRIDGE_HOME"; ok "Bridge venv removed"; }
  run rm -f "$BRIDGE_RECORD"
  if [ "$PURGE" = 1 ] && [ -d "$BRIDGE_STATE" ] && ask "Delete Bridge's state (${BRIDGE_STATE}: agents, conversations, schedules, knowledge indexes)?"; then
    run rm -rf "$BRIDGE_STATE"; ok "deleted"
  fi
}

# ------------------------------------------------------------------------------------------------ uninstall
if [ "$MODE" = uninstall ]; then
  bridge_uninstall
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
step "Check the machine" "Kindi runs on 64-bit Linux: on an NVIDIA GPU from the Quadro K420 (Kepler) up, on an AMD RDNA3 or RDNA 3.5 GPU, or on the CPU."
[ "$(uname -s)" = Linux ] || die "Kindi needs Linux (found $(uname -s))"
[ "$(uname -m)" = x86_64 ] || die "Kindi needs an x86-64 CPU (found $(uname -m))"
OS_ID=$(. /etc/os-release 2>/dev/null && echo "${ID:-unknown}") || OS_ID=unknown
# WSL2 (Kindi for Windows, install.ps1): the Windows driver serves the GPU; its tools are in /usr/lib/wsl/lib
IS_WSL=0; grep -qsi microsoft /proc/sys/kernel/osrelease && IS_WSL=1
if [ "$IS_WSL" = 1 ] && [ -d /usr/lib/wsl/lib ]; then case ":$PATH:" in *:/usr/lib/wsl/lib:*) ;; *) PATH="$PATH:/usr/lib/wsl/lib" ;; esac; fi
RAM_MB=$(awk '/MemTotal/ {print int($2 / 1024)}' /proc/meminfo)
AVAIL_MB=$(awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo)
CORES=$(nproc)
[ "$DRY" = 1 ] || mkdir -p "$DIR" 2>/dev/null || die "cannot create ${DIR}"
d="$DIR"; while [ ! -d "$d" ]; do d=$(dirname "$d"); done      # the nearest folder that exists (a dry run makes none)
FREE_DISK=$(df -Pk "$d" | awk 'NR == 2 {print $4 * 1024}')
ok "Linux ($OS_ID), ${CORES} CPU threads, $((RAM_MB / 1024)) GB RAM, $(human "$FREE_DISK") free for ${DIR}"
if [ "$IS_WSL" = 1 ]; then
  info "WSL2: this RAM is the WSL virtual machine's. Windows gives it about half of the PC's RAM by default;"
  info "memory= in %UserProfile%\\.wslconfig changes it (install.ps1 -WslMemoryGB N does this)."
fi

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
        *"GTX TITAN X"*) c=5.2 ;;                                                      # GM200 (Maxwell): before "GTX TITAN"
        # data-centre parts first: "SXM4" in a name would match the Maxwell M[0-9] pattern below
        *"Tesla P100"*|*"Quadro GP100"*) c=6.0 ;;                                       # GP100: no dp4a
        *V100*|*"TITAN V"*|*"Quadro GV100"*) c=7.0 ;;                                   # GV100
        *A100*|*A30*|*"CMP 170HX"*) c=8.0 ;;                                            # GA100: before A[0-9]
        *H100*|*H200*|*GH200*) c=9.0 ;;
        # the K20 and K40 patterns must not match the K2000 and the K4000: a digit after "K20" or "K40" is another card
        *K6000*|*K5200*|*K40[!0-9]*|*K40|*K20[!0-9]*|*K20|*"GTX TITAN"*|*"GTX 780"*) c=3.5 ;;   # GK110
        *K80*) c=3.7 ;;                                                                # GK210
        *K420*|*K600*|*K2000*|*K4000*|*K4200*|*K5000*|*K520*|*K10[!0-9]*|*K10|*"GTX 6"*|*"GTX 7"[0-4]*|*"GT 7"*) c=3.0 ;;   # GK104, GK106, GK107
        # Maxwell 2 (GM200, GM204, GM206) is sm_52: before the M[0-9] line below (issue #30: the M5000 showed sm_50)
        *"Quadro M2000"|*"Quadro M4000"|*"Quadro M5000"|*"Quadro M6000"*|*"Quadro M3000M"*|*"Quadro M4000M"*|*"Quadro M5000M"*|*"Tesla M4"|*"Tesla M40"*|*"Tesla M6"|*"Tesla M60"*|*"GTX 950"|*"GTX 960"|*"GTX 970"*|*"GTX 980"*|*"GTX 965M"*) c=5.2 ;;
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
  # (no lspci in minimal systems and containers: the name then comes from the gfx id)
  name=$( (lspci -s "$bus" 2>/dev/null || true) | sed 's/^[^ ]* [^:]*: //; s/ (rev [0-9a-f]*)$//; s/Advanced Micro Devices, Inc. //; s/\[AMD\/ATI\] //')
  [ "$gfx" = gfx1103 ] && name="Radeon 780M/760M, ${name:-Phoenix}"
  [ "$gfx" = gfx1152 ] && name="Radeon 840M/860M, ${name:-Krackan}"
  [ "$gfx" = gfx1151 ] && name="Radeon 8060S/8050S, ${name:-Strix Halo}"
  [ "$gfx" = gfx1150 ] && name="Radeon 890M/880M, ${name:-Strix Point}"
  case "$gfx" in
    gfx110[0-3]|gfx115[0-2]) ;;
    *) warn "AMD GPU ${bus} (${name:-$gfx}, ${gfx}): Kindi supports AMD RDNA3 (gfx1100-gfx1103) and RDNA 3.5 (gfx1150-gfx1152) GPUs only"; continue ;;
  esac
  # an APU (the 780M) has a small carve-out and uses system memory (GTT): count the larger one
  m=$vram; [ "$gtt" -gt "$m" ] && [ "$vram" -lt 8192 ] && m=$gtt
  ANAME+=("${name:-AMD $gfx}"); AGFX+=("$gfx"); AMEM+=("$m"); ABUS+=("$bus")
done
NA=${#AGFX[@]}
for k in $(seq 0 $((NA - 1))); do
  case "${AGFX[$k]}" in gfx1103|gfx1151|gfx1152) nm="" ;; *) nm=" (not yet measured with Kindi; the Radeon 780M, 840M and 8060S are)" ;; esac
  ok "AMD GPU ${k}: ${ANAME[$k]}, ${AGFX[$k]}, ${AMEM[$k]} MiB usable${nm}"
done
if [ "$IS_WSL" = 1 ] && [ "$NA" = 0 ] && [ "$NG" = 0 ]; then
  info "WSL2: no GPU that Kindi can use is visible here, so Kindi uses the CPU engine (an AMD GPU cannot be used in WSL2: its engine needs /dev/kfd)"
elif [ "$NA" = 0 ] && lspci -n 2>/dev/null | grep -E ' 03(00|80): 1002:' >/dev/null; then
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
  [ "$IS_WSL" = 1 ] && return 0      # WSL2: no Linux kernel modules for the GPU; the Windows driver controls it
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
step "Check Docker and GPU access" "Kindi runs in a Docker container with CUDA 10.2 (NVIDIA) or ROCm 6.3 / 7 (AMD) inside, so the host needs only the driver."
if ! command -v docker >/dev/null; then
  warn "Docker is not installed"
  case "$OS_ID" in
    ubuntu|debian|linuxmint|pop)
      if ask "Install Docker (apt package docker.io) and add ${USER} to the docker group?"; then
        need_sudo; run $SUDO apt-get update -qq; run $SUDO apt-get install -y -qq docker.io
        run $SUDO usermod -aG docker "$USER"; [ "$DRY" = 1 ] || DOCKER="$SUDO docker"   # a dry run must not wait for a sudo password
      else die "Kindi needs Docker: https://docs.docker.com/engine/install/"; fi ;;
    *) die "install Docker for ${OS_ID} (https://docs.docker.com/engine/install/), then run this again" ;;
  esac
fi
if [ "$DRY" = 0 ] && ! docker info >/dev/null 2>&1; then
  if sudo -n docker info >/dev/null 2>&1 || { need_sudo; $SUDO docker info >/dev/null 2>&1; }; then
    DOCKER="sudo docker"; warn "your user cannot use Docker yet (log out and in after 'usermod -aG docker'); using sudo docker"
  elif [ "$IS_WSL" = 1 ]; then
    die "Docker does not answer. Under WSL2, Docker needs systemd ([boot] systemd=true in /etc/wsl.conf, then 'wsl --terminate <distribution>' in Windows) or Docker Desktop's WSL integration. install.ps1 sets this up."
  else die "Docker does not answer (is the daemon running? 'sudo systemctl start docker')"; fi
fi
ok "Docker $(docker --version 2>/dev/null | awk '{print $3}' | tr -d ,)"
# rootless Docker maps the container's root to this user: '-u' would make /data read-only, and the daemon is a user service
ROOTLESS=0
if [ "$DRY" = 0 ] && $DOCKER info -f '{{println .SecurityOptions}}' 2>/dev/null | grep -q rootless; then
  ROOTLESS=1; ok "rootless Docker: the containers run as your user without '-u'"
fi
restart_docker() {
  if [ "$ROOTLESS" = 1 ]; then run systemctl --user restart docker; else run $SUDO systemctl restart docker; fi
}

GPU_MODE=none
if [ "$NG" -gt 0 ] && [ "$GPU" != cpu ]; then
  if [ "$IS_WSL" = 1 ] && [ "$DRY" = 0 ] && $DOCKER info -f '{{.OperatingSystem}}' 2>/dev/null | grep -q 'Docker Desktop'; then
    GPU_MODE=toolkit; ok "Docker Desktop (WSL2): the container gets the GPUs with --gpus"
  elif $DOCKER info 2>/dev/null | grep -i 'runtimes.*nvidia' >/dev/null || command -v nvidia-ctk >/dev/null || command -v nvidia-container-toolkit >/dev/null; then
    GPU_MODE=toolkit; ok "NVIDIA container toolkit found: the container gets the GPUs with --gpus"
  else
    warn "the NVIDIA container toolkit is not installed"
    if [ "$OS_ID" = ubuntu ] || [ "$OS_ID" = debian ]; then
      if ask "Install the NVIDIA container toolkit (NVIDIA's apt repository)?"; then
        need_sudo
        run sh -c "curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | $SUDO gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
        run sh -c "curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | $SUDO tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null"
        run $SUDO apt-get update -qq; run $SUDO apt-get install -y -qq nvidia-container-toolkit
        if [ "$ROOTLESS" = 1 ]; then run nvidia-ctk runtime configure --runtime=docker --config="$HOME/.config/docker/daemon.json"
        else run $SUDO nvidia-ctk runtime configure --runtime=docker; fi
        restart_docker
        GPU_MODE=toolkit
      fi
    fi
    if [ "$GPU_MODE" = none ] && [ "$IS_WSL" = 1 ]; then
      die "under WSL2 a container reaches the GPU only through the NVIDIA container toolkit or Docker Desktop. Install one of them (install.ps1 does), or use --gpu cpu."
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
  [ "$IS_WSL" = 1 ] && return 0
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
card_class() {    # NVIDIA card position -> tc | pascal8 | p100 | maxwell | kepler
  local V=${GMEM[$1]} c=${GCC[$1]/./}
  if [ "$c" -ge 80 ] && [ "$V" -ge 9500 ]; then echo tc
  elif [ "$c" -ge 61 ] && [ "$V" -ge 7500 ]; then echo pascal8
  elif [ "$c" -eq 60 ] && [ "$V" -ge 12000 ]; then echo p100     # Tesla P100 (sm_60, no dp4a): docs/CARDS.md
  elif [ "$c" -lt 50 ]; then echo kepler
  else echo maxwell; fi
}
# Nanbeige 33B: the engine puts its layers in order on 1..N cards. The cards together must hold the 13.1 GB of
# weights, the K/V of a short context and about 700 MiB on each card. sm_80 and later only in 0.4.0 (sm_61 reads
# prompts in 8-token passes there and is not tested on a Pascal card)
NB_MIB=15500; NB_CARD_MIB=700
nb_cards() {      # NVIDIA card positions (candidates) -> the fewest of them, largest first, that hold Nanbeige
  local k got="" n=0 have=0 order
  order=$(for k in "$@"; do { [ "${GCC[$k]/./}" -ge 80 ] || [ "${GCC[$k]/./}" -eq 60 ]; } 2>/dev/null && echo "${GMEM[$k]} $k"; done | sort -k1,1nr -k2,2n | awk '{print $2}')
  for k in $order; do
    got="${got:+$got,}${GIDX[$k]}"; n=$((n + 1)); have=$((have + ${GMEM[$k]}))
    [ "$have" -ge $((NB_MIB + NB_CARD_MIB * n)) ] && { echo "$got"; return 0; }
  done
  return 1
}
card_models() {   # NVIDIA card position -> the models that run on it, the best first
  local V=${GMEM[$1]}
  case "$(card_class "$1")" in
    tc|pascal8) if [ "$V" -ge 5800 ]; then echo "27b lightning nemotron 9b 4b 2b"; else echo "27b nemotron 9b 4b 2b"; fi ;;
    p100) echo "27b lightning 9b 4b 2b" ;;
    # Kepler: Lightning on the 12 GB cards (Quadro K6000); no Nemotron Bolt
    kepler) if [ "$V" -ge 11000 ]; then echo "27b lightning 9b 4b 2b"; elif [ "$V" -ge 3800 ]; then echo "9b 4b 2b"
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
  if [[ "${AGFX[$1]}" == gfx115* ]]; then
    # RDNA 3.5 (the Radeon 840M): the Ternary-Bonsai-2 models were measured; Nemotron was not
    [ "$V" -ge 12000 ] && m="27b"
    [ "$V" -ge 4000 ] && m="$m 9b"
    [ "$V" -ge 2500 ] && m="$m 4b"
    echo "$m 2b"; return 0
  fi
  [ "$V" -ge 12000 ] && [ "$RAM_MB" -ge 24000 ] && m="27b"
  [ "$V" -ge 6000 ] && m="$m nemotron"
  echo "$m 2b"
}
model_label() {
  case "$1" in
    27b) echo "Ternary-Bonsai-2 27B" ;;
    nemotron) echo "NVIDIA Nemotron 3.5 Bolt E4B" ;;
    lightning) echo "Nemotron 3.5 Lightning 30B-A3B (ternary)" ;;
    nanbeige) echo "AH-Nanbeige4.3-33B IQ3_XXS" ;;
    *) echo "Ternary-Bonsai-2 ${1^^}" ;;
  esac
}
# A setup is one or more servers. Each server is "backend|model|cards", where cards is a card list ("0,1": one model
# split over both), "kind:0 3 5" (any free card of this kind: each pick takes the next one) or "" (the CPU).
short() { case "$1" in nemotron) echo Nemotron ;; lightning) echo Lightning ;; nanbeige) echo Nanbeige ;; *) echo "${1^^}" ;; esac; }
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
    elif [[ "$GPU" == *,* ]] && ! { want nanbeige && nb_cards "$pa" "$pb" >/dev/null; }; then die "GPUs ${GPU} cannot run one model together (two Maxwell cards of 4 GB, such as two K1200s, can; Nanbeige 33B needs sm_80 cards or Tesla P100s with 16.2 GB in total, plus 0.7 GB for each more card)"
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
  if [ "$NG" -gt 0 ] && want nanbeige && [[ "$GPU" != amd* ]]; then
    case "$GPU" in auto|all) cand=$(seq 0 $((NG - 1))) ;; *,*) cand="$pa $pb" ;; *) cand=$(card_k "$GPU") || cand="" ;; esac
    if nbc=$(nb_cards $cand); then
      add_cfg "cuda|nanbeige|$nbc" "$(model_label nanbeige) on GPU$([[ "$nbc" == *,* ]] && echo "s ${nbc}, layers split over the cards" || echo " ${nbc}")"
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
    if [ "$M" = nanbeige ]; then   # the Nanbeige backend: layers in order over one or more cards (bench-0.4.0)
      PROFILE=nanbeige; C=${CTX/auto/16384}; MEM=10g
      ENV+=(-e KINDI_REPEAT_STOP=1 -e KINDI_MAX_TOKENS=16384)
    elif [[ "$CARDS" == *,* ]]; then
      PROFILE=k1200x2; C=${CTX/auto/24576}; MEM=12g
      ENV+=(-e KINDI_CARDS=2 -e KINDI_REPEAT_STOP=1 -e KINDI_CHECKPOINT_BUDGET_MB=4096)
    else
      PROFILE=$(card_class "$k")
      case "$PROFILE" in
        tc) C=${CTX/auto/131072}; MEM=14g; [ "$M" = 2b ] && NEED_MTP=1
            ENV+=(-e KINDI_CARDS=1 -e KINDI_REPEAT_STOP=1 -e KINDI_CHECKPOINT_BUDGET_MB=1280 -e KINDI_PARK_MAX=1
                  -e KINDI_DENSE_SPACING=2048 -e KINDI_PREFIX_MAX=2 -e KINDI_MAX_TOKENS=65536)
            if [ "$M" = 27b ]; then   # the profile of the RTX 3080 cluster servers (docs/CARDS.md)
              [ "${NO_DSPINE:-0}" = 0 ] && NEED_DSPINE=1
              ENV+=(-e KINDI_TC=1 -e KINDI_FD=1 -e KINDI_ATT_TC=1 -e KINDI_CKPT_VRAM=1 -e KINDI_MEM_RESERVE_MB=160 -e KINDI_SPEC=auto)
              if dspine_ready; then
                # auto (docs/DSPINE.md): the DSpine block drafter, the current token + 7 drafts per verify pass
                ENV+=(-e KINDI_DSPINE=/data/dspine -e KINDI_TC_TMAX=8)
              else
                # auto (docs/MTP.md): the 2B drafter, k=2 below 1K tokens of context and k=3 from 1K
                ENV+=(-e KINDI_SPEC_K=3 -e KINDI_SPEC_PACK=/data/pack-2b -e KINDI_SPEC_WINDOW=8192 -e KINDI_TC_TMAX=4)
                XPACKS+=(2b)
              fi
            elif [ "$M" = 2b ] && mtp_ready; then   # the 2B with its own MTP head: the plan engine on the tensor cores
              ENV+=(-e KINDI_TC=1 -e KINDI_TP=1 -e KINDI_FD=1 -e KINDI_ATT_TC=1 -e KINDI_TC_TMAX=4
                    -e KINDI_SPEC=auto -e "KINDI_MTP_HEAD=/data/mtp/$MTPF")
            fi ;;
        pascal8) C=${CTX/auto/65536}; MEM=14g; ENV+=(-e KINDI_CARDS=1 -e KINDI_REPEAT_STOP=1 -e KINDI_MAX_TOKENS=65536) ;;
        # Tesla P100 (sm_60, no dp4a): the plain engines, no speculation (on sm_60 a verify pass costs about one
        # step per token, so drafting is slower: docs/CARDS.md)
        p100) C=${CTX/auto/65536}; MEM=14g; ENV+=(-e KINDI_CARDS=1 -e KINDI_REPEAT_STOP=1 -e KINDI_MAX_TOKENS=65536) ;;
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
  # Lightning: 32K by default (more context takes card memory from the experts on 6 to 8 GB cards); the host
  # experts are pinned host memory, so the container needs room for them
  if [ "$M" = lightning ]; then C=${CTX/auto/32768}; [ -z "$MEM" ] || [ "${MEM%g}" -lt 14 ] && MEM=14g; fi
  [ -n "$MEMORY" ] && MEM="$MEMORY"
  return 0
}
model_name() { case "$1" in nemotron) echo "nemotron-bolt-e4b" ;; lightning) echo "nemotron-lightning-30b" ;; nanbeige) echo "nanbeige-33b-iq3" ;; *) echo "bonsai2-$1" ;; esac; }

PACKS=(); BACKENDS=" "; ram_need=0; cwsr=0
for i in $(seq 0 $((NS - 1))); do
  profile_of "$i"; M=${SV_M[$i]}
  [ -n "${GGUF[$M]:-}" ] || die "unknown model '${M}' (2b, 4b, 9b, 27b, lightning, nemotron or nanbeige)"
  for p in "$M" ${XPACKS[@]+"${XPACKS[@]}"}; do case " ${PACKS[*]+${PACKS[*]}} " in *" $p "*) ;; *) PACKS+=("$p") ;; esac; done
  case "$BACKENDS" in *" ${SV_B[$i]} "*) ;; *) BACKENDS="$BACKENDS${SV_B[$i]} " ;; esac
  [[ "$MEM" =~ ^[0-9]+g$ ]] && ram_need=$((ram_need + ${MEM%g} * 1024))
  [ "${SV_B[$i]}" = rocm ] && [ "${AGFX[${SV_C[$i]}]}" = gfx1103 ] && cwsr=1
  [ "$M" = nanbeige ] && info "model card: ${NB_CARD_URL}"
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
# AMD: ROCm 6.3 for RDNA3 (gfx110x, the measured 780M setup), ROCm 7 (HIP 7.16) for RDNA 3.5 (gfx115x), which ROCm 6.3
# does not know. image_of BACKEND [SERVER]
rocm_image() { case "$1" in gfx115*) echo "kindi-rocm7:${KINDI_VERSION}" ;; *) echo "kindi-rocm:${KINDI_VERSION}" ;; esac; }
image_of() {
  if [ "$1" = rocm ]; then rocm_image "${AGFX[${SV_C[${2:-0}]}]:-gfx1100}"
  else echo "kindi:${KINDI_VERSION}"; fi
}
UIDGID="$(id -u):$(id -g)"
# the images run as a fixed user (1000:1000); rootful Docker runs them as this user, rootless Docker as the
# container's root, which is this user on the host (the image's own user would be a subuid that cannot write $DIR)
USERARG=(-u "$UIDGID"); [ "$ROOTLESS" = 1 ] && USERARG=(-u 0:0)

# ------------------------------------------------------------------------------------------------ 4. image
src_id=""
# one image for each backend; the AMD backend can need both ROCm images (an RDNA3 and an RDNA 3.5 GPU)
IMAGES=""
for i in $(seq 0 $((NS - 1))); do
  e="${SV_B[$i]}|$(image_of "${SV_B[$i]}" "$i")"; case " $IMAGES " in *" $e "*) ;; *) IMAGES="$IMAGES $e" ;; esac
done
for e in $IMAGES; do
  b=${e%%|*}; IMAGE=${e#*|}
  if [ "$b" = rocm ] && [[ "$IMAGE" == kindi-rocm7:* ]]; then
    DOCKERFILE=docker/Dockerfile.rocm7
    step "Build the Kindi image (${IMAGE})" "The image holds ROCm 7 (HIP 7.16, for RDNA 3.5 GPUs: the Radeon 840M and 8060S), CuPy 14.2 and Kindi. The first build downloads about 9 GB and takes 10-20 minutes."
  elif [ "$b" = rocm ]; then
    DOCKERFILE=docker/Dockerfile.rocm
    step "Build the Kindi image (${IMAGE})" "The image holds ROCm 6.3, CuPy 14.2 (built for ROCm) and Kindi. The first build downloads about 6 GB and takes 30-45 minutes."
  else
    DOCKERFILE=docker/Dockerfile
    step "Build the Kindi image (${IMAGE})" "The image holds CUDA 10.2, CuPy 9.6 and Kindi. The first build downloads about 3 GB and takes 10-15 minutes."
  fi
  if [ -n "$SOURCE" ] && [ -d "$SOURCE" ]; then SRC="$SOURCE"
  else
    SRC="$DIR/src-${KINDI_VERSION}"
    tarball="${SOURCE:-$SOURCE_URL}"
    # an explicit --source tarball or URL is unpacked again on every run: a rerun with newer source must not keep
    # the tree of the first run (the default release tarball of this version does not change, so it is kept)
    if [ ! -f "$SRC/$DOCKERFILE" ] || [ -n "$SOURCE" ]; then
      [ -n "$SOURCE" ] && run rm -rf "$SRC"
      run mkdir -p "$SRC"
      if [ -f "$tarball" ]; then run tar -xzf "$tarball" -C "$SRC" --strip-components=1
      else run sh -c "curl -fsSL '$tarball' | tar -xz -C '$SRC' --strip-components=1"; fi
    fi
  fi
  [ "$DRY" = 1 ] || src_id=$(cd "$SRC" && find kindi docker -type f | sort | xargs cat | sha256sum | cut -c1-12)
  have=$($DOCKER image inspect -f '{{index .Config.Labels "kindi.source"}}' "$IMAGE" 2>/dev/null || true)
  if [ -n "$have" ] && [ "$have" = "$src_id" ]; then ok "${IMAGE} is up to date (source ${src_id})"
  else
    [ -n "$have" ] && info "the image is from other source (${have}): upgrading"
    run $DOCKER build -q --label "kindi.source=${src_id}" -t "$IMAGE" -f "$SRC/$DOCKERFILE" "$SRC" >/dev/null
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
# the daemon finds the NVIDIA toolkit only when it starts: check '--gpus' before convert and verify
nv=0; for b in $BACKENDS; do [ "$b" != cpu ] && [ "$b" != rocm ] && nv=1; done
if [ "$GPU_MODE" = toolkit ] && [ "$nv" = 1 ] && [ "$DRY" = 0 ]; then
  gpu_ok() { $DOCKER run --rm --gpus all --entrypoint true "kindi:${KINDI_VERSION}" >/dev/null 2>&1; }
  if ! gpu_ok; then
    warn "Docker cannot give the container the GPUs ('could not select device driver'): restart the Docker daemon after the toolkit install"
    if ask "Restart the Docker daemon now?"; then [ "$ROOTLESS" = 1 ] || need_sudo; restart_docker; fi
    if [ "$ROOTLESS" = 1 ]; then rc="systemctl --user restart docker"; else rc="sudo systemctl restart docker"; fi
    gpu_ok || die "'docker run --gpus all' does not work. Run '${rc}', then run this installer again"
  fi
  ok "docker run --gpus all works"
fi

# ------------------------------------------------------------------------------------------------ 5. download
step "Download the model$([ "${#PACKS[@]}" -gt 1 ] && echo s)" "From Hugging Face (Prism ML; unsloth for Nemotron Bolt; boro64 for Nemotron Lightning; ahtx for Nanbeige). The sha256 of each file must match the value in this script."
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

# the MTP head, for tensor-core profiles (sha256 pinned above); without it the servers run as before
if [ "${NEED_MTP:-0}" = 1 ] && ! mtp_ready; then
  if [[ "$MTP_URL" == *downloads.invalid* ]]; then
    info "MTP head: no download location is set yet (KINDI_MTP_URL); MTP stays off"
  else
    [ "$DRY" = 1 ] || mkdir -p "$DIR/mtp"
    f="$DIR/mtp/$MTPF"
    if run curl -fL --retry 5 --retry-delay 5 -C - -sS -o "$f.part" "$MTP_URL/$MTPF"; then
      if [ "$DRY" = 0 ]; then
        got=$(sha256sum "$f.part" | cut -d' ' -f1)
        if [ "$got" = "$MTPSHA" ]; then mv "$f.part" "$f"; touch "$f.sha256-ok"
        else rm -f "$f.part"; info "MTP head: sha256 ${got} is not the expected ${MTPSHA} (deleted); MTP stays off"; fi
      fi
    else
      info "MTP head: the download failed; MTP stays off"
    fi
  fi
fi
mtp_ready && ok "MTP is on: KINDI_SPEC=auto with ${MTPF} (sha256 ${MTPSHA:0:16}...; docs/MTP.md)"

# the DSpine drafter for the 27B on tensor-core cards (sha256 pinned above); without it the 27B drafts with the 2B
if [ "${NEED_DSPINE:-0}" = 1 ] && ! dspine_ready; then
  info "DSpine drafter: Tencent's Qwen3.8-27B-DSpine-b16, Apache-2.0 ($(human "$DSPSIZE")); --no-dspine leaves it out"
  [ "$DRY" = 1 ] || mkdir -p "$DIR/dspine"
  okd=1
  for j in 0 1 2; do
    f="$DIR/dspine/${DSPF[$j]}"
    [ -f "$f.sha256-ok" ] && continue
    if [ -t 2 ]; then prog=--progress-bar; else prog=-sS; fi
    if run curl -fL --retry 5 --retry-delay 5 -C - "$prog" -o "$f.part" "$DSP_URL/${DSPF[$j]}"; then
      if [ "$DRY" = 0 ]; then
        got=$(sha256sum "$f.part" | cut -d' ' -f1)
        if [ "$got" = "${DSPSHA[$j]}" ]; then mv "$f.part" "$f"; touch "$f.sha256-ok"
        else rm -f "$f.part"; okd=0; info "DSpine ${DSPF[$j]}: sha256 ${got} is not the expected ${DSPSHA[$j]} (deleted)"; fi
      fi
    else
      okd=0; info "DSpine ${DSPF[$j]}: the download failed"
    fi
  done
  [ "$okd" = 1 ] || info "DSpine stays off: the 27B drafts with the 2B (run the installer again to retry)"
fi
if dspine_ready; then
  ok "DSpine is on: KINDI_SPEC=auto drafts the 27B with Tencent's DSpine (sha256 ${DSPSHA[2]:0:16}...; docs/DSPINE.md)"
  info "NOTICE: the DSpine weights are (c) Tencent, under the Apache License 2.0 ($DIR/dspine/LICENSE); Kindi's drafter"
  info "        code is Kindi's own. The weights are downloaded from Hugging Face, not redistributed by Kindi."
  [ "$DRY" = 1 ] || cat > "$DIR/dspine/NOTICE" <<NOTICE_EOF
DSpine draft model weights: tencent/Qwen3.8-27B-DSpine-b16 (Hugging Face), Copyright Tencent, licensed under the
Apache License, Version 2.0 (LICENSE in this folder). Downloaded at revision 5f54174d301dbca65d9aa4448dc47e56732d9a12.
Kindi uses them unmodified (quantized to int4 in memory at load); Kindi's drafter code (kindi/dspine.py, kindi/dspine.cu)
is its own implementation.
NOTICE_EOF
fi

# ------------------------------------------------------------------------------------------------ 6. convert + verify
step "Convert and verify" "kindi convert builds Kindi's weight pack; kindi verify checks the output against llama.cpp, token for token, once for each model on each kind of card."
state_of() {      # the cache folder of server i: one per model, and one per card when a model runs more than once
  local i=$1 j n=0
  for j in $(seq 0 $((NS - 1))); do [ "${SV_M[$j]}" = "${SV_M[$i]}" ] && n=$((n + 1)); done
  if [ "$n" -gt 1 ]; then echo "state-${SV_M[$i]}-${SV_B[$i]}${SV_C[$i]//,/-}"; else echo "state-${SV_M[$i]}"; fi
}
base_of() {
  local sd; sd=$(state_of "$1")
  BASE=(${USERARG[@]+"${USERARG[@]}"} -v "$DIR:/data" -e HOME=/tmp -e "CUPY_CACHE_DIR=/data/${sd}/cupy" -e "CUDA_CACHE_PATH=/data/${sd}/cudacache"
        -e "KINDI_STATE_DIR=/data/${sd}" -e KINDI_CPU_CACHE=/data/cpu)
}
for p in ${PACKS[@]+"${PACKS[@]}"}; do
  pk="$DIR/pack-$p"
  if [ -f "$pk/kindi_pack.json" ] && grep -q "${SHA[$p]}" "$pk/kindi_pack.json"; then ok "pack-$p is up to date"; continue; fi
  base_of 0; ci=$(image_of "${SV_B[0]}" 0)
  run $DOCKER run --rm ${BASE[@]+"${BASE[@]}"} "$ci" convert "/data/models/${GGUF[$p]}" --out "/data/pack-$p" --force
  ok "pack-$p converted"
done
# this installer's servers hold the GPUs: stop them for the checks (the chosen ones start again below)
for c in $OURS $(printf '%s ' "${SV_NAME[@]}"); do
  if $DOCKER ps --format '{{.Names}}' 2>/dev/null | grep -x "$c" >/dev/null; then info "stopping '${c}' for the check"; run $DOCKER stop "$c" >/dev/null; fi
done
SPEED=(); declare -A SPK
for i in $(seq 0 $((NS - 1))); do
  profile_of "$i"; M=${SV_M[$i]}; b=${SV_B[$i]}
  if [ "$b" = rocm ]; then kind="amd/${AGFX[${SV_C[$i]}]}"; elif [ "$b" = cpu ]; then kind=cpu
  else k=$(card_k "${SV_C[$i]%%,*}"); kind="${GNAME[$k]}/${GMEM[$k]}/${SV_C[$i]//[0-9]/}"; fi
  key="$M/$PROFILE/$kind"
  if [ -n "${SPK[$key]:-}" ]; then SPEED+=("${SPK[$key]}"); ok "server $((i + 1)): checked above (the same model on the same kind of card)"; continue; fi
  [ "$M" = 27b ] && [ "$PROFILE" != k1200x2 ] && [ "$PROFILE" != kepler ] && info "the first 27B start writes its 5-trit weights into the pack (5.6 GB, up to 15 minutes)"
  base_of "$i"; gpu_args "$b"; IMAGE=$(image_of "$b" "$i")
  VERIFY_OUT="$DIR/verify-$((i + 1)).log"; sp="?"
  if [ "$DRY" = 1 ]; then info "\$ $DOCKER run --rm ${BASE[*]} ${GARGS[*]:-} ${ENV[*]:-} -e KINDI_PACK=/data/pack-$M $IMAGE verify"
  else
    VX=(); [ -n "$CPUS" ] && VX=(--cpus "$CPUS")
    if $DOCKER run --rm ${VX[@]+"${VX[@]}"} ${BASE[@]+"${BASE[@]}"} ${GARGS[@]+"${GARGS[@]}"} ${ENV[@]+"${ENV[@]}"} -e "KINDI_PACK=/data/pack-$M" -e KINDI_HEAD=exact "$IMAGE" verify > "$VERIFY_OUT" 2>&1; then
      grep -E "PASS|FAIL" "$VERIFY_OUT" | sed 's/^\[[^]]*\] /    /' | sed -n 1,6p
      sp=$(grep -o '[0-9.]* tokens/s over the run' "$VERIFY_OUT" | awk '{print $1}')
      if [ "$M" = nanbeige ] || [ "$M" = lightning ]; then ok "server $((i + 1)): agrees with llama.cpp on each prompt, up to a llama.cpp tie (${sp} tokens/s over the check)"
      elif [ "$M" = nemotron ]; then ok "server $((i + 1)): agrees with llama.cpp on the first 12 tokens or more of each prompt (${sp} tokens/s over the check)"
      else ok "server $((i + 1)): token-exact with llama.cpp (${sp} tokens/s over the check, prompt tokens included)"; fi
    else
      tail -15 "$VERIFY_OUT" | sed 's/^/    /'
      die "kindi verify did not pass for server $((i + 1)) (log: ${VERIFY_OUT}). No server is started."
    fi
  fi
  SPEED+=("$sp"); SPK[$key]=$sp
done

# ------------------------------------------------------------------------------------------------ 7. service
# network access: 127.0.0.1 only (the default), or the local network with --lan. The API key goes to the
# containers through an env file (mode 600), so it is not on a command line or in the dry-run output.
BIND=127.0.0.1; KEYFILE="$DIR/api-key"; KEYENV="$DIR/api-key.env"; KEY=""
if [ "$APIKEY" = none ]; then KEY=""
elif [ -n "$APIKEY" ]; then KEY=$APIKEY
elif [ "$LAN" = 1 ]; then
  if [ -s "$KEYFILE" ]; then KEY=$(cat "$KEYFILE")      # the key of an earlier --lan install: clients keep working
  else KEY="kindi-$(od -An -N24 -tx1 /dev/urandom | tr -d ' \n')"; fi
fi
[ "$LAN" = 1 ] && BIND=0.0.0.0
if [ "$DRY" = 0 ]; then
  if [ -n "$KEY" ]; then
    (umask 077; printf '%s\n' "$KEY" > "$KEYFILE"; printf 'KINDI_API_KEY=%s\n' "$KEY" > "$KEYENV")
  else rm -f "$KEYENV"; fi
fi
if [ "$LAN" = 1 ]; then
  step "Start the server$([ "$NS" -gt 1 ] && echo s)" "Docker containers with restart=unless-stopped: they start again after a reboot. --lan: other computers on this network can reach them."
  warn "--lan: anyone who can reach this computer on the network can use these servers$([ -n "$KEY" ] && echo " if they have the API key" || echo ", with no key (--api-key none)")."
  warn "Use --lan only on a network that you trust. Do not forward these ports from the internet."
  [ -z "$KEY" ] && warn "No API key: any program on the network can send requests and read the answers."
else
  step "Start the server$([ "$NS" -gt 1 ] && echo s)" "Docker containers with restart=unless-stopped: they start again after a reboot. Only this machine can reach them (127.0.0.1)."
fi
for c in $OURS; do      # an earlier install's servers that this setup does not keep
  case " ${SV_NAME[*]} " in *" $c "*) ;; *)
    if $DOCKER ps -a --format '{{.Names}}' | grep -x "$c" >/dev/null; then info "removing the earlier server '${c}'"; run $DOCKER rm -f "$c" >/dev/null; fi ;; esac
done
GOT_CTX=(); GOT_VER=$KINDI_VERSION; GOT_SHA=""; JSON=""
for i in $(seq 0 $((NS - 1))); do
  profile_of "$i"; M=${SV_M[$i]}; b=${SV_B[$i]}; nm=${SV_NAME[$i]}; pt=${SV_PORT[$i]}
  base_of "$i"; gpu_args "$b"; IMAGE=$(image_of "$b" "$i")
  $DOCKER ps -a --format '{{.Names}}' 2>/dev/null | grep -x "$nm" >/dev/null && run $DOCKER rm -f "$nm" >/dev/null
  SARGS=(-d --name "$nm" --restart unless-stopped ${BASE[@]+"${BASE[@]}"} ${GARGS[@]+"${GARGS[@]}"} ${ENV[@]+"${ENV[@]}"} -p "${BIND}:${pt}:8093"
         -e "KINDI_PACK=/data/pack-$M" -e "KINDI_CTX=$C" -e KINDI_HEAD=exact -e "KINDI_MODEL_NAME=$(model_name "$M")" -e KINDI_HOST=0.0.0.0)
  [ -n "$KEY" ] && SARGS+=(--env-file "$KEYENV")
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

# ------------------------------------------------------------------------------------------------ Kindi Bridge (optional)
BRIDGE_DONE=0
[ "$BRIDGE" = ask ] && [ -f "$BRIDGE_RECORD" ] && BRIDGE=yes      # installed before: upgrade it too
if [ "$BRIDGE" = ask ] && [ "$YES" = 0 ] && [ "$DRY" = 0 ] && [ -r /dev/tty ]; then
  echo
  info "Optional: Kindi Bridge, a terminal workspace for agents on Kindi (conversations, scheduled prompts,"
  info "completion checks, local knowledge folders and live Kindi telemetry). You can add it later with --with-bridge."
  ask "Install Kindi Bridge too?" && BRIDGE=yes
fi
[ "$BRIDGE" = yes ] && bridge_install "${SV_PORT[0]}"

# ------------------------------------------------------------------------------------------------ done
echo
echo "${G}${B}Kindi is running$([ "$NS" -gt 1 ] && echo ": ${NS} servers").${N}"
echo
echo "  Kindi      ${GOT_VER}${GOT_SHA:+ (commit ${GOT_SHA})}"
for i in $(seq 0 $((NS - 1))); do
  echo "  ${SV_NAME[$i]}  http://127.0.0.1:${SV_PORT[$i]}/v1  $(model_name "${SV_M[$i]}") on $([ "${SV_B[$i]}" = cpu ] && echo "the CPU" || echo "$([ "${SV_B[$i]}" = rocm ] && echo "AMD ")GPU ${SV_C[$i]}"), ${GOT_CTX[$i]:-?} tokens, ${SPEED[$i]:-?} tokens/s in the check"
done
URL="http://127.0.0.1:${SV_PORT[0]}/v1"; MODEL_NAME=$(model_name "${SV_M[0]}")
AUTH_H=""; [ -n "$KEY" ] && AUTH_H=" -H \"Authorization: Bearer \$(cat ${KEYFILE})\""
if [ "$LAN" = 1 ]; then
  LANIP=$(hostname -I 2>/dev/null | awk '{print $1}')
  echo
  echo "  Local network: $(for i in $(seq 0 $((NS - 1))); do printf 'http://%s:%s/v1  ' "${LANIP:-<this computer>}" "${SV_PORT[$i]}"; done)"
  if [ -n "$KEY" ]; then echo "  API key: in ${KEYFILE} (clients send it as: Authorization: Bearer <key>)"
  else echo "  API key: none (--api-key none)"; fi
  command -v ufw >/dev/null && ufw status 2>/dev/null | grep -q "Status: active" && \
    echo "  The ufw firewall is on: allow the port(s) for your network: sudo ufw allow from <your network>/24 to any port ${SV_PORT[0]}"
fi
cat <<EOF

  Each server is an OpenAI-compatible endpoint; status: /health on its port. Logs: ${DOCKER} logs -f ${SV_NAME[0]}
  The containers restart unless stopped.

Try it:
  curl ${URL}/chat/completions -H 'Content-Type: application/json'${AUTH_H} \\
    -d '{"messages": [{"role": "user", "content": "Hello!"}]}'

OpenAI clients:
  export OPENAI_BASE_URL=${URL}  OPENAI_API_KEY=$([ -n "$KEY" ] && echo "\$(cat ${KEYFILE})" || echo none)

pi (~/.pi/agent/models.json; one provider for each server):
  {"providers": {
$(for i in $(seq 0 $((NS - 1))); do mn=$(model_name "${SV_M[$i]}"); pv=${SV_NAME[$i]}
  printf '    "%s": {"baseUrl": "http://127.0.0.1:%s/v1", "api": "openai-completions", "apiKey": "%s",\n' "$pv" "${SV_PORT[$i]}" "$([ -n "$KEY" ] && echo "<the key in ${KEYFILE}>" || echo none)"
  printf '      "compat": {"supportsDeveloperRole": false, "supportsReasoningEffort": false,\n'
  printf '                 "thinkingFormat": "qwen-chat-template", "thinkingTokenBudgetField": "thinking_budget_tokens"},\n'
  printf '      "models": [{"id": "%s", "name": "%s", "reasoning": true, "input": ["text"],\n' "$mn" "$mn"
  printf '                  "contextWindow": %s, "maxTokens": 4096}]}%s\n' "${GOT_CTX[$i]:-24576}" "$([ "$i" -lt $((NS - 1)) ] && echo ,)"
done)
  }}
  (the thinking format sends chat_template_kwargs.enable_thinking, which both Kindi model families read)

Run this installer again to upgrade, repair or change the setup. Remove Kindi: bash install.sh --uninstall [--purge]
EOF
if [ "$BRIDGE_DONE" = 1 ]; then cat <<EOF

Kindi Bridge:
  kindi-bridge        opens the workspace (Agents > New agent starts an agent on Kindi; Help > Getting started)
  Over SSH: ssh -t HOST kindi-bridge. Detach with View > Detach: the agents keep running.
EOF
fi
