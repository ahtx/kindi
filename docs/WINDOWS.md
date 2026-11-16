# Kindi on Windows

Kindi runs on Windows 10 and Windows 11 through WSL2. WSL2 is a small Linux virtual machine that is a part of
Windows. One PowerShell command installs everything: it turns on WSL2, makes a Linux system only for Kindi, installs
Docker, checks the GPU, and runs the normal Kindi installer inside. Then Windows programs use Kindi at
`http://localhost:8093/v1`, the same as on Linux.

## Install

Open PowerShell (you do not need "Run as administrator") and run:

```powershell
irm https://amirhusain.com/kindi/install.ps1 | iex
```

To see each step before anything changes, do a dry run first:

```powershell
& ([scriptblock]::Create((irm https://amirhusain.com/kindi/install.ps1))) -DryRun
```

The installer prints each step and the reason for it. It asks before it changes something.

**If Microsoft Defender or your company's security software stops the one-line command:** some security tools
block every "download and run" PowerShell command. Download the script, check it, and run it as a file instead:

```powershell
Invoke-WebRequest https://amirhusain.com/kindi/install.ps1 -OutFile install.ps1
Get-FileHash install.ps1        # compare with the sha256 in the release notes
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

When WSL2 starts for the first time, Windows can show a "Welcome to Windows Subsystem for Linux" window. You
can close it: the install goes on in its own window. Every run writes a log to `%LOCALAPPDATA%\Kindi\install.log`.

| Step | What happens |
|---|---|
| 1. Check the PC | An x86-64 PC, Windows 10 2004 (build 19041) or newer, virtualization, RAM, disk, the GPU and its Windows driver, the proxy |
| 2. Turn on WSL2 | Only if it is off. Windows asks for administrator approval once, and the PC restarts. The installer continues by itself after you sign in again |
| 3. A Linux system for Kindi | A new WSL2 distribution with the name `Kindi`. It is Ubuntu 24.04 from Canonical, with a sha256 check. Your other WSL distributions do not change. If `Kindi` exists, the installer reuses it |
| 4. Docker and the GPU | If you have Docker Desktop, Kindi uses it. If not, the installer puts Docker Engine and the NVIDIA container toolkit into the `Kindi` distribution. Then the installer checks that the GPU is visible inside WSL2 |
| 5. The Kindi installer | `install.sh` runs inside WSL2. It shows the setups for your GPU, the recommended one first, downloads and checks the model, and starts the server |
| 6. Windows items | A check that `http://localhost:8093` answers on Windows, Start-menu items, a desktop shortcut, and an entry in Settings > Apps |

Run the same command again to upgrade or repair Kindi. The installer keeps the models that you downloaded.

## What you need

- **An x86-64 PC** (Intel or AMD). Windows on ARM (for example Snapdragon laptops) cannot run Kindi.
- **Windows 10 version 2004 or newer, or Windows 11.** For the GPU in WSL2, you need Windows 11 or Windows 10 21H2
  (build 19044) or newer. On older builds, Kindi uses the CPU engine.
- **Virtualization on** in the BIOS/UEFI (Intel VT-x or AMD SVM). Most PCs have it on.
- **An NVIDIA GPU with a current Windows driver** from nvidia.com (Game Ready or Studio). Do not install a Linux NVIDIA
  driver inside WSL2: the Windows driver serves WSL2. Without an NVIDIA GPU, Kindi uses the CPU engine.
- **Disk:** about 15-40 GB, for Linux, Docker, the models and their packs.

**AMD GPUs:** Kindi's AMD engine needs the Linux amdgpu compute driver (`/dev/kfd`). WSL2 does not have it. On a
Windows PC with an AMD GPU, Kindi uses the CPU engine. For the AMD GPU speed, use native Linux.

## Use Kindi

- **Kindi Console** (Start menu > Kindi > Kindi Console, and on the desktop): the status of each server, a chat window,
  the logs, a restart, and the settings for OpenAI clients.
- **Kindi Shell (WSL)**: a Linux shell in the `Kindi` distribution. The `kindi` commands and `docker` work there.
- **Any OpenAI client on Windows:** set the base URL to `http://localhost:8093/v1` and the API key to `none`.
  See [AGENTS.md](AGENTS.md) for agents and chat clients.

```powershell
$env:OPENAI_BASE_URL = 'http://localhost:8093/v1'; $env:OPENAI_API_KEY = 'none'
```

WSL2 stops an idle Linux system after a short time. With Docker Engine inside the `Kindi` distribution, a small hidden
process (`Kindi WSL keep-alive` in the Startup folder) keeps it running, so the server is always ready after you sign
in. With Docker Desktop, Docker Desktop keeps the server running.

## Options

Add options after `-File install.ps1`, or after the `scriptblock` form above.

| Option | What it does |
|---|---|
| `-DryRun` | Print every step and command, change nothing |
| `-Check` | Only check the PC and say what the install would do |
| `-Yes` | Answer yes to every question |
| `-LinuxArgs '...'` | Options for `install.sh`, for example `'--model 27b'`, `'--gpu cpu'` or `'--config 2'` (see [INSTALL.md](INSTALL.md)) |
| `-Distro NAME` | Use another WSL2 distribution. The installer uses a distribution that you have again, and never replaces it. The installer asks before it adds Docker or turns on systemd in it |
| `-DistroDir PATH` | Where a new distribution keeps its disk (default `%LOCALAPPDATA%\Kindi\wsl`). Use it to put Kindi on a larger drive |
| `-DockerMode MODE` | `auto` (default: Docker Desktop if you have it), `desktop` or `engine` |
| `-WslMemoryGB N` | Give the WSL2 virtual machine N GB of RAM (see below) |
| `-WithBridge` | Also install Kindi Bridge, the agent workspace (see the Kindi Bridge chapter). The installer adds Python 3, tmux and Node.js 22 to the `Kindi` distribution for it, and a Start-menu item "Kindi Bridge" that opens it in Windows Terminal. Without this option, an interactive run asks (the default is no); `-Yes` alone never adds it |
| `-NoBridge` | Do not install Kindi Bridge and do not ask |
| `-NoDesktopShortcut` | No desktop shortcut |
| `-NoReboot` | After the installer turns on WSL2, do not restart. You restart, and the install continues after you sign in |
| `-Uninstall`, `-Purge` | Remove Kindi (see below) |

## RAM in WSL2

Windows gives the WSL2 virtual machine about half of the PC's RAM by default. The installer shows both numbers. The
27B and the larger models need more. To give WSL2 more RAM, run the installer with `-WslMemoryGB N`, for example
`-WslMemoryGB 48` on a 64 GB PC. This sets `memory=` in `%UserProfile%\.wslconfig` (the installer keeps a copy of the old file)
and runs `wsl --shutdown` once. That command stops all WSL distributions and Docker Desktop, so the installer asks
first.

## Proxies

The installer uses the Windows proxy setting (including PAC and WPAD), or the `HTTPS_PROXY` variable. It sets the
proxy for the downloads, for `apt` and Docker inside WSL2, and for `docker build`. A proxy that runs on the PC's own
`localhost` (for example Px or Cntlm) is not visible from WSL2, unless `networkingMode=mirrored` is set in
`.wslconfig`. Otherwise, give the proxy's network address in `HTTPS_PROXY`.

## Remove Kindi

Use Settings > Apps > Kindi (WSL2) > Uninstall, or Start menu > Kindi > Uninstall Kindi. It removes the Kindi servers
and image, the keep-alive process, the Start-menu items, the shortcut and the Settings entry. Then it asks if it
should also delete the `Kindi` distribution with its models. With `-Purge`, it deletes them without a question. A
distribution that the installer did not make (`-Distro`) is never deleted: only Kindi is removed from it.

```powershell
powershell -ExecutionPolicy Bypass -File "$env:LOCALAPPDATA\Kindi\install.ps1" -Uninstall
```

## Problems

| Message | What to do |
|---|---|
| "this PC has an ARM processor" | Kindi needs an x86-64 PC |
| "Windows build ... is too old for WSL2" | Settings > Windows Update: install the updates, then run the installer again |
| "virtualization is off in the BIOS/UEFI" | Restart the PC and open the firmware setup (often F2, F10, F12 or Del during start). Turn on Intel Virtualization Technology (VT-x) or AMD SVM Mode, save, and run the installer again |
| "the NVIDIA GPU does not reach WSL2" | Install the current NVIDIA Windows driver from nvidia.com and restart. Do not install a Linux driver inside WSL2. For the CPU engine only: `-LinuxArgs '--gpu cpu'` |
| "Docker Desktop does not answer in 'Kindi'" | Docker Desktop > Settings > Resources > WSL integration: turn on `Kindi`, then Apply & restart. Or use `-DockerMode engine` |
| "does not answer ... from Windows" | WSL2 forwards `localhost` by default. If `.wslconfig` has `localhostForwarding=false`, remove it and run `wsl --shutdown` |
| "systemd did not start" | Run `wsl --update` (WSL 0.67.6 or newer is needed), then run the installer again |

The Linux installer's own messages are in [INSTALL.md](INSTALL.md) and [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
