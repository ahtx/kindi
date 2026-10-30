# The one-command installer

`install.sh` installs Kindi with one command. It finds your NVIDIA and AMD graphics, shows each setup that your
computer can run, with the best one first, and then downloads and checks the models, converts them, checks the
output against llama.cpp and starts the servers as services.

```bash
curl -fsSL https://amirhusain.com/kindi/install.sh | bash
```

To see each step and each command before anything changes, do a dry run first:

```bash
curl -fsSL https://amirhusain.com/kindi/install.sh | bash -s -- --dry-run
```

## Install with a checksum check

Download the script, compare its sha256 with the value below, read it, and then run it:

```bash
curl -fsSLO https://amirhusain.com/kindi/install.sh
echo "4079515dd373384cd4e68e8a747ba9e667bf36d81ac3132f87886273ae96d38a  install.sh" | sha256sum -c -
bash install.sh --dry-run
bash install.sh
```

| File | sha256 |
|---|---|
| `install.sh` (Kindi 0.3.0) | `4079515dd373384cd4e68e8a747ba9e667bf36d81ac3132f87886273ae96d38a` |

If `sha256sum -c` does not print `install.sh: OK`, do not run the script. The same script is in the repository root
and in the 0.3.0 release.

## What it does

| Step | What it does | sudo |
|---|---|---|
| 1. Check the machine | Linux x86-64, RAM, free disk, each NVIDIA GPU with its compute capability, and each AMD RDNA3 GPU. No GPU: the CPU engine. | no |
| 2. Docker and GPU access | Finds Docker, and the NVIDIA Container Toolkit or the AMD devices `/dev/kfd` and `/dev/dri`. If something is missing, it asks before it installs it (apt). Without the toolkit, it gives the container the NVIDIA devices and the driver libraries directly. | only to install |
| 3. Select the setup | Shows each setup for this computer, the recommended one first, and asks which one (below). | no |
| 4. Build the image | Builds the dependency base image once, then `kindi:0.3.0` (CUDA 10.2) or `kindi-rocm:0.3.0` (ROCm 6.3) on top of it from the release source, and checks that the Kindi modules load. A re-run with the same source does not build again. | no |
| 5. Download the models | Downloads each GGUF from Hugging Face. The sha256 must match the value in the script (table below). | no |
| 6. Convert and verify | `kindi convert`, then `kindi verify` one time for each model on each kind of card. | no |
| 7. Start the servers | One Docker container for each server, with `--restart unless-stopped`, on `127.0.0.1`. It waits until each `/health` answers. | no |

At the end it shows each server's address, model, context and the speed of the check, and the settings for OpenAI
clients and for pi (one provider for each server). It writes no secrets and asks for no keys. The servers listen on
127.0.0.1 only.

If your user is not in the `docker` group, the script uses `sudo docker`.

## Select a setup

The installer lists each setup that your computer can run. A computer with an RTX 3060 and a GTX 1080 shows:

```text
    1) every GPU, one server each: 27B on GPU 0 + 27B on GPU 1 (recommended)
    2) Ternary-Bonsai-2 27B on GPU 0 (NVIDIA GeForce RTX 3060)
    3) NVIDIA Nemotron 3.5 Bolt E4B on GPU 0 (NVIDIA GeForce RTX 3060)
    ...
    7) Ternary-Bonsai-2 27B on GPU 1 (NVIDIA GeForce GTX 1080)
    ...
    12) Ternary-Bonsai-2 2B on the CPU (80 threads; slow, no GPU needed)
```

- One number gives one server.
- Two or more numbers, for example `2,8`, give one server for each setup, side by side.
- `NxK` gives setup N on K cards of the same kind, for example `5x2` for a 4B on each of two K1200 cards.

A kind of card shows one time in the list, also when the computer has many cards of that kind. A card has at most one
server. Each server gets its own container (`kindi`, `kindi-2`, ...), the next free port from 8093, and its own
cache folder when a model runs more than one time.

To select without questions, add `--auto`: the installer takes the recommended setup and continues. In a script, add
`--config` and the numbers, for example `--config 2,8`. `--dry-run` prints the list.

### The recommended setup

| Your computer | Recommended | Other setups in the list |
|---|---|---|
| No GPU | 2B on the CPU | none |
| Quadro K420 1 GB | 2B | 2B on the CPU |
| Quadro K420 2 GB | 4B | 2B, 2B on the CPU |
| One Quadro K1200 4 GB | 9B | Nemotron, 4B, 2B, 2B on the CPU |
| Two Quadro K1200 4 GB | 27B split over both cards (tensor parallel) | Nemotron split over both cards; 9B, Nemotron, 4B or 2B on each card; 2B on the CPU |
| GeForce GTX 1080 8 GB | 27B | Nemotron, 9B, 4B, 2B, 2B on the CPU |
| GeForce RTX 3060 / 3080 (sm_86, 9.5 GB or more) | 27B with tensor cores and the 2B draft model | Nemotron, 9B, 4B, 2B, 2B on the CPU |
| AMD Radeon 780M (gfx1103) | 27B | Nemotron, 2B, 2B on the CPU |
| Two or more GPUs that do not split one model | the best model on every GPU, one server each | each setup of each card |

| Profile | Cards | Context asked |
|---|---|---|
| cpu | no GPU, or `--gpu cpu` | 24,576 |
| kepler | sm_30-sm_37 | 24,576; 65,536 at 1.5 GB or more; 23,808 for the 4B on a 2 GB card |
| maxwell | other cards below sm_61 or below 7.5 GB | 24,576 |
| k1200x2 | two cards of 3.8 GB or more, sm_50-sm_60 | 24,576 |
| pascal8 | sm_61 or newer, 7.5 GB or more | 65,536 |
| tc | sm_80 or newer, 9.5 GB or more | 131,072 |
| rocm-gfx1103 | AMD Radeon 780M | 24,576 |

Nemotron always asks for 24,576. The engine fits the context to the free memory and reports what it got. For
example, Nemotron split over two K1200 cards gets 8,448.

## AMD graphics

The Kindi ROCm image holds ROCm 6.3.3, CuPy 14.2.0 and Kindi, so you do not install ROCm on the computer. The
computer needs the `amdgpu` driver of Linux (kernel 6.2 or newer and `linux-firmware`) and Docker. The installer gives
the container `/dev/kfd` and `/dev/dri`. On the Radeon 780M it also sets `HSA_OVERRIDE_GFX_VERSION=11.0.2` and
checks the kernel option `amdgpu.cwsr_enable=0` (see [AMD.md](AMD.md)). It asks before it adds that option, and it
never adds it with `--yes` or `--auto`.

## Kernel changes

The installer changes the kernel only when you answer `y` at the keyboard. `--yes` and `--auto` never answer these
questions; they only print what to do.

- **gpu-unlock** (GeForce RTX 30 cards): the installer tells you which cards are not unlocked and offers to download
  and install the third-party [gpu-unlock](https://github.com/Zagitta/gpu-unlock) module at the reviewed version.
  Kindi does not include it.
- **`amdgpu.cwsr_enable=0`** (Radeon 780M): see [AMD.md](AMD.md).

## Model files

The script pins these files. It does not use a file with a different sha256.

| Model | File | Size (bytes) | sha256 |
|---|---|---|---|
| 2B | `Ternary-Bonsai-2-2B-PTQ1_0.gguf` | 429,183,456 | `36061ad406df8a4903a6db005be350c50fabfaf717f9ef0acf4b76adcd9929fa` |
| 4B | `Ternary-Bonsai-2-4B-PTQ1_0.gguf` | 949,549,920 | `56711263ec9a9ffeaa74a3a0e58dece8127cf9268404746c721113ca76994d60` |
| 9B | `Ternary-Bonsai-2-9B-PTQ1_0.gguf` | 1,997,486,976 | `3561df7ac7d26a71ffa991ac2db740b10841c3a350aa2676ccd3ff5d428b4e8b` |
| 27B | `Ternary-Bonsai-2-27B-PTQ1_0.gguf` | 5,946,648,928 | `53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3` |
| Nemotron | `NVIDIA-Nemotron-3.5-Bolt-E4B-UD-Q3_K_XL.gguf` (unsloth) | 4,219,133,120 | `3f8032b9d2c18536c4ac51b97959a563cc27434de99823640c98147edc62ce9b` |

## Options

| Option | What it does |
|---|---|
| `--dry-run` | Shows each step, each command and the list of setups. It changes nothing. |
| `--yes`, `-y` | Answers yes to each question (for scripts) and takes the recommended setup. Kernel changes are never answered. |
| `--auto` | Takes the recommended setup and continues with no questions. Kernel changes are only described. |
| `--config N` | Setup N of the list; `N,M` for several servers; `NxK` for setup N on K cards. |
| `--model M` | Lists only the setups with this model: `2b`, `4b`, `9b`, `27b` or `nemotron`. |
| `--ctx N` | The context length in tokens. |
| `--gpu G` | `N` (one NVIDIA card), `N,M` (two cards), `amd` or `amdN`, `all`, or `cpu`. |
| `--port P` | The port of the first server on 127.0.0.1 (default 8093). Later servers take the next free ports. |
| `--dir D` | The data folder (default `~/.kindi`). |
| `--name N` | The name of the first container (default `kindi`; later ones are `N-2`, `N-3`, ...). |
| `--threads N` | The CPU engine: the number of threads. |
| `--memory M` | The container memory limit, for example `8g`. |
| `--source S` | The Kindi source: a folder, or a `.tar.gz` path or URL (default: the 0.3.0 release). |
| `--uninstall` | Removes each server container and image that the installer made. The models stay. |
| `--uninstall --purge` | Also removes the data folder. |

## Upgrade, repair and remove

Run the script again to upgrade, repair or change the setup. It keeps the steps that are already correct: the image
(same source), the models (sha256 already checked) and the packs (made from the same files). It always runs
`kindi verify` again and starts new containers. Servers of the earlier install that the new setup does not keep are
removed.

```bash
bash install.sh --uninstall          # the containers and the images; the models stay in ~/.kindi
bash install.sh --uninstall --purge  # also ~/.kindi
```

## Tested

Each test is a real install, a chat request, a tool-call request and `--uninstall --purge` (Kindi 0.3.0-dev,
2026-10-24).

| Computer | Setup | Result |
|---|---|---|
| 2× Quadro K1200 | 27B split over both cards | PASS: verify 4 of 4 token-exact, 13.4 tokens/s |
| 2× Quadro K1200 | Nemotron split over both cards | PASS: verify 4 of 4 (16-40 of 40 tokens agree), 11.6 tokens/s; context 8,448 |
| 2× Quadro K1200 | `5x2`: a 4B on each card | PASS: verify 4 of 4 token-exact, 39.9 tokens/s; both servers answer |
| GeForce GTX 1080 8 GB, without the toolkit | 27B | PASS: verify 4 of 4 token-exact, 30.3 tokens/s |
| Quadro K6000 12 GB (NVIDIA 470, in a KVM virtual machine) | 27B, context 156,672 | PASS: verify 4 of 4 token-exact, 20.4 tokens/s; the cache answers the second request |
| GeForce GTX 1080 8 GB, without the toolkit | Nemotron | PASS: verify 4 of 4 (16-40 of 40 tokens agree), 24.1 tokens/s |
| AMD Radeon 780M (Ryzen 9 7940HS, 64 GB), without ROCm on the host | 27B, ROCm image built by the installer | PASS: verify 4 of 4 token-exact, 9.7 tokens/s; context 24,576 |

For Nemotron, `kindi verify` compares the first 40 tokens of four prompts with llama.cpp's output for the same file.
A prompt passes when the first 12 or more tokens agree. Kindi's Nemotron engine computes in a different order than
llama.cpp, so long outputs can part on a near tie.
