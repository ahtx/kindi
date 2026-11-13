<p align="center"><img src="assets/kindi-banner.svg?v=0.6.0" alt="Kindi 0.6.0: language models on low-cost GPUs, from a 1 GB Quadro K420 to the RTX 50 series, the CMP 170HX and Radeon graphics. Ternary Bonsai 2B to 27B, AH-Nanbeige4.3-33B, Nemotron 3.5 Lightning 30B-A3B and DeepSeek-V4.1-Flash." width="100%"></p>

# Kindi 0.6.0

[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)
[![Commercial licenses available](https://img.shields.io/badge/commercial%20licenses-available-green.svg)](COMMERCIAL-LICENSE.md)
[![Release v0.6.0](https://img.shields.io/badge/release-v0.6.0-lightgrey.svg)](https://github.com/ahtx/kindi/releases/tag/v0.6.0)

Kindi is a custom inference engine for language models on low-cost GPUs. It runs Prism ML's ternary
Bonsai models (2B, 4B, 9B and 27B), AH-Nanbeige4.3-33B, Nemotron 3.5 Lightning 30B-A3B and DeepSeek-V4.1-Flash.
Kindi supports NVIDIA cards from the 1 GB Quadro K420 (Kepler) to the RTX 50 series and the CMP 170HX. It also
supports AMD Radeon 780M, 840M and 8060S graphics, and x86-64 CPUs. The OpenAI-compatible server has a prompt
cache, tool calls, a thinking mode and speculative decoding. With the exact head, the greedy output is the same as
the output of llama.cpp for the same model file, token for token. Version 0.6.0 adds two mixture-of-experts models
that keep experts in host memory, the DSpine drafter for the 27B, and a one-command installer for Windows.

### Measured speed

Kindi and llama.cpp on the same card, with the same model file and the same context. Tokens per second, one stream,
greedy. The method and the other cards: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

| Card | Model | Context | Kindi read / write | llama.cpp read / write |
|---|---|---:|---:|---:|
| Quadro K420, 1 GB | Bonsai 2B | 512 | **33.3 / 20.8** | 1.2 / 1.65 |
| GeForce GTX 1080, 8 GB | Nemotron 3.5 Lightning 30B-A3B | 8,192 | **687 / 89.5** | 615 / 4.7 |
| GeForce RTX 2070 SUPER, 8 GB | Nemotron 3.5 Lightning 30B-A3B | 8,192 | **2,129 / 120.3** | 1,137 / 111.8 |
| Tesla P100, 16 GB | Nemotron 3.5 Lightning 30B-A3B | 8,192 | **702 / 144.8** | 506 / 118.4 |
| GeForce RTX 3080, 10 GB, 190 W | Nemotron 3.5 Lightning 30B-A3B | 32,768 | **3,777 / 276.6** | 2,138 / 182.4 |
| Two GeForce RTX 3080 | AH-Nanbeige4.3-33B IQ3_XXS | 8,192 | **1,267 / 40.9** | 1,249 / 33.3 |
| Radeon 8060S (Strix Halo) | Bonsai 27B | 8,192 | **333 / 28.42** | 292 / 24.34 |

On the 8 GB cards, the Lightning model does not fit on the card. Both engines keep some experts in host memory
(llama.cpp: `-ncmoe 9` on the GTX 1080 and `-ncmoe 12` on the RTX 2070 SUPER). On these two cards the write test
uses a real text as the prompt.

Two more results:

- **The 27B with DSpine.** On an RTX 3080 at 190 W and a context of 32,768 tokens, the 27B writes 115.5 tokens/s
  with the DSpine drafter, with the same output as plain decoding ([docs/DSPINE.md](docs/DSPINE.md)).
- **DeepSeek-V4.1-Flash (552B).** On one RTX 3060 12 GB, a Quadro M4000 for the most used experts and two Xeon
  E5-2673 v4 CPUs with the model in RAM, Kindi writes 18.7 tokens/s with its built-in DSpark drafter
  ([docs/DEEPSEEK.md](docs/DEEPSEEK.md)).

### Supported cards and models

| Hardware | Models |
|---|---|
| NVIDIA Kepler: Quadro K420 (1 and 2 GB), Quadro K6000 (12 GB) | Bonsai 2B and 4B; on the K6000, the 27B and Lightning |
| NVIDIA Maxwell: Quadro K1200, one or two cards | Bonsai 2B, 4B, 9B; 27B on two cards |
| NVIDIA Pascal: GeForce GTX 1080, Tesla P100 | Bonsai 2B to 27B, Lightning |
| NVIDIA GeForce RTX 20, 30, 40 and 50 series | Bonsai 2B to 27B (DSpine for the 27B on tensor-core cards), Lightning, AH-Nanbeige4.3-33B (RTX 30 and later) |
| NVIDIA CMP 170HX, 40 GB | Bonsai 27B, AH-Nanbeige4.3-33B |
| AMD Radeon 780M, 840M, 8060S | Bonsai 2B to 27B (the 27B needs 32 GB of system memory or more) |
| x86-64 CPU with AVX2 | Bonsai 2B, 4B and 9B (the CPU engine) |
| One NVIDIA card with 12 GB or more and about 560 GB of RAM | DeepSeek-V4.1-Flash |

The measured cards and the per-card settings: [docs/CARDS.md](docs/CARDS.md).

### Install

Linux x86-64:

```bash
curl -fsSL https://amirhusain.com/kindi/install.sh | bash
```

Windows 10 and 11 (PowerShell; the installer sets up WSL2):

```powershell
irm https://amirhusain.com/kindi/install.ps1 | iex
```

The installer finds your cards and shows the setups that they support, the recommended setup first. It downloads
the model, checks its sha256, runs `kindi verify` and starts the server on `127.0.0.1:8093`. Add `--dry-run` to see
each step first ([docs/INSTALL.md](docs/INSTALL.md), [docs/WINDOWS.md](docs/WINDOWS.md)). DeepSeek-V4.1-Flash has
its own installer ([docs/DEEPSEEK.md](docs/DEEPSEEK.md)).

### Links

- [CHANGELOG.md](CHANGELOG.md): what changed in each version
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md) and [docs/CARDS.md](docs/CARDS.md): the measurements and the settings
- [WHY.md](WHY.md): the one-page summary
- The manual: [How to get work done with Kindi](https://amirhusain.com/software/kindi/How-To-Get-Work-Done-with-Kindi-0.6.0.pdf)
- Project page: [amirhusain.com/kindi](https://amirhusain.com/kindi/)

By [Amir Husain](https://www.amirhusain.com).

## Contents

- [First steps after the install](#first-steps-after-the-install)
- [Requirements](#requirements)
- [Install](#install) - with Docker, or natively
- [Get the model and convert it](#get-the-model-and-convert-it)
- [Run the server](#run-the-server)
- [Use it](#use-it) - curl, Python, streaming, tools, agents
- [Exact and fast output heads](#exact-and-fast-output-heads)
- [The prompt cache](#the-prompt-cache)
- [Commands](#commands)
- [Configuration](#configuration)
- [Performance](#performance)
- [Agent benchmark](#agent-benchmark)
- [Limits](#limits)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## First steps after the install

1. Find the port of the server. At the end, the installer shows the port of each server. The first server uses
   port 8093, the second server uses 8094, and so on.
2. Make sure that the server runs. Open `http://localhost:8093/health` in a browser, or use
   `curl -s localhost:8093/health`. The answer shows the model, the context size and the server state.
3. Connect a chat client. Kindi uses the OpenAI API at `http://localhost:8093/v1`. In Open WebUI, LM Studio, Jan or
   pi, add an "OpenAI-compatible" connection with this address. The API key can be any text, unless you set an
   `api_key`. The model id is in `/health` (for example `bonsai2-2b`).
4. Send a test request with curl:

```sh
curl -s localhost:8093/v1/chat/completions -H "Content-Type: application/json"   -d '{"model": "bonsai2-2b", "messages": [{"role": "user", "content": "Say hello."}]}'
```

5. Or send it with Python (`pip install openai`):

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8093/v1", api_key="none")
r = client.chat.completions.create(model="bonsai2-2b", messages=[{"role": "user", "content": "Say hello."}])
print(r.choices[0].message.content)
```

If the server is on a different computer, use its address in place of `localhost`. By default, the server accepts
connections only from the same computer (see [Run the server](#run-the-server)). For more about clients, see
[Use it](#use-it) and [docs/AGENTS.md](docs/AGENTS.md).

## Requirements

| | |
|---|---|
| GPU | NVIDIA, compute capability 3.0 (Kepler) or later. Tested: Quadro K420 (1 GB and 2 GB), Quadro K1200 (one card, or two in tensor parallel), Quadro K6000, Tesla P100 16 GB, GeForce GTX 1080, GeForce RTX 3080. Supported, and `kindi verify` passes: the GeForce RTX 20, 30, 40 and 50 series (speed numbers are only for the cards in the tables below). AMD: Radeon 780M (RDNA3, gfx1103) with 32 GB of computer memory or more; Radeon 840M and Radeon 8060S (RDNA 3.5, gfx1152 and gfx1151); see [docs/AMD.md](docs/AMD.md). With no GPU, the CPU engine runs on x86-64 with AVX2. |
| Driver | Kepler: the NVIDIA **470** series (the last driver branch with Kepler support). Later cards: a current driver. The image has CUDA 10.2; the host needs only the driver. AMD: the `amdgpu` driver of Linux and the kernel option `amdgpu.cwsr_enable=0`; ROCm 6.3 is in the `kindi-rocm` image, and ROCm 7 (for RDNA 3.5) is in the `kindi-rocm7` image. |
| OS | Linux x86-64 (developed on Ubuntu 24.04 with Docker; the image itself is Ubuntu 18.04) |
| RAM / disk | 2B: 4 GB RAM, ~1.5 GB disk. 27B: ~14 GB of free RAM for its first start (two K1200 cards: ~5 GB) and ~19 GB of disk. The Docker image takes ~4 GB more. |
| Software | Docker with the NVIDIA Container Toolkit (or the installer's device passthrough) - **or** CUDA 10.2 + Python 3.6 + CuPy 9.6 (see below) |

[docs/CARDS.md](docs/CARDS.md) gives the model and the settings for each card. The card must be free: Kindi fits the
context to the free card memory. On the K420 1 GB, Kindi uses about 870 MB of its 980 MB with the default
24,576-token context. A desktop session on the same card takes a few MB and is fine.

## Install

### Containers: ready-made images

The images are on GitHub's container registry: `ghcr.io/ahtx/kindi:cuda` (NVIDIA and the CPU engine; also `latest`), `:rocm6` (AMD RDNA3), `:rocm7` (AMD RDNA 3.5). With Docker Compose or podman-compose, no clone and no build:

```sh
curl -fsSLO https://raw.githubusercontent.com/ahtx/kindi/main/compose.yaml
KINDI_MODEL=bonsai-4b docker compose up -d
```

The first start downloads the model (or uses a GGUF file in the volume), builds its pack once and starts the server
on `127.0.0.1:8093`. AMD, Podman, users and all the settings: [docs/CONTAINERS.md](docs/CONTAINERS.md).

### Build the Docker image yourself

CUDA 10.2 and CuPy 9.6 are the last releases that support Kepler, and they need the Ubuntu 18.04 userland; the image packages
exactly that.

```sh
git clone https://github.com/ahtx/kindi.git && cd kindi
docker build -t kindi:0.6.0 -f docker/Dockerfile .   # downloads the CUDA 10.2 toolkit (~2.5 GB), ~15 minutes
```

For AMD RDNA3 graphics, build `docker/Dockerfile.rocm` instead (`kindi-rocm:0.6.0`, about 6 GB to download). For
RDNA 3.5 graphics, build `docker/Dockerfile.rocm7` (`kindi-rocm7:0.6.0`, about 9 GB to download). See
[docs/AMD.md](docs/AMD.md).

You can also use the release archive. Download `kindi-0.4.0.tar.gz` from the
[v0.4.0 release](https://github.com/ahtx/kindi/releases/tag/v0.4.0), then run
`tar xzf kindi-0.4.0.tar.gz && cd kindi-0.4.0` and build the image with the same `docker build` command.

Check that Docker sees the GPU (create `~/.kindi` first, so it belongs to you and not to root):

```sh
mkdir -p ~/.kindi
docker run --rm --gpus all -u $(id -u):$(id -g) -v ~/.kindi:/data kindi:0.4.0 info
```

More in [docs/DOCKER.md](docs/DOCKER.md).

### Natively

You need CUDA 10.2 (toolkit with NVRTC and cuBLAS), Python 3.6 or later that CuPy 9.6 supports, and:

```sh
pip install "numpy==1.19.5" "fastrlock==0.8.3" "cupy-cuda102==9.6.0" "fastapi==0.79.1" "uvicorn==0.16.0" "pydantic==1.9.2"
```

Then run Kindi from the repository root with `python3 -m kindi ...` (put the repository on `PYTHONPATH` to run it
from elsewhere).

## Get the model and convert it

These steps use the 2B. For the 4B, 9B and 27B, use their PTQ1_0 files from the same publisher. The installer
selects and downloads the right one; [docs/INSTALL.md](docs/INSTALL.md) lists the files and their sha256. Download
Prism ML's PTQ1_0 file (430 MB, Apache 2.0) from Hugging Face:

```sh
mkdir -p ~/.kindi && cd ~/.kindi
wget https://huggingface.co/prism-ml/Ternary-Bonsai-2-2B-gguf/resolve/main/Ternary-Bonsai-2-2B-PTQ1_0.gguf
```

Convert it once into Kindi's weight pack (about a minute; exact - the same weights, re-laid out):

```sh
# Docker
docker run --rm -u $(id -u):$(id -g) -v ~/.kindi:/data kindi:0.4.0 convert /data/Ternary-Bonsai-2-2B-PTQ1_0.gguf
# native
python3 -m kindi convert ~/.kindi/Ternary-Bonsai-2-2B-PTQ1_0.gguf          # writes ~/.kindi/pack
```

Check the engine against llama.cpp's reference output:

```sh
docker run --rm --gpus all -u $(id -u):$(id -g) -v ~/.kindi:/data kindi:0.4.0 verify
```

```
prompt 1 'The sky is blue because'          PASS  40/40 tokens match
prompt 2 'The capital of France is'         PASS  40/40 tokens match
prompt 3 'Water boils at a lower temper...' PASS  40/40 tokens match
prompt 4 'def add_two_numbers('             PASS  40/40 tokens match
ALL PASS: token-exact with llama.cpp - 22.1 tokens/s over the run
```

The speed in the last line includes the prompt tokens, so it is not the generation speed. It also depends on the
fast-head file of the next step. Without this file, `verify` uses a plain layout of the exact head and shows about 20
tokens/s. With the file, `verify` shows about 22 tokens/s. In both cases, the output is token-exact. `kindi bench`
measures the generation speed (see [Performance](#performance)).

Optional - build the cluster file for the fast output head (about 2 minutes):

```sh
docker run --rm --gpus all -u $(id -u):$(id -g) -v ~/.kindi:/data kindi:0.4.0 fast-head
```

## Run the server

```sh
docker run -d --name kindi --restart unless-stopped --gpus all -u $(id -u):$(id -g) \
    -p 127.0.0.1:8093:8093 -v ~/.kindi:/data kindi:0.4.0
# native:  python3 -m kindi serve
```

The first start compiles Kindi's CUDA kernels for your card (about 1.5 minutes on the K420). Kindi caches them under
`state_dir`, and later starts take about 10 seconds. Then:

```sh
curl -s localhost:8093/health
```

The server listens on `127.0.0.1:8093` (natively) or on the port you publish (Docker). To serve your network, run the
installer with `--lan` (it sets an API key; see [docs/INSTALL.md](docs/INSTALL.md)), or publish `-p 8093:8093`
(Docker) or set `host = 0.0.0.0`, and set an `api_key`.

## Use it

The endpoint is `http://localhost:8093/v1`, the model id `bonsai2-2b` (configurable).

```sh
curl -s localhost:8093/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "bonsai2-2b",
  "messages": [{"role": "user", "content": "Write a bash one-liner that counts the .txt files here."}]
}'
```

Python (`pip install openai`):

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8093/v1", api_key="none")   # your api_key if you set one
r = client.chat.completions.create(model="bonsai2-2b",
                                   messages=[{"role": "user", "content": "What is the capital of Italy?"}])
print(r.choices[0].message.content)
```

Kindi supports these request fields: `messages` (system / user / assistant / tool, text content), `tools` and
`tool_choice` (`"none"` disables tools), and `stream` and `stream_options.include_usage`. It also supports
`temperature` (0 = greedy), `top_p`, `top_k`, `min_p`, `repeat_penalty` and `repeat_last_n` (as in llama.cpp; see
[docs/CONFIGURATION.md](docs/CONFIGURATION.md)), `max_tokens` / `max_completion_tokens`, `stop` and `seed`. It also
supports `chat_template_kwargs.enable_thinking` (off by default). When thinking is on, the thinking comes back as
`reasoning_content` (also streamed as `delta.reasoning_content`). `thinking_budget_tokens` caps the thinking. At the
budget, the server closes the thinking and the model then gives its answer.
Responses carry OpenAI `usage` (with `prompt_tokens_details.cached_tokens`) and a `timing` block.

**Tool calls** come back as OpenAI `tool_calls` with `finish_reason: "tool_calls"`; send the results back as
`role: "tool"` messages.

**Agents and chat UIs.** Point any OpenAI-compatible client at `http://localhost:8093/v1` with model `bonsai2-2b`.
Two settings matter on the smaller cards - see [docs/AGENTS.md](docs/AGENTS.md):
- Give the client a **long request timeout**. An agent's first prompt of 5,000+ tokens takes a few minutes to read the
  first time. The cache makes later turns fast. Do not use automatic retries, because they queue behind the current
  request.
- Tell the client the context window that `/health` shows (on the K420 1 GB: **24,576** tokens).

`kindi chat` is a terminal chat that needs no server.

## Exact and fast output heads

The output layer scores all 248,320 vocabulary entries for every generated token - about 9 ms of a 46 ms step.

- **exact** (default): the full head. Greedy output is token-for-token identical to llama.cpp.
- **fast**: a shortlist head. `kindi fast-head` clusters the vocabulary once. For each token, Kindi scores the 1,024
  cluster centres. Then it computes exact scores only for the 256 clusters with the best scores (~13 % of the
  vocabulary). Generation is about 15 % faster. On 310 measured steps, it chose the same token as the full head
  99.4 % of the time.

To change the head while the server runs, with no restart, use `kindi head fast` / `kindi head exact` (Docker:
`docker exec kindi python3 -m kindi head fast`). `/health` shows the current mode.

## The prompt cache

Bonsai mixes attention with recurrent Gated-DeltaNet layers, whose state cannot be rewound. Kindi caches anyway,
exactly:

- **checkpoints** of the recurrent state every 256 tokens. A request that shares a start with the previous request
  resumes at the last checkpoint before the first difference.
- **parked conversations**: Kindi keeps up to 3 other conversations in host RAM. It resumes each conversation when its
  next turn comes. (Agents alternate between their main prompt and side prompts.)
- **saved openings** on disk. When two requests share an opening of 512+ tokens and then differ, Kindi saves the
  state of that opening under `state_dir`. Typically, this opening is a system prompt with tool definitions. Later
  sessions load it in a fraction of a second, also after restarts.

With the Bonsai models, resumed output is identical to the output when Kindi reads the prompt from scratch. With
Nemotron, the cache reads the new part in other batch sizes, so a reply can take the other token on a near tie (283
of 295 agent turns were the same). The details are in [docs/PROMPT-CACHE.md](docs/PROMPT-CACHE.md).

## Commands

| | |
|---|---|
| `kindi convert MODEL.gguf [--out DIR] [--force]` | build the weight pack (default: the configured `pack`) |
| `kindi fast-head [--clusters N]` | build the fast-head cluster file (stop the server first: needs ~300 MB of GPU memory) |
| `kindi verify` | check greedy output against llama.cpp's reference |
| `kindi bench` | speed at several context positions, prompt read speed, GPU memory |
| `kindi serve [--host H] [--port P] [--api-key K] [--head exact\|fast]` | the OpenAI-compatible server |
| `kindi chat [--system TEXT]` | terminal chat |
| `kindi head exact\|fast` | set the output head while the server runs |
| `kindi info` | configuration, pack and GPU |

Global options: `--config FILE`, `--pack DIR`, `--state-dir DIR`, `--ctx N`.

## Configuration

Defaults, then a config file (`--config`, `$KINDI_CONFIG`, `./kindi.conf` or `~/.kindi/kindi.conf`), then
environment variables `KINDI_<KEY>`, then command-line flags. Copy [kindi.conf.example](kindi.conf.example) and edit
it. The file [docs/CONFIGURATION.md](docs/CONFIGURATION.md) describes the keys: paths, server address, API key, context length,
head mode, cache sizes and generation defaults.

## Performance

Quadro K420, Ternary-Bonsai-2-2B, `kindi bench` (dev build de83d2f, blocking sync off, the installer's setting):

| | exact head | fast head |
|---|---|---|
| generation, short context | 20.9 tokens/s | 24.1 tokens/s |
| generation at 2,000 tokens of context | 20.0 | **22.9** |
| generation at 8,000 tokens of context | 18.5 | 20.9 |
| generation at 16,000 | 16.7 | 18.6 |
| generation at 24,000 | 15.2 | 16.8 |

Prompt reading on the same card (Kindi 0.2.0, exact head): 33.3 tokens/s at 512 tokens, 24.2 at 24,000.

The other cards, at a short context, exact head:

| Card | Model | Generation | Prompt reading (512 tokens) |
|---|---|---|---|
| Quadro K1200 | 2B / 4B / 9B | 81.0 / 39.2 / 23.4 tokens/s | 194.5 / 77.4 / 41.7 tokens/s |
| 2x Quadro K1200 | 27B | 12.6 | 19.5 |
| Quadro K6000 | 27B | 20.4 | 36.2 |
| Tesla P100 16 GB (SXM2, 300 W) | 27B | 36.0 | 167.6 |
| GeForce GTX 1080 | 27B | 29.4 | 256.7 |
| GeForce RTX 3080 (190 W) | 27B | 63.2 (99.9 with the 2B draft model, on agent prompts) | 1,093 |
| AMD Radeon 780M | 27B | 10.6 | 103.7 |

llama.cpp on the same card, with the same model file and the same context: on the K420, it generates 1.65
tokens/s and reads prompts at 1.2 tokens/s (512 tokens). The tables for each card, with their method, are in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md).
How Kindi works: [docs/HOW-IT-WORKS.md](docs/HOW-IT-WORKS.md).
The settings for each tested card (K420, K1200, K6000, Tesla P100, GTX 1080, RTX 3080, RTX 20/30/40/50, Radeon 780M, CPU): [docs/CARDS.md](docs/CARDS.md).

## Agent benchmark

[kbench](https://github.com/ahtx/kindi-bench) has 65 agent tasks with hidden tests and a runner. It also has
the results of our runs with Kindi on the K420. kbench works with any OpenAI-compatible server.

## AH-Nanbeige4.3-33B IQ3_XXS

Kindi has a backend for AH-Nanbeige4.3-33B IQ3_XXS (our IQ3_XXS quantization of Nanbeige4.3-33B, made with bartowski's imatrix), 13.1 GB. This model
is a separate download from Hugging Face. It is not a part of Kindi. Model card: https://huggingface.co/ahtx/AH-Nanbeige4.3-33B-IQ3_XXS-GGUF

It is a strong model for coding work, and it can do even better on a slightly faster GPU.

- Cards: one card with 16 GB or more (for example an RTX 3090), or several cards with about 17 GB in total (for
  example two RTX 3080 10 GB cards or two Tesla P100 cards). The engine puts the layers on the cards in order. On a
  Quadro K6000 (12 GB), it keeps 13 layers in host memory and computes part of them on the CPU.
- Get the model: `kindi use nanbeige-33b`, or `install.sh --model nanbeige`. Both check the sha256 of the file after
  the download.
- `kindi verify` compares the output with llama.cpp. A prompt can end at a llama.cpp tie; that is not a failure.
- Prompt lookup is on by default. When the model copies or edits text, it writes about 2 times as fast.

Speed, same cards and same file as llama.cpp. On two RTX 3080 cards (190 W, PCIe x1), Kindi writes 45.5, 44.3, 40.7
and 37.0 tokens/s at 0, 2K, 8K and 16K of context. llama.cpp writes 32.3, 35.3, 33.3 and 31.0. On two Tesla P100
cards, Kindi writes 21.0 tokens/s at 512 tokens and 19.6 at 8K; llama.cpp writes 9.6 and 8.9 (about 2.2 times). On
one Quadro K6000, Kindi writes 3.56 tokens/s at 8K of context, and llama.cpp writes 1.52 (2.3 times). The tables are in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md).

Agent test: in an initial test of 115 tasks (the 65 tasks of kbench and 50 LiveCodeBench problems, one run of each
task), the 33B solved 98 and the 27B solved 96.

Both models run on Kindi's OpenAI-compatible server, so an agent tool such as opencode can use both. For example,
use the 27B for fast steps and the 33B for harder coding steps.

## Limits

- **One request at a time.** Concurrent requests queue (the card holds one engine).
- **On the K420, Kindi reads prompts at about 25-33 tokens/s**, so a long new prompt takes time. A new prompt of
  5,000 tokens takes about 3 minutes there. The cache makes repeated prompts fast. Kindi reads text that is not in
  the cache (for example, a large file or long command output) at this speed.
- **The Ternary-Bonsai-2 models** (the PTQ1_0 GGUFs: 2B, 4B, 9B, 27B), Nemotron 3.5 Bolt E4B and AH-Nanbeige4.3-33B
  IQ3_XXS only. The 27B needs 8 GB of card memory (GTX
  1080, RTX 3080) or two 4 GB cards (2x K1200).
- **At most two GPUs per model.** Two cards run one model in tensor parallel. More cards run more servers.
- **Small models make mistakes.** The 2B is good at short code, shell and factual questions, tool calls and
  summaries. Check what it writes.

## Troubleshooting

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md). The usual causes are:

- Another program uses the card (out of memory at start).
- On Kepler, the driver is not the 470 branch.
- A client timed out while it waited for a long first prompt.
- On rootless Docker, the commands must not have `-u $(id -u):$(id -g)` (see [docs/DOCKER.md](docs/DOCKER.md)).

## About

[Amir Husain](https://www.amirhusain.com). The [Deep Tech with Amir](https://www.youtube.com/@DeepTechWithAmir)
series on VectorFall covers this work. The [Kindi launch video](https://youtu.be/ddAnghj8HpE) shows Kindi on the K420. More broadly, it also shows how to get the most from inexpensive GPUs.

## License

**AGPL-3.0; commercial licenses available.**

Kindi is free software under the [GNU Affero General Public License v3.0](LICENSE). If you distribute Kindi or a
changed version, the AGPL says that you must give the source code to the recipients. This is also true when users
use your version over a network. For use in a closed product or a hosted service, a paid commercial license is
available. See [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md).

The model is Prism ML's Ternary-Bonsai-2-2B under Apache 2.0. You download it from its publisher. The third-party
notices are in [NOTICE](NOTICE).

## Acknowledgements

Kindi exists because of the models it runs. Thank you to the teams that release their models openly:

- [DeepSeek](https://huggingface.co/deepseek-ai) for
  [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) under the MIT License.
  Kindi's DSpark drafter uses the model's own MTP stages. Thank you also to
  [unsloth](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) for the GGUF conversion.
- [Prism ML](https://huggingface.co/prism-ml) for the Ternary Bonsai models, 2B to 27B, and the
  Qwen team at Alibaba Cloud for the Qwen base models of Ternary Bonsai.
- NVIDIA for Nemotron 3.5 Lightning 30B-A3B, Nota AI for its ternary version, and
  [boro64](https://huggingface.co/boro64/Ternary-Nemotron-3.5-Lightning-30B-A3B-GGUF) for the GGUF conversion.
- [Tencent](https://huggingface.co/tencent/Qwen3.8-27B-DSpine-b16) for the DSpine drafter of the 27B, under
  Apache 2.0.
- The Nanbeige team for Nanbeige4.3-33B, the base model of AH-Nanbeige4.3-33B.

Thank you to slavalevski-tech, who first ran Nemotron 3.5 Lightning on Kindi, on a Quadro K6000. Thank you also to
the [llama.cpp](https://github.com/ggml-org/llama.cpp) authors. Their implementation is the reference that we check
Kindi's output against.

## Contributions

See [CONTRIBUTING.md](CONTRIBUTING.md) for bug reports, submitted work and the
[Contributor License Agreement](CLA.md). AI-assisted work is welcome on the same terms.
