# Changelog

## Unreleased

### New

- **Two-tier Docker images.** Each image is now split into a dependency base image (`kindi-base:0.3.0`,
  `kindi-base-rocm:0.3.0`) and a code image (`kindi:0.3.0`, `kindi-rocm:0.3.0`) that layers `COPY kindi` on top.
  `docker/build-base.sh` and `docker/build-code.sh` rebuild each tier, so a code change rebuilds in seconds instead
  of 15-45 minutes. A `.dockerignore` keeps the converted models out of the build context.
  [docs/DOCKER.md](docs/DOCKER.md)

## 0.3.0 (2026-10-28)

### New

- **AMD Radeon 780M graphics.** A ROCm image (`docker/Dockerfile.rocm`) runs the 2B, the 27B and Nemotron on the
  780M. The installer finds the card and builds the image. [docs/AMD.md](docs/AMD.md) gives the settings.
- **Quadro K6000.** The installer runs the 27B on this 12 GB Kepler card. [docs/CARDS.md](docs/CARDS.md) gives the
  settings.
- **NVIDIA Nemotron 3.5 Bolt E4B (experimental).** Kindi has its own decoder for this model, with tool calls and
  thinking. It runs on one card, or on two cards of the same kind in tensor parallel. Use `--model nemotron` with
  the installer.
- **More prompt cache.** Nemotron has its own prompt cache (`KINDI_G4_PROMPT_CACHE`). With the Bonsai models, the
  server can also resume a request inside a parked conversation (`KINDI_PARK_BRANCH`, on by default on ROCm).
  `KINDI_PROMPT_CACHE=0` turns the Bonsai cache off. [docs/PROMPT-CACHE.md](docs/PROMPT-CACHE.md) gives the
  results, and `examples/prompt_cache_bench.py` is the test.
- **A setup menu.** The installer shows each setup that the machine supports, the recommended setup first.
  `--auto` takes the recommended setup. `--config N,M` runs several setups side by side, and `--config NxK` runs
  one setup on K cards of the same kind. Each server gets its own container and port.
- **CPU servers next to GPU servers.** The installer offers a 2B server on the CPU on each machine.
- **gpu-unlock.** The installer finds the GeForce RTX 30 cards that NVIDIA limits to half-rate 16-bit arithmetic.
  It offers to unlock them with the [gpu-unlock](https://github.com/Zagitta/gpu-unlock) module by Zagitta. It never
  does this without a question.
- **`kindi models` and `kindi use`.** A model catalog, the models that fit this machine, and a resumable download
  with a sha256 check.

### Changed

- Sampling uses the order of llama.cpp by default: top-k, top-p, then temperature.
- The server returns OpenAI `logprobs` and `top_logprobs` for Nemotron.
- With the 2B draft model, a guard stops speculation when it is slower than plain steps (`KINDI_SPEC_GUARD=0` turns
  the guard off).
- The installer detects a server that restarts in a loop, and it checks that the Kindi modules load after the build.
- The installer makes one pi provider for each server.
- On a 2 GB Kepler card, the installer asks for a context of 23,808 tokens for the 4B. This is the context that fits.

### Fixed

- `model_name = auto` also reads the model size from the GGUF file name. Before, a 27B pack could report
  `bonsai2-2b`.
- `/health` shows a git commit only for a Kindi checkout. Before, a copy of Kindi in another git folder showed the
  commit of that folder.
- The installer works on machines with one GPU and with Kepler workstation cards.
- The cache brings a parked conversation back in place, and the server continues after an engine error.
- The tensor-core path starts on cards after the RTX 30 series (RTX 40, RTX 50 and Hopper). Before, the 27B
  with the installer's tensor-core profile and the Nemotron model stopped at start-up with `CUDA_ERROR_INVALID_PTX`.
- Kindi starts on RTX 50 cards. Before, no kernel compiled on these cards, because CuPy compared the card versions
  as text.

## 0.2.0 (2026-10-22)

### New

- **More models and cards.** Kindi runs Ternary-Bonsai-2-2B, -4B, -9B and -27B (the PTQ1_0 GGUFs). Tested on the
  Quadro K420 (1 GB and 2 GB), the Quadro K1200 (one card, or two in tensor parallel), the GeForce GTX 1080, the
  GeForce RTX 3080 and the CPU. [docs/CARDS.md](docs/CARDS.md) gives the settings for each card.
- **A one-command installer** (`install.sh`, [docs/INSTALL.md](docs/INSTALL.md)). It finds the cards, selects the
  model and the settings, downloads the model and checks its sha256, converts it, runs `kindi verify` and starts the
  server as a Docker service. It works with the NVIDIA Container Toolkit or without it. `--dry-run` shows each
  command. A re-run upgrades or repairs, and `--uninstall [--purge]` removes Kindi.
- **The 27B.** A plan engine (`kindi/tp.py`) runs it on one card (GTX 1080 8 GB, RTX 3080) or on two cards (2x
  Quadro K1200, tensor parallel). The K/V cache is q8_0, as llama.cpp `-ctk q8_0 -ctv q8_0`.
- **Prompts in chunks.** The engine reads up to 256 prompt tokens in one pass. The tokens are the same as with one
  token at a time. `KINDI_CHUNK=0` turns it off.
- **5-trit weights** on Pascal and later (1.75 bits per weight). The GTX 1080 holds the 27B with a context of 65,536
  tokens.
- **RTX 3080 (sm_80 and later):** int8 tensor-core products and CUDA graphs (`KINDI_TC=1`), split flash-decode
  attention (`KINDI_FD=1`) and tensor-core attention for the prompt chunks (`KINDI_ATT_TC=1`). `KINDI_SPEC=draft2b`
  uses the 2B as a draft model; the greedy output is the same as without it.
- **A CPU engine** (`KINDI_DEVICE=cpu`) for the 2B, 4B and 9B. It is token-exact with llama.cpp and needs no CUDA.
- **Checkpoints in card memory** (`KINDI_CKPT_VRAM=1`, one card): the copy to the host waits until the server is
  idle.
- **`/health` shows the version and the source commit.** `kindi verify` uses the reference of the pack's model.
- **[docs/BENCHMARKS.md](docs/BENCHMARKS.md)** compares Kindi and llama.cpp on the same card, with the same model
  file and the same context.

### Changed

- The context length fits the free card memory (`KINDI_MEM_RESERVE_MB`).
- The prompt cache: the memory budget counts every checkpoint and every parked conversation, and there is a
  checkpoint at the end of the opening. Saved openings belong to one engine configuration, and the server does not
  resume an opening of another configuration. A damaged opening file does not stop a request.
- Blocking sync: while it waits for the card, the host thread sleeps and does not keep a CPU core busy. It is on by
  default on CUDA (`KINDI_BLOCKING_SYNC=0` turns it off). It costs 1-3% of the decode speed, so the installer turns
  it off for one server on its own card.
- Before the 27B loads on two cards, the engine checks the free host RAM and stops with a clear message when it is
  too small (`KINDI_HOST_RAM_CHECK=0` skips the check).
- `model_name = auto` (the default) takes the model id from the pack.
- The K420 comparison in the documentation now uses the same model file for llama.cpp (PTQ1_0). The 0.1.0 page used
  llama.cpp's Q2_0 file. With the PTQ1_0 file, llama.cpp generates 1.65 tokens/s on the K420; Kindi generates 20.8.

### Fixed

- The fast head always scores the special tokens (`</think>` and the end tokens). Before, it could fail to close
  the thinking.
- Host memory: the server frees the CuPy pinned-memory pool after the load, and freed checkpoints go back to the
  system. Before, a server grew with each request.

## 0.1.0 - first public release

- License: GNU AGPL-3.0. Commercial licenses are available (see COMMERCIAL-LICENSE.md).
- An engine for Ternary-Bonsai-2-2B on the NVIDIA Quadro K420. It generates 20.8 tokens/s with the exact head and
  23 tokens/s with the fast head. It reads prompts at 26.5 tokens/s. The context is 24,576 tokens in 1 GB.
- With the exact head, the output is token-for-token identical to llama.cpp (`kindi verify`).
- `kindi convert` makes an exact conversion from the PTQ1_0 GGUF.
- An OpenAI-compatible server with chat completions, streams, tool calls, an optional API key and live progress in
  `/health`.
- Thinking mode: the server sends the thinking as `reasoning_content` (in a stream, as `delta.reasoning_content`).
  `thinking_budget_tokens` sets a limit. At the limit, the server closes the thinking and the model gives its answer.
- When `max_tokens` stops a reply in the middle of a tool call, the server returns that text. The client can then
  see that the reply is not complete.
- A prompt cache: checkpoints of the recurrent state, parked conversations in RAM and saved openings on disk.
- An optional fast (shortlist) output head. You can change the head at run time (`kindi head`).
- A Docker image with the CUDA 10.2 / CuPy 9.6 stack that Kepler needs.
- When a client disconnects (timeout, Ctrl-C), the server stops the request at the next token. This is also true
  during a long prompt read.
