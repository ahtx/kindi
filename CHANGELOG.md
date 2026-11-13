# Changelog

## 0.6.0 (2026-11-12)

### New

- **DeepSeek V4.1 Flash is a supported model** ([docs/DEEPSEEK.md](docs/DEEPSEEK.md)). It uses one CUDA card with
  12 GB or more and host RAM for the experts. A second card can hold hot experts. It drafts with DSpark and saves its
  prompt cache.
  - On one RTX 3060 12 GB, a Quadro M4000 as the expert card and two Xeon E5-2673 v4 CPUs, Kindi writes 18.7
    tokens/s with DSpark (the default). It reads a real-text prompt at about 95 to 100 tokens/s.
  - `tools/install_dsv41.sh` checks the host and picks a profile. The big-RAM profile is measured. The NVMe profile is
    experimental, and we did not measure it yet.
  - The script downloads the files and checks their sha256. It writes a systemd service. It does not replace an
    existing service without `--force`.
  - `python -m kindi.dsv41 verify` checks 20 prompts of 64 tokens against the reference. The gate is 98.5% top-1
    agreement. Each miss must be a near-tie, with a top-2 gap of 0.75 or less.
  - The default numerics are DeepSeek's own FP8/FP4 rounding. `--flags 160` uses the rounding of the reference.
- **Nemotron 3.5 Lightning 30B-A3B (ternary).** Kindi has a backend for Nota AI's ternary version of NVIDIA
  Nemotron 3.5 Lightning 30B-A3B (GGUF by boro64, Q2_0). Use `kindi use lightning-30b`. The model has Mamba-2,
  mixture-of-experts and attention layers. On an RTX 3080 at 190 W, Kindi writes 310 tokens/s at 512 tokens and
  277 at 32K (llama.cpp 202 and 182). It reads a 32K prompt at 3,777 tokens/s (llama.cpp 2,138). On an RTX 3090
  at 220 W, it writes 341 tokens/s at 512 tokens (llama.cpp 284). On a GTX 1080 with 8 GB, 95 of the 128 experts
  of each layer are on the card, and Kindi writes 75 tokens/s after a text prompt of 512 tokens (llama.cpp, with the experts of 9 layers on
  the CPU: 4.7). On a Quadro K6000, it writes 61.9
  tokens/s and reads 206 tokens/s at 512 tokens (llama.cpp 35.1 and 165). The server supports chat, streaming, the thinking block and tool calls.
  [docs/CARDS.md](docs/CARDS.md) and [docs/BENCHMARKS.md](docs/BENCHMARKS.md) give the numbers. The weights are
  not part of Kindi; their licence is OpenMDW 1.1.
- **Host-memory experts.** On cards with less than 10 GB, the least used experts of the Lightning model stay in
  host memory. The card reads them over PCIe. During generation, each layer keeps its last used host experts in 10
  slots on the card (`KINDI_LT_DCACHE`), so an expert that the next token uses again is
  not read again. On an RTX 2070 SUPER (8 GB, PCIe 3.0 x16), Kindi writes 116.4 tokens/s after a 512-token text
  prompt and 120.3 after 8K (llama.cpp on the same card, with the experts of 12 layers on the CPU: 113.5 and
  111.8), and reads 2,121 tokens/s (llama.cpp 1,137).
- **Nemotron 3.5 Lightning is a supported model.** The installer offers it (`--model lightning`) on NVIDIA cards
  with 6 GB or more from the GTX 10 series on, on the Tesla P100 and on the 12 GB Kepler cards, after the 27B. The
  default context is 32K.
- **One mixture-of-experts placement for every MoE engine** (`kindi/moe.py`). It decides which experts of which
  layer go to the card, to host memory and (for the DeepSeek engine on hosts with little RAM) to the disk, from
  measured pick counts (`kindi/lightning-experts.json` now has them). With host experts, the card slots go to the
  most used experts over all layers. `KINDI_MOE_WEIGHTED=0` keeps the same count in each layer.
- **The DSpine drafter for the 27B** ([docs/DSPINE.md](docs/DSPINE.md)). It is a block drafter that reads five
  layers of the 27B. Its weights are Apache-2.0. The output is the same as plain decoding.
  - RTX 3080 at 190 W, context 32768: DSpine writes 115.5 tokens/s and accepts 3.10 tokens per pass.
  - The installer downloads DSpine for the 27B on tensor-core cards. `KINDI_SPEC=auto` uses it. `--no-dspine` keeps
    the 2B drafter.
- **Exact token counts and the real context.** New endpoints: `/v1/tokenize` and `/v1/detokenize`. With
  `"dry_run": true`, a request gives the exact `prompt_tokens` and does not generate. `/v1/models` gives the context
  limit (`max_model_len`, `meta.n_ctx`). A prompt that does not fit gets an HTTP 400 at once, with the code
  `context_length_exceeded`.
- **Progress during long reads.** A stream gets a `: kindi progress {...}` comment every 10 s. A client that does not
  stream can ask for keep-alive spaces with `kindi_keepalive`.
- **A save after each turn.** The server writes the state to the disk in the background after each answer. A restart
  does not lose the turn. DeepSeek in thinking mode saves the state before `<think>`.
- **Conversations by session.** Send the `X-Kindi-Session` header, or `kindi_session` or `user`. When the cache is
  full, older states of a session go before the newest states of other sessions. The DeepSeek server keeps 2 to 4
  conversations in RAM with `--slots N`.
- **Priority.** A request is chat (the default) or background (`kindi_priority`, `X-Kindi-Priority`). Chat goes
  first. A background read stops between chunks for a chat request. Then it continues from the cache, and the answer
  does not change.
- **DeepSeek sampling.** The server uses temperature, top_p, top_k and min_p. DSpark drafts for sampled requests too,
  and the output has the distribution of plain sampling. A repeat guard is on by default.
- **`/health` reports drafting:** the method, the drafter, and the drafted and accepted tokens of the last request.
- **`install.sh --lan`** serves the local network. It makes an API key in `~/.kindi/api-key` and gives the key to the
  container in an env file. `--api-key K` sets the key. `--lan` is off by default.
- **Kindi Bridge in the installer.** It is optional: `--with-bridge`, `--no-bridge` and `--bridge-source D`, and
  `-WithBridge` on Windows. Bridge 0.6.0 shows the Kindi servers and their features. Its menus are Agents, Schedule,
  Kindi, Files, System, View and Help. The manual has a Kindi Bridge chapter. Bridge is AGPL-3.0, as Kindi is.
- **AH-Nanbeige4.3-33B with an 8-bit K/V cache.** The new default is `kv_cache auto`. It uses f16 when the model and
  the f16 cache fit at the chosen context. Otherwise it uses q8_0. Both caches pass `kindi verify`. The q8_0 check
  counts the tokens up to the first difference from the f16 reference. Kindi keeps 475 of 586 tokens.
- **Kindi on Windows 10 and 11 with one PowerShell command** (`install.ps1`, [docs/WINDOWS.md](docs/WINDOWS.md)).
  - The installer checks the PC: x86-64, the Windows build, virtualization, RAM, the GPU driver and the proxy.
  - It turns on WSL2 if it is off. After the restart, it continues by itself.
  - It makes a WSL2 distribution only for Kindi (Ubuntu 24.04 from Canonical, sha256-checked). It does not change your
    other distributions.
  - It uses Docker Desktop if you have it. If not, it installs Docker Engine and the NVIDIA container toolkit in the
    Kindi distribution.
  - It checks that the GPU reaches WSL2, and it runs `install.sh` inside. Then it checks the server on
    `http://localhost:8093` from Windows.
  - It adds the Kindi Console (status, chat, logs, API settings), a shell item, an uninstall item and a Settings > Apps
    entry.
  - AMD GPUs use the CPU engine on Windows, because WSL2 has no `/dev/kfd`. A native Windows build is not in 0.6.0.
- **`install.sh` under WSL2.** It finds `nvidia-smi` in `/usr/lib/wsl/lib`. It says that the RAM is the RAM of the
  WSL virtual machine. It does not make kernel changes (gpu-unlock, the 780M option). It uses the GPU support of
  Docker Desktop.
- **The model on the first start of a Kindi container.** Set `KINDI_MODEL` to a model id or to a GGUF file in the volume. The container
  downloads the model and checks its sha256. It builds the pack once into `/data/packs/<name>` and uses it again at
  the next starts. Two containers on the same volume build it once. `KINDI_VERIFY_ON_START=1` runs `kindi verify`
  once for each pack and kind of card.
### Changed

- DeepSeek: the card does its part of each layer in fewer steps, and some kernels run at the same time. The card
  time for each token goes from 34.6 to 24.1 ms. The output does not change, bit for bit.
- DeepSeek: DSpark calculates the cost of a draft from the experts that all the drafted tokens use together. This
  is on by default. The output does not change.
- The images run as the user 1000:1000, not as root. With rootless Docker, the installer now uses `-u 0:0`.
- The CUDA image does not have the CUDA static libraries, NPP and nvJPEG. Kindi does not use them.
- Drafting on the tensor-core path uses `KINDI_FD=1` when `KINDI_FD` is not set. With `0`, a drafted answer could be
  different from plain decoding at near-ties. 0.5.0 had this too. The profiles of the installer already set `1`.
- The installer names more cards correctly with the 470 driver. The Maxwell Quadro M cards are compute 5.2 (it
  printed sm_50). The Tesla P100 is sm_60 (it guessed 6.1). The A100 is sm_80 (it matched a Maxwell pattern).

### Fixed

- DeepSeek: when `max_tokens` or the context stopped a reply in its tool calls, the server sent the raw tool-call
  markup as text. Now the reply has `finish_reason` "length", the complete calls, and the cut call with the start
  of its arguments, as in the OpenAI API. A request without `max_tokens` can use the remaining context (the limit
  was 4096 tokens).
- Lightning on 8 GB cards with host experts stopped with "out of memory" at the decode slots (an RTX 2070 SUPER):
  the plan's reserve of 350 MB did not cover what the plan does not count. The engine now frees everything, keeps
  free what the failed allocation was short of (plus 96 MB), and plans again (up to 3 times). `kindi verify` passes
  on the RTX 2070 SUPER.
- Tool calls of the Lightning server: one call per reply unless the request asks for `parallel_tool_calls` (the
  model could write the same call 20 times, 3,000+ tokens), a repeated call ends a parallel reply, a call cut by a
  stop token is closed and parsed (a call without a required parameter is still dropped), the slips
  `<parameter=name=value>` and `</function_call>` are read, a `<tool_call>` quoted in backticks is text, and a call
  of a tool that the request did not offer (the chat template's example call) is not returned. 57 of 60 test
  requests come back as correct calls (`tools/lt_tooltest.py`, 5 rounds), without the long repeated-call replies.
- Sampled speculation used the temperature before top-p, and it did not use min_p. So the server turned drafting
  off for min_p requests. Now plain and speculative sampling use one distribution (#32).
- On native Windows, the host RAM read 8 GiB, so the checkpoint budget was too small (#4).
- A parked conversation was removed at once when the cache was full (#26).
- `install.sh --dry-run --yes` waited for a sudo password on a computer without Docker.
- The sampled DeepSeek path gave HTTP 500.

### Thanks

- The Nemotron 3.5 Lightning backend was started by the work of slavalevski-tech
  ([github.com/slavalevski-tech/kindi](https://github.com/slavalevski-tech/kindi)). slavalevski-tech first ran this model on Kindi
  and shared that backend under the Kindi CLA 1.1. That work set the scope of the backend. The Kindi code is a new
  implementation ([docs/dev/LIGHTNING-CLEANROOM.md](docs/dev/LIGHTNING-CLEANROOM.md)). Thank you.

### Not in 0.6.0

- Gemma 4 31B Dense: Kindi was not faster on the same cards (two RTX 3080s).
- Qwen3.8-Flash-Next on many cards: moved to 0.7.0.
- The RTX 20 speed work: moved to 0.8.0.

### Container images (added on 2026-11-12)

- Ready-made images are on GitHub's container registry ([docs/CONTAINERS.md](docs/CONTAINERS.md)): `ghcr.io/ahtx/kindi:0.6.0-cuda`, `ghcr.io/ahtx/kindi:0.6.0-rocm6`, `ghcr.io/ahtx/kindi:0.6.0-rocm7`, and the
  moving tags `cuda`, `latest`, `rocm6`, `rocm7`. Each image passed `kindi verify` on a real card before it was published: `0.6.0-cuda` on NVIDIA GeForce RTX 3080; `0.6.0-rocm6` on AMD Radeon Graphics (with HSA_OVERRIDE_GFX_VERSION=11.0.2); `0.6.0-rocm7` on AMD Radeon 780M Graphics (RDNA3, run with HSA_OVERRIDE_GFX_VERSION=11.0.2).
- `compose.yaml` and `compose.amd.yaml` start a server with Docker Compose or podman-compose, without a clone or a build.

## 0.5.0 (2026-11-09)

### New

- **The NVIDIA CMP 170HX (GA100, sm_80) is a supported card.** The board that we measured has 40 GB of HBM2e
  (device id 0x2082) on PCIe gen4 x16, and its FP32 FMA rate is not limited. Kindi writes faster than llama.cpp on
  the same card for the 27B (one and two cards) and for AH-Nanbeige4.3-33B IQ3_XXS (one and two cards, up to 32K
  tokens of context). We did not test the CMP 170HX board with 8 GB.
  [docs/CARDS.md](docs/CARDS.md) and [docs/BENCHMARKS.md](docs/BENCHMARKS.md) have the numbers.
- **The b2 weight copy (sm_80).** On sm_80 cards with 20 GB or more, the decode products read a second copy of the
  ternary weights with 2 bits for each digit. Each mma register then needs one logic operation instead of the
  five-trit decode. The products are bit-identical. The copy is 1.29 times the size of the weights. `KINDI_TC_B2=0`
  turns it off.
- **The 4B on the tensor-core path.** Widths that are not a multiple of 1024 (the 4B: 2560) use padded units. On an
  RTX 3080 at 190 W, the 4B writes 258 tokens/s at 512 tokens (the single-card engine: 118).
- **The AMD Radeon 840M (RDNA 3.5, gfx1152) is a supported graphics unit.** It is the graphics unit of the Ryzen AI 5
  340. ROCm runs it natively, without `HSA_OVERRIDE_GFX_VERSION`. On the same computer and with the same model files,
  Kindi writes faster than llama.cpp for the 2B, 4B, 9B and 27B, at 512 and 8K tokens, and it reads the 27B faster.
  [docs/CARDS.md](docs/CARDS.md) and [docs/BENCHMARKS.md](docs/BENCHMARKS.md) have the numbers.
- **AMD Radeon 8060S (Strix Halo, Ryzen AI MAX+ 395, gfx1151).** Kindi runs the 2B, 4B, 9B and 27B on it, and
  `kindi verify` passes on all four. On the same computer and with the same model files, Kindi writes and reads
  faster than llama.cpp for all four models, at 512 and 8K tokens. [docs/BENCHMARKS.md](docs/BENCHMARKS.md) has the
  numbers.
- **The `kindi-rocm7` image for RDNA 3.5.** The installer finds gfx1150, gfx1151 and gfx1152 graphics and builds
  `kindi-rocm7:0.5.0` (`docker/Dockerfile.rocm7`: ROCm 7 with HIP 7.16 and CuPy 14.2.0). These graphics units do not
  need `HSA_OVERRIDE_GFX_VERSION`. The installer offers the 27B, 9B, 4B and 2B on them. We did not measure Strix Point
  (gfx1150). RDNA3 graphics (gfx1100-gfx1103, the Radeon 780M) keep the `kindi-rocm` image (ROCm 6.3.3).

### Changed

- The new decode defaults (128-position flash-decode chunks, 4-warp product blocks and the one-pass RMS fold on
  cards with 80 SMs or more, the thread-per-element SiLU fold, NVRTC's own `expf` in the tensor-core kernels) are on
  for sm_80 and sm_86 only. Other cards keep the 0.4.0 behaviour until we measure them. `KINDI_TESTED_CC` adds a
  compute capability (for example `8.9`).
- The bank-free IQ3_XXS codebook of the 33B decode is on for sm_80 only (on an RTX 3090 it was 14% slower).
- `kindi verify` counts a first difference as a tie when llama.cpp's own top-2 gap there is under 0.01 nats, or when
  another llama.cpp run of the same file gives Kindi's token. This is the rule of the 33B backend. The 4B and 27B
  references have the gaps (`tools/bonsai_gaps.py` makes them from a running llama-server).
- Weights go to the card with a plain synchronous copy (a large pinned staging buffer failed on hosts with little
  free memory).
- On AMD graphics, the GEMV configurations are set for each graphics unit (the Radeon 780M and the Radeon 840M).
  Kindi times the configurations at start on other AMD graphics units. On AMD, a row-parallel product uses 8 column
  splits at most, because 16 splits changed the 27B output on the 840M.
- On the Radeon 8060S (gfx1151), the 2B, 4B and 9B use the plan engine, with GEMV kernels and settings for this
  unit. Kindi waits for the graphics unit in a loop instead of a blocking wait (the blocking wait woke late at
  random). `KINDI_BLOCKING_SYNC=1` keeps the blocking wait.
- The installer finds AMD graphics on computers and containers without `lspci`.
- Kindi copies the weights to the card in pieces, one tensor at a time. The host does not keep a full copy of the
  weights. A computer or container with 2 GB of free memory can load the 9B and the 27B.

### Fixed

- The 9B on the tensor-core path stopped with an illegal memory access (the embedding fold had a 1024-value buffer
  for Hadamard blocks of 4096).
- The 9B on the tensor-core path differed from llama.cpp at prompt 1 token 36 with Kindi's own `expf`; it is
  token-exact with NVRTC's `expf`.
- The 9B on AMD graphics gave wrong output (`kindi verify` FAIL). A barrier in the embedding kernel was inside a
  condition, so on AMD the waves did not wait for each other. The 9B is now token-exact on the Radeon 780M and 840M.
- On AMD graphics, prompts were read one token at a time when the weight cache had no lane-layout copy (Radeon 780M,
  27B: 10.8 tokens/s). Kindi now makes the missing copy (105 tokens/s).

## 0.4.0 (2026-11-04)

### New

- **Speculative decoding by default on tensor-core cards.** On NVIDIA sm_80 and later, the installer sets
  `KINDI_SPEC=auto`. The 27B uses the 2B as its draft model: k=2 below 1K tokens of context, k=3 from 1K. On an
  RTX 3080 at 190 W, the 27B writes 1.46 times as fast at 512 tokens of context and 1.82 times as fast at 8K. The
  output is the same as without speculation. [docs/MTP.md](docs/MTP.md) gives the measurements.
- **An MTP head for the 2B.** `kindi-mtp-2b-v1.npz` drafts tokens for Ternary-Bonsai-2-2B, up to 8K tokens of context
  (about 1.1 times as fast). The installer downloads it from
  [Hugging Face](https://huggingface.co/ahtx/kindi-mtp-heads) and checks its sha256.
- **The Tesla P100 (sm_60) is a supported card.** The 27B writes 36.0 tokens/s on one P100 (SXM2, 300 W) at 512 tokens, against
  18.7 for llama.cpp on the same card. A new fp16 prompt path is on by default for sm_60 (`KINDI_P100_FP16_PROMPT=0`
  turns it off). With it, Kindi reads prompts 12% to 14% faster than llama.cpp. Speculation is off on the P100. The
  33B runs on two P100 cards at about 2.2 times the speed of llama.cpp. [docs/CARDS.md](docs/CARDS.md) has the numbers.
- **The GeForce RTX 20, 30, 40 and 50 series.** All four series are supported. `kindi verify` passes on them. The
  speed numbers are in [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for the cards that we measured.
- **AH-Nanbeige4.3-33B IQ3_XXS.** Kindi has a backend for AH-Nanbeige4.3-33B IQ3_XXS (our IQ3_XXS quantization of Nanbeige4.3-33B, made with bartowski's imatrix). It puts the layers on one or more NVIDIA
  cards (two RTX 3080, two Tesla P100, one RTX 3090), and on a Quadro K6000 with layers in host memory (3.56 tokens/s at 8K, 1.52 for llama.cpp). Use `kindi use nanbeige-33b` or `install.sh --model nanbeige`. On two RTX 3080 cards, it writes 45.5
  tokens/s, and llama.cpp writes 32.3 with the same file. Prompt lookup is on by default (2 times as fast on copy and edit
  work). In an initial test of 115 tasks, the 33B solves 98 and the 27B solves 96. Model card:
  https://huggingface.co/ahtx/AH-Nanbeige4.3-33B-IQ3_XXS-GGUF
- **A start-up self-check for the 33B.** The AH-Nanbeige4.3-33B server checks itself at start-up. If the check
  fails, the server stops, so the service restarts it before it gives a wrong reply.
- **Sampler settings.** Requests can set `min_p`, `repeat_penalty` and `repeat_last_n` (#4).
- **Pinned checkpoints.** `KINDI_CKPT_PINNED` keeps the host copies of the prompt-cache checkpoints in pinned
  memory. It is on by default under WSL2 (#4).

### Changed

- The speculation guard works from 32K tokens of context (`KINDI_SPEC_GUARD_FROM`). Below 32K, speculation was
  faster on each card that we measured.
- The installer works with rootless Docker, and it checks `--gpus` before `kindi verify` (#8).
- The README tells what to do after the install: the port, `/health`, clients, curl and Python (#11).

### Fixed

- The CPU engine restores a cached state in place (#9).
- `kindi bench` times the CPU engine with the host clock when there is no CuPy (#10).
- A streaming request with a prompt that is too long gets HTTP 400 (`context_length_exceeded`) before the first
  byte. Before, the server sent HTTP 200 and then an error in the stream (#18).
- `kindi fast-head` works on a model with more than 4,096 hidden values, for example the 27B (1,280 threads in one
  block was more than the limit of 1,024) (#4).
- `/health` shows the head that runs. A model that has no fast head (the 27B, Nanbeige) shows `exact` also when
  `kindi head fast` was set (#4).
- The installer finds the compute capability of the Tesla K10 and K20 (Kepler) and of the GTX TITAN X (Maxwell,
  sm_52) from the card name. The Quadro K2000 and K4000 are no longer taken for a K20 and a K40 (#19).

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
