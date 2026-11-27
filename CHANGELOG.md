# Changelog

## 0.8.0 (2026-11-27)

The speed settings below are on by default. Each one keeps an environment variable to turn it off.

- Bonsai 2B, 4B and 9B fuse the Gated DeltaNet and attention setup kernels on one card. In 2,048-token tests, write speed rose from 56.09 to 71.91 tokens/s on the GTX 1080, from 79.52 to 82.55 on the Quadro K1200 and from 220.77 to 241.29 on an RTX 3080. The tokens and final logits matched. `KINDI_DENSE_FUSE=0` turns fusion off.
- Qwen3.8-Flash-Next reads long prompts faster on one Ampere or newer card when its experts are page-locked. On an RTX 3060 server, a 14,064-token prompt read at 870 tokens/s, up from 361 with the 0.8 settings before this change. A 4,601-token prompt read at 454 tokens/s greedy and 504 at T=0.7, up from 345. Write speed changed by -5% to +3%. The profile stages experts early, coalesces uploads and sizes the expert batch to the free card memory. In the default fast mode it also uses fp16 prompt GEMMs, the fused prompt kernels and causal key limits. `KINDI_QXG_PROFILE=0` turns the profile off. `KINDI_FAST=0` keeps the exact prompt arithmetic.
- DeepSeek V4.1 Flash uses a partial sort for top-p sampling. On the RTX 3060 and M4000, sampled write speed rose from 12.37 to 15.61 tokens/s at 4,284 prompt tokens and from 12.84 to 16.49 at 13,779. Read speed and the seeded answers were unchanged. `KINDI_DSV_TOPP_PARTIAL=0` uses the full sort.
- Nemotron 3.5 Lightning batches up to 1,024 prompt tokens on cards that otherwise use 512-token chunks. The GTX 1080 read a 2,048-token prompt at 758.19 tokens/s, up from 647.21. Write speed moved from 86.77 to 85.99. The tokens matched. `KINDI_LT_ADAPT_PREFILL=0` keeps 512-token chunks.
- Run Ternary Bonsai 2.5 27B with its PTQ1_0 GGUF. After `python -m kindi.bonsai25_mtp Ternary-Bonsai-2.5-27B-PTQ1_0-mtp.gguf` writes the head into the pack, Ampere and newer cards draft two tokens with it by default. On the RTX 3090, greedy write speed rose from 85-87 to 135-167 tokens/s across two documents at 512 and 4,096 prompt tokens. The tokens matched. `KINDI_SPEC=0` turns drafting off.
- Nanbeige uses an MTP head saved in the pack as `nanbeige-mtp.npz` by default. The output is unchanged. No head is published yet. Set `mtp_head = none` to turn it off.
- Serve and verify Qwen3.8-Flash-Next on the CPU. Prompt chunks of 16 tokens or more use fp32 matrix products and tiled attention, the per-layer embedding table warms in the background, and the thread pool synchronizes per NUMA node. MTP drafting is on for the CPU too and gives the same greedy tokens as plain decoding. On dual Xeon E5-2673 v4 CPUs (20 cores per socket, 40 threads, 8,192 context), 5,870-token prompts read at 82 to 86 tokens/s over HTTP, against 78 to 81 for ik_llama.cpp with the same GGUF file and harness. Write speed is 14.5 to 15.1 tokens/s greedy and 14.4 to 14.8 at T=0.7, against 9.5 to 10.1 and 9.8 to 10.2. `kindi verify` passes with 173 of 174 reference tokens.
- Inkling 2 Mini avoids a host copy when all experts fit on the card. The GTX 1080 test gained 3.9 to 4.7% in greedy write speed against 0.7.2, with the same output.
- The Flash-Next planner accepts Maxwell cards with 8 GB free. The release archive includes the Flash-Next installer and its MTP helpers. The general installer accepts a successful model verification without a speed line.

- Kindi serve answers the Anthropic Messages API at `POST /v1/messages` and `/v1/messages/count_tokens`, on the same
  chat path as `/v1/chat/completions` (text, tools, thinking, streaming, `x-api-key`). Images and documents get HTTP 400.
- Admission limits: `queue_max` (64) waiting requests, then HTTP 429 `server_busy`; `max_body_mb` (32), then HTTP 413;
  an optional per-client `rate_limit_per_min`. `/health` shows them.
- `kindi bench --sample-vram` logs each card's VRAM peak during the run.
- Turing (RTX 20, T4) tensor cores on the `KINDI_TC=1` path with `KINDI_TC_TURING=1`. Off by default.
- The DeepSeek card calibration loads the CUDA driver through the same code as the rest of Kindi.
- Thanks to atimelessrage for #47, #51, #63, #64 and #75. We wrote these changes again in Kindi's code from their
  pull requests. Their #50 and #58 shipped in 0.7.2.

See the [0.8 benchmarks](docs/SPEED.md) for model files, settings and baselines.

## 0.7.2 (2026-11-24)

Everything in 0.7.1 and 0.7.0 stays. This includes Qwen3.8-Flash-Next, AMD RDNA2 support, native Windows, several
requests at once and card placement. Kindi 0.7.2 adds LoRA adapters for the ternary Bonsai models and two fixes from
atimelessrage.

### New

- **LoRA adapters.** `kindi --lora FOLDER serve` loads a PEFT adapter and runs the model with it. The folder holds
  `adapter_config.json` and `adapter_model.safetensors`. The tensor names can be Hugging Face Qwen3.5 names or GGUF
  names. Kindi folds the adapter into the rotated activations once, at load. No extra rotation runs for each token.
  Kindi stores the converted adapter in the pack, and you can name it again by its hash. An adapter on a projection
  adds one small kernel after each adapted product. Kindi merges adapters on `ssm_alpha` and `ssm_beta` into the
  float weights at load. They cost nothing for each token. The embeddings and the output head are not adapted.
  - On a Quadro K1200 with the 4B and a rank-16 adapter, the adapter costs 2.15 ms for each token. The model writes
    35.9 tokens/s. Without an adapter it writes 39.6 tokens/s.
  - The K420 test used a random rank-32 adapter on all 12 module types. Kindi gave the same tokens as the float
    Hugging Face and PEFT reference in 240 of 240 teacher-forced tests. The largest logit difference was 0.003.

### Fixed

- Kindi reads pack and cache JSON files with `kindi.jsonio.load_json`, which closes the file. On Windows, an open file
  could block a `kindi convert` or a cache rewrite that ran at the same time. Thank you to atimelessrage (PR #50).
- Every error from `kindi serve` and `kindi route` has the OpenAI error body (`{"error": {"message", "type", "param",
  "code"}}`), also in streams. The 404, 405 and 422 answers of the web framework use it too. A `messages` value of the
  wrong type gives a 400 with `param` `messages`, in place of an HTTP 500. Thank you to atimelessrage (PR #58).

## 0.7.1 (2026-11-23)

0.7.1 keeps everything in 0.7.0 (Qwen3.8-Flash-Next, AMD RDNA2, native Windows, several requests at once and card
placement) and adds three changes and one model. Flash-Next now drafts for sampled requests. The CPU engines use one
thread for each physical core. A new check finds the warp-shuffle mistakes that hung kernels on RTX 20, 30 and 40
cards. Kindi also runs Inkling 2 Mini.

### New

- **Inkling 2 Mini** (Thinking Machines Lab), experimental. `kindi convert` accepts the IQ2_M GGUF (about 7.5 GiB),
  and `kindi serve` runs it with its chat template, streaming, thinking levels, tool calls, the sampling values from
  the GGUF and prompt reuse. All 3,072 experts fit in VRAM on an 8 GB card. The
  server answers one request at a time for this model: a second request waits for the first. On a GTX 1080 (8 GB,
  full VRAM) it reads a 512-token prompt at 120.9 tokens/s and writes at 19.8 tokens/s with a 2,048-token context.
  On an RTX 3080 10 GB, a context of 32,768 tokens fits.
- **Drafting for sampled requests in Flash-Next.** A request with temperature above 0, top-k, top-p, min-p or a
  repeat penalty now drafts with the MTP head. Each draft is kept with the probability that plain sampling gives
  that token, so the distribution of the output does not change. Greedy requests keep the 0.7.0 stream exactly.
  `KINDI_FN_DRAFT_SAMPLED=0` turns it off.
- **One CPU thread for each physical core.** Where Kindi picks CPU threads itself (the dense CPU engines, the
  Flash-Next expert pool, the DeepSeek expert pools), it takes one logical CPU for each physical core. Environment
  settings still override it. On a Ryzen 9 7940HS, 8 threads decode at 11.5 tokens/s and 16 threads at 10.0 to 10.8.

### Fixed

- Inkling 2 Mini hung on RTX 20, 30 and 40 cards (compute capability 7.0 and later) because two warp shuffles named
  lanes that never reach them. Both shuffles now name the lanes that run.
- `tools/check_warp_masks.py` lists every warp shuffle and barrier under a condition that changes inside a warp and
  fails on any that is not reviewed. The release checks run it. The expert kernels now stop with a clear error when
  a model has a shape they cannot handle, in place of a hang.
- The native Windows package stays at 0.7.0: 0.7.1 has no new Windows build.
- `install.sh` and `kindi cards` stop with a clear message when the NVIDIA driver and library versions differ, and
  `install.sh` names the package when `apt-get` cannot install Docker or the NVIDIA container toolkit. It no longer
  takes an `nvidia` runtime entry in `daemon.json` for an installed toolkit.


## 0.7.0 (2026-11-22)

0.7.0 adds Qwen3.8-Flash-Next, a 125B mixture-of-experts model that runs on one 8 GB card. It also adds AMD RDNA2
support, native Windows with a tray monitor, several requests at once and card placement. Every number is from the
default install. Each comparison uses the same card, model file and context.

### New

- **Qwen3.8-Flash-Next** (about 125B parameters, about 6B used for each token;
  [docs/FLASHNEXT.md](docs/FLASHNEXT.md)). `install.sh --model flashnext` installs ISTA-DASLab's GSQ-RCO IQ3_S file.
  The dense layers, the output head and the most used experts run on the card. The other experts run on the CPU from
  host RAM, and the card's and the CPU's expert kernels give the same bits. Most experts stay in host RAM, so about
  96 GB of RAM or more is recommended. On hosts with less than about 128 GB, set `KINDI_QXG_PLE_LOCK=0`. The server is
  Kindi's OpenAI server with streaming, thinking, tool calls and prompt reuse. `/health` reports the experts on the
  card and the card hit rate. The installer fetches the MTP head from Qwen's release by byte range and converts it on
  your computer (Qwen Community License 1.0; `--no-mtp` leaves it out). `kindi verify` checks the exact path against
  a reference run of the same file (174 of 174 tokens).
  - Speed on one RTX 3060 12 GB with two Xeon E5-2673 v4: it writes 39.4 tokens/s (three runs, new chats). It reads
    489 tokens/s at 4K and 465 at 16K.
  - Three speed features are on by default. Fast mode skips the lowest-ranked CPU expert picks; `KINDI_FAST=0` gives
    exact output, and kbench gives 62 of 65 tasks in both modes. Drafting with the MTP head can differ from plain
    decoding at near-ties below compute capability 8.0; `KINDI_FN_DRAFT=0` gives exact output. The dense layers use
    8-bit integer inputs; `KINDI_QXG_IDENSE=0` gives the float path (kbench: 62 of 65 against 61 of 65).
  - Maxwell and Pascal cards work from compute capability 5.0. Where a card has no `dp4a`, the kernels use exact byte
    products. Kindi loads a cubin for the card, with a PTX fallback. Below compute capability 7.5 the installer picks
    a CUDA 12.x PyTorch build.
  - A Quadro M4000 8 GB (Maxwell) as the only card, with the same CPUs, writes 13.1 tokens/s and reads 4K at 54.7.
    llama.cpp on the same card and file writes 5.4 and reads 4K at 34.2.
- **AMD RDNA2** (gfx1030 to gfx1036; #28). The int8 dot product uses gfx103x's signed dot4, which is exact. The HIP
  kernels compile for gfx1030, and the plan engine runs there. On a Radeon PRO W6800 32 GB, `kindi verify` passes on
  the Ternary Bonsai 2B, 9B and 27B. At 8,192 tokens of context the 27B reads 506 tokens/s and writes 38.8.
  llama.cpp with the same file and q8_0 K/V reads 263 and writes 33.4.
  [docs/BENCHMARKS.md](docs/BENCHMARKS.md) has all the rows.
- **Kindi for Windows.** `kindi.exe` runs natively with its own Python, CuPy and NVRTC. It needs no WSL and no
  `CUDA_PATH`. Kindi Monitor is a tray app with a dashboard window. It shows write and read speed, prompt progress,
  cache hits, context, slots, card memory, utilisation, power, temperature, clocks, host RAM and CPU. It also starts,
  stops and restarts the servers on the PC and selects the model. We tested it on Windows 11 with an RTX 3080, and
  `kindi verify` passed on the 2B and the 27B from the package. `/health` has `slots`, `last` and the card's id.
  Native Kindi loads `nvcuda.dll` itself (#4).
- **More than one request at a time** (`kindi serve --slots N`, config `slots`; #22). The plan engine with the
  tensor-core plans keeps N conversations on one card. Each has its own K/V cache, recurrent state and prompt cache.
  One pass writes one token for every writing conversation, so they share the weight reads. Prompts are read one
  chunk at a time between the passes. Greedy output is the same as when a conversation runs alone (checked token for
  token on the 2B and the 27B). The default stays one request at a time.
  - On an RTX 3080 with `--slots 8`, one pass writes 396 tokens/s in all with the 27B (one slot: 70). It writes 2,852
    with the 2B (one slot: 481).
  - Through the server with `--slots 8`, the 27B finishes 16 requests of 256 tokens at 134 tokens/s in all. The
    default gives 52. One request alone runs as fast as without slots.
  - It works on one card with `KINDI_TC=1` for now ([docs/CONFIGURATION.md](docs/CONFIGURATION.md)).
- **VRAM auto-split on two cards** (`KINDI_TP_SPLIT`, default `auto`). Sometimes the halves of the 27B leave less room
  for the context on one card. Then the other card takes a larger part of the FFN and the output head, so both cards
  keep the same room. On two RTX 3080s with 5 GB taken on one, the context grows from 75,008 to 166,656 tokens
  (tensor cores: 66,816 to 147,968). Cards that fit the context with the halves keep them.
- **Card inventory, placement and one front door** ([docs/INSTALL.md](docs/INSTALL.md)). `kindi cards` lists every
  card with its compute capability, memory, free memory, PCIe link, bandwidth and driver. It says whether Kindi can
  use the card, or why not. `kindi plan` places models on the cards and writes `~/.kindi/plan.json`, so no card is in
  two servers. It uses the fastest card as DeepSeek's main card. It splits a model over two cards by free memory when
  one card is too small. It puts DeepSeek's hot experts on all spare VRAM. `install.sh --place "dsv41:0+" "27b:3"`
  and `--plan FILE` install a placement. With two or more servers, `kindi route` is one OpenAI endpoint. It sends
  each request to the server of its `model`.
- **DeepSeek V4.1 Flash: streamed experts** (`KINDI_DSV_EXPERTS=stream`, the installer's nvme profile). Kindi reads
  only the routed experts of each call from the model file. It uses direct I/O with many reads in flight and a RAM
  expert cache of fixed size. The cache pins the most used experts and prefetches the predicted experts of the next
  two layers. The results are the same bits as the mmap mode. We have not measured speeds on NVMe drives yet
  ([docs/DEEPSEEK.md](docs/DEEPSEEK.md#streamed-experts)).
- **DeepSeek V4.1 Flash: edits far back in a conversation.** The engine copies the window ring to host RAM every
  1,024 positions (about 11 MB each). A request that changes the conversation more than one prompt chunk back
  resumes at the nearest copy. Before, it read the whole prompt again
  ([docs/DEEPSEEK.md](docs/DEEPSEEK.md#edits-far-back-in-a-conversation)). On a 24,386-token conversation with
  message 6 edited, the read takes 228.2 s instead of 256.1 s, and the answers are the same. The first build crashed
  in the ring save, and we fixed it (`tools/dsv41_ring_test.cu`).
- **`tool_choice` and `response_format`** (#38). Kindi now honours them or refuses them with a clear 400 error.
  `required` and a named function force a tool call on Ternary Bonsai. The other engines, `json_object` and
  `json_schema` return the 400. `KINDI_RESPONSE_FORMAT=ignore` gives the old behaviour.

### Fixed

- `install.sh` checks for `tmux` and prints the install commands. Before, a missing `tmux` ended it silently (#41).
- `install.sh --model` is not case sensitive. The CPU engine offers the 4B and the 9B with RAM checks, not only the
  2B (#45).
- Docker no longer runs out of memory with a long context (#40). Kindi counts the container's cgroup RAM limit and the
  parent limits. It does not count clean page cache as used. It skips a saved turn whose K/V copy does not fit in
  host RAM. (PR #48 by atimelessrage, plus our changes.)
- Gemma-4 gives pinned host memory back after the weight upload and when the prompt cache drops copies (#44).
- A tokenizer pack with a missing piece fails at load with a clear error (PR #49, atimelessrage).
- MTP mode A drafts one token by default (PR #72, atimelessrage). Hadamard-4096 packs use the exact tensor-core
  activations by default (the `tc_hp_for` default of PR #76, atimelessrage).
- A container started without `--gpus` ran the CPU engine without a word. It now prints a warning, and `/health` has
  `device`. DOCKER.md has recipes for LAN access (B11), Podman and NAS.
- Kindi Bridge says "For Kindi 0.7.0" and lists DeepSeek, Lightning and DSpark as supported. It takes a server for
  Kindi only when `/health` or `/v1/models` says so (B9).
- The DeepSeek server stops cleanly on SIGTERM and SIGINT. The request in progress finishes, then the exit handlers
  run (B17).
- A new tokenizer test covers Cyrillic, Greek, Arabic, Hebrew, Devanagari, Thai, CJK, Korean and emoji. It found no
  fault (B13).
- Not fixed: B1 (the web UI sends the locale `en_US`). We found no locale code in Kindi or Bridge, and we wait for the
  reporter's request. B12 (pi did not connect) does not reproduce with the `models.json` provider. pi's built-in
  OpenAI provider sends to api.openai.com, as DOCKER.md says.
- A weight pack whose files change after `kindi convert` is now caught (B16). `kindi convert` records each file's size,
  a fast hash and its sha256 in `kindi_pack.json`; every start checks the sizes, `kindi verify` and every build of the
  5-trit cache check the sha256, and a mismatch names the file. A cache made from other weights is made again. On a
  test host a pack's `bcodes.bin` had changed in place, and every new cache made from it gave wrong tokens with no
  message. Packs made before 0.7.0 have no record and are used as before (`kindi convert --force` adds it).
- The 2B on tensor-core cards with its MTP head (the installer's profile for RTX 30 cards and later) stopped at start
  with `cuGraphAddKernelNode failed (1)` since 0.5.0: the head's fold launch missed one argument (B18). It starts and
  passes `kindi verify` again.
- Two cards (27B, tensor parallel) on cards with many SMs (an RTX 3080): with the 2-bit weights (`KINDI_G5=0`), the
  output head was split into column parts that were never added, so every token was 0 and `kindi verify` failed.
- The installer asks each NVIDIA card on its own. A card whose `nvidia-smi` query fails (exit 14, "infoROM is corrupted")
  gets a warning and its compute capability from the card name; the other cards stay. Before, one such card made the
  installer report no NVIDIA GPU and install for the CPU.
- The DeepSeek installer (`tools/install_dsv41.sh`) picks the ram profile from 496 GiB of RAM, the measured need (the
  466 GiB model in the page cache, the server's 11 GiB and 18 GiB for the system). Before, it asked for 560 GiB, so a
  512 GB build got the nvme profile. It now prints the RAM it found and the RAM the profile needs.
- Blocking sync is off by default with the 580 driver (CUDA 13.0) and later, where it cost 5 to 15% of the write
  speed (#30). On an RTX 3080 with the 580 driver, the 2B writes about 11% faster. `KINDI_BLOCKING_SYNC=1` or `0`
  still decides. The installer's tensor-core profile keeps blocking sync on, as before.
- On native Windows, Kindi loads the CUDA driver from `nvcuda.dll` (#4). The prompt buffer of the 5-trit models is
  sized from the card's own weights, which gives back context that 0.4.0 took (#4). Not tested on native Windows yet.
- `kindi bench` logs the positions it measures and the ones that do not fit the context, and it measures the deepest
  position that fits. Before, positions past the context were dropped without a word (#33).
- `KINDI_P100_FP16_PROMPT=1` on a card whose cuBLAS cannot run the fp16 product (an RTX 3080 with CUDA 10.2) keeps
  the normal prompt path, with one warning. Before, a long prompt gave HTTP 500.
- Nemotron 3.5 Lightning reads `<think>`, `</think>`, `<tool_call>` and `<tool_response>` (and its closing tag) as
  one token each, as the model was trained (B19). Before, they were split into text pieces, as llama.cpp does with this
  GGUF. Prompts are 3 to 21 tokens shorter, and thinking replies follow the model's usual form. `KINDI_LT_ADDED_TAGS=0`
  restores the old split.
- A new test checks the tokenizer and the streamed text on Cyrillic, Greek, Arabic, Hebrew, Devanagari, Thai, CJK,
  Korean and emoji. No fault was found.

- **AH-Nanbeige4.3-33B on Volta (Tesla V100, sm_70)**: a prompt path of its own (#25). The prompt products run on the
  fp16 tensor cores (mma m8n8k4) and give the int32 dots of the int8 tensor-core kernels, so the products are the same
  bits as on sm_80 and later cards; the prompt attention also runs on the tensor cores. On one V100 32 GB the 33B reads
  566 tokens/s at 512 tokens and 474 at 16K (before: 144 and 106), and on two cards 1,015 at 8K. `kindi verify` of the
  33B now passes on Volta; before, one prompt differed from llama.cpp at a close margin. `KINDI_NB_VOLTA=0` gives the
  old path ([docs/CARDS.md](docs/CARDS.md)).

### Thanks

- atimelessrage contributed four pull requests to 0.7.0. They add the container RAM limit (#48), the tokenizer pack
  check (#49), one MTP draft by default (#72) and exact tensor-core activations for Hadamard-4096 packs (#76).
- The reports and tests that shaped this release came from epscheel, JugglerMaster, vntgeai, rojecas, kidkal,
  scuderiapk, ggeorgovassilis, Immac, exolvo, Rares14324, maximhausleiter and PlagaMedicum.
- Thank you to the Qwen team at Alibaba Cloud for Qwen3.8-Flash-Next and its MTP head. Thank you to ISTA-DASLab for
  the GSQ-RCO IQ3_S GGUF that Kindi runs. The README Acknowledgements name the other model sources.
### Docs

- CARDS.md: why `KINDI_TC=1` can end `kindi verify` prompt 1 early (#31). Its 8-bit activations move the logits by up
  to 0.15, and llama.cpp's own top two at those tokens differ by 0.0064 and 0.0105, so it is a near-tie.
  `KINDI_TC_HP=1` passes all four prompts (on an RTX 3080 it writes about 4% slower).
- CARDS.md: the GeForce GTX 1650 SUPER, 4 GB (Turing, sm_75), a user report (PR #24).
- CARDS.md: the Tesla V100 32 GB (Volta, sm_70), measured on one and two cards, with llama.cpp on the same cards.

## 0.6.1 (2026-11-16)

### Fixed

- The 2B with its MTP head on tensor-core cards stopped at start with `cuGraphAddKernelNode failed (1)`, and
  `kindi verify` failed. A kernel that the MTP head uses got one more argument in 0.5.0, and the MTP head did not pass
  it. Affected: 0.5.0 and 0.6.0, the installer's setup "Ternary-Bonsai-2 2B" on an NVIDIA card of compute capability
  8.0 or later with 9.5 GB or more (RTX 3060 12 GB, 3080, 3090, 40 series, A100), chosen with `--model 2b`,
  `--config` or the setup list (the recommended setup on these cards is the 27B, which does not use the MTP head), and
  any setup that sets `KINDI_MTP_HEAD` or `KINDI_SPEC_DRAFT_HEAD` by hand. The 2B, 4B, 9B and 27B without an MTP head
  were not affected.
- When a CUDA graph cannot be built, the error names the node, its grid and block, and the number of arguments.

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
