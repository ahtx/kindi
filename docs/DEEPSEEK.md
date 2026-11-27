# DeepSeek-V4.1-Flash on Kindi

DeepSeek-V4.1-Flash has 552B parameters, and 16B of them are active for each token. Kindi runs the published 4-bit
weights (the UD-Q4_K_XL GGUF, about 494 GB) on one NVIDIA card and the host computer. The engine is Kindi's own code.
The dense layers and the attention run on the card. The experts run on the CPU, from host memory or from a fast drive.
An optional second card holds the most used experts. This card can be any NVIDIA card from sm_50 on, with 4 GB or
more. The DSpark drafter uses the model's own three MTP stages. It drafts up to 5 tokens for each step. The model
checks every draft, so the output does not change.

## Requirements

| Part | Requirement |
|---|---|
| OS and CPU | Linux x86-64, AVX2 |
| Card | NVIDIA, sm_50 or newer; 12 GB or more for the dense layers and the drafter (an RTX 3060 12 GB works) |
| Expert card (optional) | any NVIDIA card sm_50+ with 4 GB or more; Kindi fills it with the most used experts |
| Software | NVIDIA driver, CUDA toolkit (`nvcc`), `gcc`, `python3` with `numpy` |
| Disk | about 530 GB free for the download and the converted drafter |
| RAM | see the two profiles below |

## The two host profiles

**ram (496 GiB of RAM or more, as `MemTotal` in `/proc/meminfo` reports it).** The whole model stays in the Linux
page cache. The 496 GiB are the 9 GGUF shards (466 GiB), the server's own memory (about 11 GiB) and room for the
kernel and the rest of the system (18 GiB), measured on the test host. A build with 512 GB of memory modules reports a
few GiB less than 512 in `MemTotal`. From 496 GiB it gets this profile, with little room left for other programs. The installer prints the RAM it
found and the RAM the profile needs, and `--profile ram` or `--profile nvme` overrides its choice. A host can have two CPU
sockets. Then Kindi puts the rows of each expert on the memory node of the cores that compute them. The service does
this before it starts (`python -m kindi.dsv41 numa MODEL --rowsplit`). So both memory controllers read at the same
time. The model file can be on any disk. Kindi reads the file one time at the first start, and then uses the copy in
RAM. On a hard disk, the first read takes 40 minutes or more.

**nvme (less RAM).** Kindi streams the experts from an NVMe drive (`KINDI_DSV_EXPERTS=stream`, see "Streamed
experts" below). The speed depends on the read speed of the drive and on the part of the experts that RAM holds. The
installer does not accept this profile on a hard disk. Two or more NVMe drives in RAID 0 increase the read speed. We
did not measure this profile yet.

## Install

```
bash tools/install_dsv41.sh                 # checks the host, picks the profile, downloads, starts the service
bash tools/install_dsv41.sh --dry-run       # prints the plan, changes nothing
```

| Option | What it does |
|---|---|
| `--dir DIR` | The folder for the model |
| `--port 8101` | The port of the server |
| `--ctx 65536` | The context length |
| `--profile auto\|ram\|nvme` | The host profile |
| `--no-service` | Does not make the service |
| `--skip-download` | Uses the files that are in `DIR/gguf` and `DIR/dspark` |
| `--yes` | Does not ask questions |
| `--force` | Replaces an `kindi-dsv41.service` that you have. Without it, the installer never replaces the service. |

The installer compares each file with the published sha256 lists in `tests/data/dsv41/`. If a download stops, run
the installer again. It continues from the point where it stopped.

The service is a systemd user service, `kindi-dsv41.service`. It gives the OpenAI API on `127.0.0.1:PORT`. At the
first start, it compiles the CPU and CUDA kernels into `~/.cache/kindi`, and it reads the model. This takes a few
minutes.

## Check the install

```
curl -s 127.0.0.1:8101/health
python3 -m kindi.dsv41 verify MODEL-00001-of-00009.gguf [--sha]
```

`verify` is the release gate. It runs 20 prompts of 64 tokens, with the reference implementation's greedy tokens as
input. It uses the numerics of the reference (`--flags 160`: an f16 cache and the attention order of the
reference). It passes when the top-1 agreement is 98.5% or more and each difference is a near-tie. A near-tie is a
reference margin of 0.75 or less. Kindi's default numerics use DeepSeek's own FP8 and FP4 roundings. These roundings
are different from the reference build, so the gate does not use them. The prompt set is not in the repository.
`KINDI_DSV_GATE` names its folder. `--sha` also compares every shard with the published sha256. This is slow: it
reads 494 GB.

## Sampling

`temperature` 0 is greedy (the default). When `temperature` is more than 0, the server samples from one
distribution, in this order: temperature, `top_k`, `top_p` and `min_p`. DSpark stays on. The server keeps a draft
with the probability that the model gives it. When it refuses a draft, it takes a sample from the rest of the
distribution. So the output has the same distribution as plain sampling (`tools/test_dsv41_sampling.py` checks
this).

The top-p sampler uses a partial stable sort by default (`KINDI_DSV_TOPP_PARTIAL=0` restores the full sort). The sampler uses the same
normalization, tie order and probability cutoff. It sorts the full vocabulary if the candidate set does not
cover the requested probability mass. It applies to sampled requests that use a `top_p` below `1`.

## Hot experts

`tools/dsv41_hotplan.py` makes the hot-expert list from measured counts of the expert picks. It uses Kindi's shared
MoE placement (`kindi/moe.py`), the same policy as the other MoE engines. The installer makes the plan for the VRAM
of the expert cards. With the counts in the release, 7 GB of VRAM on the expert card holds 399 experts. These experts
get 36% of the expert picks.

## Streamed experts

The routed experts are 289 GB of the 494 GB file. In the ram profile, Kindi reads them in place from the memory-mapped
file (`KINDI_DSV_EXPERTS=mmap`, the default). With less RAM, the kernel then loads 4 KB pages one at a time when a
token needs them, and it drops pages by its own LRU order.

`KINDI_DSV_EXPERTS=stream` (the installer's nvme profile) uses Kindi's expert store (`kindi/xstore.c`) instead:

- Each call reads only the experts that its tokens route to. An expert is 17.9 MB (gate, up and down). The store
  reads it with direct I/O (`O_DIRECT`, past the page cache) in chunks of 2 MB, with 32 reads in flight.
- The experts stay in a RAM cache with a fixed size (`KINDI_XS_GB`). The cache holds the bytes of the file without a
  change, so the expert kernels give the same bits as in the mmap mode.
- The store pins the most used experts that the cards do not hold (half of the cache). The list is the hot-expert
  list of the installer (the shared MoE placement, `kindi/moe.py`). The store drops the other experts by a use count
  that halves every 64 forward steps. Experts used in the last three steps stay first.
- Prefetch: when the experts of layer l start, a thread applies the routers of layers l + 1 and l + 2 to the input of
  layer l. It reads the top 8 experts of each token ahead. On real activations, this guess holds 79% of the experts
  of the next layer and 74% of the layer after it (`tools/dsv41_prefetch_recall.py`). A wrong guess costs only a read.
- A DSpark verify step computes the token and its drafts in one call, so all of them share the reads of their
  experts.
- If no slot is free, or a read fails, the call reads that expert from the mapped file. The counters report it.

| Variable | Default | What it does |
|---|---|---|
| `KINDI_DSV_EXPERTS` | `mmap` | `stream` turns the store on |
| `KINDI_XS_GB` | 70% of the free RAM less 8 GB | The RAM of the expert cache (the installer: 70% less 16 GB) |
| `KINDI_XS_PIN` | 0.5 | The part of the cache for pinned experts |
| `KINDI_XS_PIN_LIST` | `KINDI_HOT_LIST` | The order of the pins ("layer expert count" lines) |
| `KINDI_XS_THREADS`, `KINDI_XS_CHUNK` | 32, 2 MB | The read threads and the size of one read |
| `KINDI_XS_PF_DEPTH`, `KINDI_XS_PF_K` | 2, 8 | Prefetch: the layers ahead and the experts per token (0 = off) |
| `KINDI_XS_DROP` | 0 | 1 drops the page cache of the expert shards after the start (the installer sets it) |
| `KINDI_XS_CHECK` | 0 | 1 compares each read with the mapped file (for tests) |

The checks: `tools/test_xstore.py` (a synthetic file), `tools/dsv41_xstore_check.py` (three layers of the real model)
and `tools/dsv41_xstore_e2e.py` (the whole model on the CPU, a few tokens). Each one compares the stream mode with the
mmap mode, and the results must be the same bits. `tools/dsv41_nvme_measure.sh` measures both modes under a RAM cap.

## Edits far back in a conversation

The attention window ring keeps only the last positions of each layer, so the state in memory can go back at most one
prompt chunk (`tmax`, 512 tokens). Before 0.7.0, a request that changed the conversation further back (an agent that
compresses its history and rewrites its first message, or an edit at an early turn) read the whole prompt again from
the start, unless a saved state on disk matched.

From 0.7.0 the engine copies the window ring to host RAM at every 1,024 positions while it reads (about 11 MB for each
copy). The compressed caches are indexed by position, so they stay valid on the card. Such a request now resumes at
the nearest copy at or before its first changed token, and reads only the tokens after it. The copies are taken where
the chunks of a full read end, so the result is the same as a full read.

| Variable | Default | What it does |
|---|---|---|
| `KINDI_DSV_RING_EVERY` | 1024 | Positions between two ring copies (rounded up to a multiple of 2 `tmax`); 0 = off |
| `KINDI_DSV_RING_MAX` | 256 | The most copies kept; past it, every second one goes and the spacing doubles |

`tools/test_dsv41_ring.py` checks the bookkeeping with a small fake engine.

Measured on 11-15 on the test host with the service's cards and settings (3060 + M4000 expert card,
`KINDI_DSV_GPU=2 KINDI_SPLIT=1`), without `--cache-dir`, with `tools/dsv41_ring_measure.py` on an agent conversation of
26 messages and 24,386 tokens (`tools/dsv41_ring_request.py`, made from this repository's files) whose message 6 (between
positions 3,072 and 4,096) is edited:

| Request | Ring off (`KINDI_DSV_RING_EVERY=0`) | Ring on (default) |
|---|---|---|
| original | 24,386 tokens read, 254.8 s | 24,386 tokens read, 255.4 s |
| edited (message 6) | 24,391 tokens read again, 256.1 s | 21,319 tokens read (from the copy at 3,072), 228.2 s |
| back to the original | 24,386 tokens read again, 257.6 s | 21,314 tokens read, 227.4 s |

The answers are the same with the ring on and off. The copies cost nothing that shows in the first read. The gain
follows the position of the edit: here the edit was early, so 3,072 tokens were saved; an edit at position 20,000
would read only the tokens after the copy at 19,456. (An earlier 11-15 run crashed in the first copy; the copy's size did not
count the drafter's rings. `tools/dsv41_ring_test.cu` checks the fix on any card.)

## Measured speeds

The test host: one GeForce RTX 3060 12 GB (PCIe 3.0 x16), a Quadro M4000 8 GB as the expert card, and two Xeon
E5-2673 v4 CPUs (AVX2) with all memory channels in use. The ram profile, with the model in the page cache, a context
of 65,536 and Kindi 0.6.0, with DSpark on (the default).

| Speed | Tokens/s |
|---|---|
| Write (the mean over agent prompts) | 18.7 |

Prompt reading on real text is about 95 to 100 tokens/s at 2K and at 8K tokens. The CPU expert work sets this speed.

The prompt chunk (`tmax`) is 512 by default. On a card with enough VRAM, start the server with
`--tmax 1024 --adaptive-prefill` to use 512-token chunks for short uncached prompts and pipelined
1,024-token chunks for uncached spans of at least 8,192 tokens. The 1,024 setting reserves more VRAM
when the server loads, including for short requests. The installer uses 512.
With DSpark, 81,920 context and `KINDI_DSV_CKV4=1`, adaptive prefill on the RTX 3060 and M4000 read at
102.5 tokens/s on a 13K agent prompt versus 91.4–92.5 at fixed 512. The 4K read and combined write rates
stayed within the fixed-512 range. The 3060 used 756 MiB more after the replay.
`KINDI_DSV_CKV4=1` is experimental: `kindi.dsv41 verify` gives 1,260 of 1,280 (98.4%) with it, below the
1,266 of 1,280 (98.9%) gate of the default KV cache. Without CKV4, the RTX 3060 runs out of memory at
81,920 context with `--tmax 1024`.
