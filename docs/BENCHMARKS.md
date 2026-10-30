# Benchmarks

Each table compares Kindi and llama.cpp on the same card, with the same model file and the same context. "Read" is
the prompt reading speed: an N-token prompt divided by its read time. "Write" is the generation speed at that depth.
All figures are tokens per second, from one stream, greedy, with Kindi's exact head. When only one engine runs at a
context, the table says why the other does not ("out of memory", "crashed", "not measured").

Unless a table says otherwise, the method is the same for the two engines. They get the same token ids. The prompt
is read in segments up to each N, and the read time of N tokens is the sum of the segments. At each N, 32 tokens are
generated. Project page: [amirhusain.com/kindi](https://amirhusain.com/kindi/).

## Kindi 0.3.0: Nemotron, the CPU engine and the prompt cache

The tables in this section are new in 0.3.0. The Bonsai tables after them are from Kindi 0.2.0. We measured the
0.3.0 engine against them again on the K1200 cards: the Bonsai rows are within 0.5%.

### NVIDIA Nemotron 3.5 Bolt E4B (UD-Q3_K_XL, sha256 3f8032b9…)

The same GGUF for both engines, f16 K/V, the same token ids, greedy, both engines warmed first.

| Hardware | Context | Kindi read | llama.cpp read | Kindi write | llama.cpp write |
|---|---|---:|---:|---:|---:|
| GeForce GTX 1080 8 GB | 512 | 742.5 | 1087.9 | 55.10 | 56.42 |
| GeForce GTX 1080 8 GB | 8,192 | 922.2 | 896.6 | 51.11 | 53.05 |
| GeForce GTX 1080 8 GB | 16,384 | 837.8 | 730.7 | 47.76 | 50.35 |
| GeForce GTX 1080 8 GB | 24,000 | 746.9 | 626.5 | 45.20 | 48.22 |
| Quadro K1200 4 GB (ctx 8,448) | 512 | 66.8 | 17.6 | 10.41 | 8.27 |
| Quadro K1200 4 GB (ctx 8,448) | 8,000 | 73.9 | 12.0 | 9.07 | 7.67 |
| 2× Quadro K1200, split (ctx 8,448) | 512 | 122.5 | 15.3 | 17.15 | 7.44 |
| 2× Quadro K1200, split (ctx 8,448) | 8,000 | 124.2 | 19.0 | 16.08 | 7.37 |
| GeForce RTX 3080, 190 W, one chat request | — | 2,614 | 2,630 | 123.1 | 88.0 |

Builds: GTX 1080: Kindi 0.3.0-dev 849840a, llama.cpp CUDA b11804 (c7fc509), 2026-10-25. K1200: Kindi 0.3.0-dev
(5350b87 + the server fix), llama.cpp Vulkan 1b29621 (llama.cpp splits by layers on two cards), 2026-10-24. RTX 3080:
Kindi 0.3.0-dev b340cbe, gpu-unlock loaded, 2026-10-24. On the GTX 1080, llama.cpp reads a 512-token prompt faster
and writes 2-7% faster; Kindi reads longer prompts faster.

### The CPU engine (Ternary-Bonsai-2, AVX2)

Read: a ~2,000-token prompt with no cache reuse. Write: 256 tokens. Median of 3 runs, each engine at its best thread
count. llama.cpp: Prism ML's fork, CPU build.

| CPU | Model | Kindi read | llama.cpp read | Kindi write | llama.cpp write |
|---|---|---:|---:|---:|---:|
| AMD Ryzen 9 7940HS (8 cores) | 2B | 90.4 | 60.5 | 64.8 | 23.6 |
| AMD Ryzen 9 7940HS | 4B | 37.8 | 25.7 | 29.1 | 16.5 |
| Intel Core i7-8700 (6 cores) | 2B | 42.50 | 33.45 | 28.92 | 18.12 |
| Intel Core i7-8700 | 4B | 17.02 | 12.97 | 12.61 | 7.75 |

### The prompt cache, GTX 1080

Seconds for each request (prompt and reply), `examples/prompt_cache_bench.py`. Kindi with its cache off and on; llama.cpp
(b11804) with its cache on. The replies with Kindi's cache off and on were the same in all 12 requests of each model.

| Job, request | Nemotron, Kindi off | Nemotron, Kindi on | Nemotron, llama.cpp on | 27B, Kindi off | 27B, Kindi on |
|---|---:|---:|---:|---:|---:|
| Agent, turn 2 | 2.76 | 0.67 | 1.29 | 15.33 | 2.52 |
| Agent, turn 3 | 2.96 | 0.86 | 1.30 | 15.65 | 2.93 |
| Regenerate | 2.69 | 0.43 | 4.93 | 15.47 | 1.85 |
| Document, question 2 | 11.03 | 0.70 | 3.16 | 49.42 | 14.53 |
| Document, question 3 | 10.71 | 0.42 | 3.10 | 47.19 | 11.46 |
| Two agents, A turn 2 | 2.58 | 0.58 | 1.40 | 14.68 | 2.61 |
| Two agents, B turn 2 | 2.57 | 0.53 | 1.44 | 14.80 | 2.79 |

### Radeon 780M

2026-10-25. Kindi 0.3.0 development build (b5230a3), `HSA_OVERRIDE_GFX_VERSION=11.0.2`, `amdgpu.cwsr_enable=0`.
The same token ids for both engines, greedy, one pass, 32 write tokens at each depth, both engines warmed first.
K/V: q8_0 for the 27B, f16 for Nemotron. llama.cpp: `-ngl 99 -fa on --parallel 1 -b 2048 -ub 512 -c 24576`; the
Prism fork (ROCm build) for the 27B, upstream b10700 (ROCm build) for Nemotron. Read / write tokens/s:

| Model | Context | Kindi | llama.cpp ROCm |
|---|---:|---:|---:|
| 27B | 512 | 103.7 / 10.61 | 60.5 / 10.33 |
| 27B | 8,192 | 95.6 / 9.99 | 58.2 / 8.52 |
| 27B | 24,000 | 80.8 / 8.75 | 54.5 / 6.13 |
| Nemotron (experimental) | 512 | 185.9 / 19.11 | 669.5 / 24.02 |
| Nemotron (experimental) | 8,192 | 391.9 / 20.03 | 598.7 / 21.36 |
| Nemotron (experimental) | 24,000 | 239.4 / 17.01 | 476.6 / 19.09 |

With the 27B, Kindi reads 1.5 to 1.7 times as fast at each context and writes 1.03, 1.17 and 1.43 times as fast.
Kindi's Nemotron support is experimental, and we are still optimizing it; on this chip llama.cpp is faster with it.

## Quadro K6000, 12 GB

2026-10-25. Quadro K6000 12 GB (Kepler GK110, sm_35) passed through to a KVM virtual machine on a dual Xeon E5-2673 v4 host, NVIDIA driver 470.256.02. The same model files and the same token ids for both engines, greedy, 32 tokens of write at each depth. Kindi 0.3.0 development build (597d2d8) in the CUDA 10.2 image, q8_0 K/V for the 27B, f16 for Nemotron. llama.cpp: Vulkan builds (the Prism fork for the 27B, upstream 1b29621 for Nemotron), `-ngl 99 -fa on`. Read / write tokens/s:

| Model | Context | Kindi | llama.cpp Vulkan |
|---|---:|---:|---:|
| 27B | 512 | 36.2 / 20.42 | 7.0 / 2.07 |
| 27B | 8,192 | 30.2 / 16.36 | 6.9 / 2.15 |
| 27B | 24,000 | 26.2 / 13.09 | did not finish in 1 hour |
| Nemotron (experimental) | 512 | 187.5 / 20.05 | 90.7 / 10.70 |
| Nemotron (experimental) | 8,192 | 182.4 / 19.08 | 33.8 / 9.80 |
| Nemotron (experimental) | 24,000 | 163.6 / 17.40 | 15.6 / 8.56 |

`kindi verify`: the 27B 4 of 4 token-exact; Nemotron 4 of 4 (the first 16 to 40 of 40 tokens agree). With the 27B,
Kindi reads 4 to 5 times as fast and writes 7 to 10 times as fast. Both engines were warmed first. The tuning settings that we tried (chunk size,
tile, split, blocking sync, batch) changed the speed by 3% or less.

## Quadro K420, 1 GB · Ternary-Bonsai-2-2B

2026-10-23. Kindi 0.2.0, ctx 24,576, f16 K/V. llama.cpp: the prebuilt Vulkan release,
`-ngl 99 -fa on -c 24640 -b 2048 -ub 512`, f16 K/V. Model file sha256 36061ad4….

| Context | Kindi read | llama.cpp read | Kindi write | llama.cpp write |
|---|---:|---:|---:|---:|
| 512 | 33.3 | 1.2 | 20.8 | 1.65 |
| 8,192 | 29.7 | crashed (Vulkan ErrorDeviceLost after 17 minutes) | 18.6 | crashed |
| 24,000 | 24.2 | not measured (it crashed at 8,192) | 15.3 | not measured |
| 32,768 | Kindi: does not fit (its maximum on 1 GB is 24,576) | loads; not measured | does not fit | not measured |
| 65,536 | does not fit | does not load (an 805 MB allocation fails) | does not fit | does not load |

## Quadro K1200, 4 GB, one card · 2B, 4B and 9B

2026-10-23 (Kindi) and 2026-10-22 (llama.cpp). Kindi 0.2.0, ctx 24,576, f16 K/V, chunked prompt reading. llama.cpp: Prism ML's fork,
Vulkan build, `-ngl 99 -fa on -c 24640 -b 2048 -ub 512 --cache-ram 0`, f16 K/V. Each model had a 1-hour limit.

| Model | Context | Kindi read | llama.cpp read | Kindi write | llama.cpp write |
|---|---|---:|---:|---:|---:|
| 2B | 512 | 194.5 | 26.7 | 81.0 | 20.8 |
| 2B | 8,192 | 178.7 | 28.1 | 73.3 | 21.9 |
| 2B | 24,000 | 152.3 | 26.5 | 58.8 | 18.6 |
| 4B | 512 | 77.4 | 11.1 | 39.2 | 8.7 |
| 4B | 8,192 | 70.2 | 10.8 | 34.3 | 8.1 |
| 4B | 24,000 | 58.9 | 10.2 | 26.4 | 6.9 |
| 9B | 512 | 41.7 | 5.8 | 23.4 | 5.9 |
| 9B | 8,192 | 39.2 | 5.7 | 21.5 | 5.6 |
| 9B | 24,000 | 35.6 | not finished (the 1-hour limit ended during this prompt) | 18.1 | not measured |

The largest context on the card:

| Model | Kindi | llama.cpp |
|---|---|---|
| 2B | 131,072 (all that was asked) | loads at 65,536 (larger was not tried) |
| 4B | 86,272 | loads at 65,536 (larger was not tried) |
| 9B | 47,360 | loads at 65,536 (Kindi does not fit there) |

## Two Quadro K1200 cards · Ternary-Bonsai-2-27B

2026-10-23 (Kindi) and 2026-10-21/22 (llama.cpp). Kindi 0.2.0, the tensor-parallel engine, q8_0 K/V. llama.cpp: Prism ML's fork, Vulkan
build, `-ngl 99 -sm layer -fa 1 -ctk q8_0 -ctv q8_0`, with its default layer split (an even split does not load).

| Context | Kindi read | llama.cpp read | Kindi write | llama.cpp write |
|---|---:|---:|---:|---:|
| short (Kindi: generation at position 128; llama.cpp: below position 150) | - | - | 12.6 | 1.69 |
| 512 (the same 512 token ids) | 19.5 | 1.63 | - | - |
| 1,000 | not measured | does not run (a 508 MB Vulkan allocation fails) | not measured | does not run |

Kindi's 512-token read is the best of 3 from an empty state. llama.cpp's was measured on 2026-10-22 with llama-server
`-ngl 99 -sm layer -fa on -ctk q8_0 -ctv q8_0 -c 1024`, with the token ids sent to `/completion`.

At 1.3 tokens/s, a 24,000-token prompt takes about 5 hours on llama.cpp, so there are no deeper rows.

## GeForce GTX 1080, 8 GB · Ternary-Bonsai-2-27B

2026-10-22 (Kindi) and 2026-10-21 (llama.cpp). Kindi 0.2.0, one-card plan engine, 5-trit weights, int8 chunked prompt reading, ctx
65,536, q8_0 K/V. llama.cpp: Prism ML's fork, CUDA build for sm_61, `-ngl 99 -fa on -ctk q8_0 -ctv q8_0 -b 2048
-ub 512 -c 49280`. Model file sha256 53107f53….

| Context | Kindi read | llama.cpp read | Kindi write | llama.cpp write |
|---|---:|---:|---:|---:|
| 512 | 254.0 | 98.1 | 27.9 | 19.4 |
| 8,192 | 227.1 | 96.2 | 25.7 | 18.1 |
| 32,768 | 169.1 | 84.9 | 19.7 | 14.3 |
| 49,152 | 144.6 | 78.6 | 17.2 | 12.6 |
| 64,000 | 127.8 | out of memory (it does not load at 64K) | 15.4 | out of memory |

## GeForce RTX 3080, 10 GB · Ternary-Bonsai-2-27B

2026-10-22, one physical card (cluster GPU 7), both engines. The cluster cards have a power limit of 190 W; 380 W is
the default of the card. Kindi 0.2.0 (the 190 W columns; the 380 W columns: see below), tensor-core engine,
`KINDI_TC=1 KINDI_FD=1 KINDI_ATT_TC=1`, exact head, q8_0 K/V. llama.cpp: Prism ML's fork 1180918, CUDA 12.4 build
for sm_86. llama-bench `-ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -r 1`: read pp512 / 3000 / 12000 / 22000 from an empty
cache, write tg32 at a depth of 1K / 32K / 64K / 90K. No CPU limit (`--cpus 4.0`).

| Context | Kindi read 190 W | llama.cpp read 190 W | Kindi read 380 W | llama.cpp read 380 W |
|---|---:|---:|---:|---:|
| 512 | 1,093 | 936 | 1,361 | 1,217 |
| 3,000 | 1,029 | 995 | 1,282 | 1,265 |
| 12,000 | 944 | 926 | 1,171 | 1,170 |
| 22,000 | 855 | 855 | 1,050 | 1,074 |

In the 380 W columns, Kindi is the dev build release-tcspec6 (2026-10-22), whose exact path is the same code as 0.2.0.

| Depth | Kindi write 190 W | llama.cpp write 190 W | Kindi write 380 W | llama.cpp write 380 W |
|---|---:|---:|---:|---:|
| 1,000 | 63.2 | 52.2 | 66.4 | 56.5 |
| 32,768 | 56.3 | 41.0 | 60.1 | 47.0 |
| 65,536 | 49.8 | 28.0 | 54.7 | 37.9 |
| 90,000 | 44.0 | 23.5 | 51.2 | 32.9 |

In the 380 W columns, Kindi is the dev build release-tcspec6 (2026-10-22), whose exact path is the same code as 0.2.0.

Kindi uses about 0.02-0.05 CPU cores per stream (0.2.0); llama.cpp about 0.77, and both stay within 5% on one core
(`--cpus 1.0`, llama.cpp `-t 1 -tb 1`; release-tcspec6 run).

Agent prompts: 30 greedy and 20 sampled (temperature 0.7) pi prompts of 1,100 to 10,500 tokens, 128 tokens out,
through the servers (llama-server `-fa on -ctk q8_0 -ctv q8_0 -c 81920 -np 1 --cache-ram 0`). Write speed, greedy /
sampled:

| Engine | 190 W | 380 W |
|---|---:|---:|
| llama.cpp | 48.7 / 54.9 | 53.2 / 60.8 |
| Kindi, exact | 61.8 / 57.7 | 66.9 / 58.6 |
| Kindi + the 2B draft model (`KINDI_SPEC=draft2b`) | 99.9 / 91.3 | 107.9 / 90.9 |
| llama.cpp + the 2B draft model (`-md`) | does not load | does not load |

In the 380 W column, Kindi is the dev build release-tcspec6 (2026-10-22): its exact path is the same code as 0.2.0, and its draft-model row is from before 0.2.0's draft-model fix.

With the draft model, Kindi's greedy output is the same as without it (30 of 30 prompts). llama.cpp 1180918 does
not load the 2B as a draft model for the 27B: the two files have different Hadamard metadata.

## GeForce RTX 4060 Laptop, 8 GB · Ternary-Bonsai-2-27B, tensor-core chunk attention

2026-10-30. One long prompt (39,292 tokens) on an RTX 4060 Laptop (Ada, sm_89, 8 GB, 60 W cap), driver 580.178.04,
Kindi 0.3.0, exact head, q8_0 K/V, greedy (temperature 0, seed 42). The same prompt before and after; the greedy
output was token-for-token identical in both runs.

The default chunked prompt path runs the attention and the chunk GEMM on the general CUDA cores. Enabling the
tensor-core plans (`KINDI_TC=1 KINDI_ATT_TC=1`) moves both onto the tensor cores; the attention arithmetic stays
bit-identical, so the output does not change.

| Path | Context | Attention GPU time | Chunk GEMM GPU time | Prompt GPU time | Prompt read |
|---|---|---:|---:|---:|---:|
| default | 64,512 | 67.17 s (30.9%) | 127.43 s (58.7%) | 217.18 s | 225.88 s (173.9 tok/s) |
| `KINDI_TC=1 KINDI_ATT_TC=1` | 55,040 | 39.05 s (39.0%) | 37.11 s (37.1%) | 100.10 s | 109.73 s (358.1 tok/s) |

On this card the tensor-core path about halves the prompt read time. The trade-off is VRAM: the tensor-core plans
and buffers cost about 400 MiB, so the context falls from 64,512 to 55,040 tokens on 8 GB.

## Why llama.cpp runs with Vulkan on the Kepler and Maxwell cards

The K420 has compute capability 3.0 (sm_30). CUDA 11 and later cannot compile code for it, and current llama.cpp
does not build with CUDA 10.2, the last toolkit for this card. The Vulkan backend runs on it with all layers on the
GPU, so the Vulkan build is the fair comparison. On the K1200 (sm_50) host, only the Vulkan build was available.

## Other published results

We looked for a published measurement of a modern language model on a Quadro K420, and on any Kepler GPU of the
sm_30 generation (GT 640/730, K420/K600). We found none. To our knowledge, Kindi is the fastest way to run a
current-generation LLM on this card. We have not had it independently verified. If you know of a faster result,
please tell us.

## How to reproduce

`kindi verify` checks the output against llama.cpp. `kindi bench` gives the write speed at several positions and
the prompt reading speed. [CARDS.md](CARDS.md) gives the settings for each card.
