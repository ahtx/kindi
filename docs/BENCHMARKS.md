# Benchmarks

Each table compares Kindi and llama.cpp on the same card, with the same model file and the same context. "Read" is
the prompt reading speed: an N-token prompt divided by its read time. "Write" is the generation speed at that depth.
All figures are tokens per second, from one stream, greedy, with Kindi's exact head. When only one engine runs at a
context, the table says why the other does not ("out of memory", "crashed", "not measured").

Unless a table says otherwise, the method is the same for the two engines. They get the same token ids. The prompt
is read in segments up to each N, and the read time of N tokens is the sum of the segments. At each N, 32 tokens are
generated. Project page: [amirhusain.com/kindi](https://amirhusain.com/kindi/).

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

## Quadro K4000, 3 GB · Ternary-Bonsai-2-2B, 4B and 9B

2026-10-26 (Kindi) and 2026-10-22 (llama.cpp). Kindi 0.2.0 (tag `v0.2.0`, commit `f5b60fb`), ctx 24,576, f16 K/V,
chunked prompt reading. llama.cpp: Prism ML's fork, Vulkan build, commit `73bfe6c`, `-ngl 99 -fa on`, f16 K/V.
Model files Ternary-Bonsai-2-2B/4B/9B-PTQ1_0.gguf; `kindi verify` is ALL PASS on the three.

| Model | Context | Kindi read | Kindi write |
|---|---|---:|---:|
| 2B | 512 | 122.1 | 63.3 |
| 2B | 8,192 | 108.9 | 57.5 |
| 2B | 24,000 | 89.5 | 48.4 |
| 4B | 512 | 47.3 | 30.3 |
| 4B | 8,192 | 41.8 | 26.9 |
| 4B | 24,000 | 33.9 | 22.1 |
| 9B | 512 | 26.2 | 18.2 |
| 9B | 8,192 | 24.4 | 16.9 |
| 9B | 24,000 | does not fit (its maximum on 3 GB is 12,032) | does not fit |

- Each figure is the median of three complete passes, and the three agree within 0.1%. The prompt is the same
  24,576 token ids for the three models, read in chunks of 128 tokens (the automatic value on sm_30).
- The 9B does not reach 24,000 tokens on this card: with `KINDI_MEM_RESERVE_MB=256` its context fits 12,032
  tokens (208 MiB free). With the default reserve of 96 MB it fits 17,152 and leaves 46 MiB free.
- llama.cpp has no rows at 512 tokens and above on this card: at 128 tokens its prompt path already takes 100 s
  (1.28 tokens/s), the same `mul_mm` cliff documented for the K420.

llama.cpp on the same card and file, at the sizes that are practical there:

| llama.cpp (Vulkan, PTQ1_0) | tokens/s |
|---|---:|
| pp64 | 24.5 |
| pp128 | 1.28 |
| tg128 | 6.03 |

- With llama.cpp's Q2_0 file of the 2B (a different file, about 28 % more bytes per token): pp64 64.7 and
  tg128 10.8. The PTQ1_0 numbers above are the same-file comparison.
- The card is sm_30, so llama.cpp runs with Vulkan here for the reason in "Why llama.cpp runs with Vulkan on the
  Kepler and Maxwell cards".

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
