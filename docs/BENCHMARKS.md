# Benchmarks

Each table compares Kindi and llama.cpp on the same card, with the same model file and the same context. "Read" is
the prompt reading speed: an N-token prompt divided by its read time. "Write" is the generation speed at that depth.
All figures are tokens per second, from one stream, greedy, with Kindi's exact head. When only one engine runs at a
context, the table says why the other does not ("out of memory", "crashed", "not measured").

Unless a table says otherwise, the method is the same for the two engines. They get the same token ids. The prompt
is read in segments up to each N, and the read time of N tokens is the sum of the segments. At each N, 32 tokens are
generated. Project page: [amirhusain.com/kindi](https://amirhusain.com/kindi/).

## Kindi 0.6.0 (in development): Nemotron 3.5 Lightning 30B-A3B (ternary, Q2_0)

Model file: Ternary-Nemotron-3.5-Lightning-30B-A3B-GGUF.gguf, sha256 38854de6…, f16 K/V for both engines.
llama.cpp: commit 785322d (CUDA, `llama-bench -ngl 99 -fa 1`). Read: `-p N -n 0`, one prompt of N tokens. Write:
`-n 64 -d N`. Kindi: `kindi bench`, the same prompt lengths; it writes 64 tokens after 16 warm-up tokens at the
same depth. Kindi's numbers are for the default settings (exact head, no speculation).

### GeForce RTX 3080, 10 GB, 190 W

| context | Kindi read | llama.cpp read | Kindi write | llama.cpp write |
|---:|---:|---:|---:|---:|
| 512 | 4,258 | 2,725 | 310.4 | 202.2 |
| 8,192 | 4,114 | 2,597 | 298.9 | 197.6 |
| 32,768 | 3,777 | 2,138 | 276.6 | 182.4 |
| 60,320 (Kindi's largest with all experts on the card) | 3,435 | 1,670 | 255.0 | 168.2 |

### GeForce RTX 3090, 24 GB, 220 W (rented machine)

| context | Kindi read | llama.cpp read | Kindi write | llama.cpp write |
|---:|---:|---:|---:|---:|
| 512 | 4,786 | 3,715 | 340.5 | 284.3 |
| 8,192 | 4,520 | 3,663 | 325.5 | 280.5 |
| 32,768 | 4,014 | 3,390 | 295.1 | 261.4 |

Write test on cards with host-memory experts (the GTX 1080 and the RTX 2060 with 6 GB held back): the random prompt
of `kindi bench` makes the next tokens use a small set of experts again and again, and that makes the decode slots
look better than they are. So for these cards the write values come from a text: the first N tokens of a long text
(the AGPL licence and Kindi documents, 77,035 tokens) as the prompt, then 128 greedy tokens, the same token ids
for both engines (`tools/lt_rtw.py`; llama.cpp through `llama-server` `/completion` with `ignore_eos`).
The read values are from `kindi bench` and `llama-bench`.

### GeForce RTX 2070 SUPER, 8 GB, PCIe 3.0 x16 (rented machine, 11-10)

An 8 GB card: Kindi keeps 74 to 122 of the 128 experts of each layer on the card (by measured use,
`kindi/moe.py`), 10 decode slots per layer, and the rest in host memory. llama.cpp 785322d (CUDA, built for sm_75)
also has to put experts in host memory: `-ngl 99 -fa 1 -ncmoe 12` is the lowest value that fits (the experts of
12 layers on the CPU). Read: `kindi bench` and `llama-bench -p N -n 0 -r 3`. Write: the text test above (a held-out
Wikipedia text, the same token ids for both engines, `tools/lt_rtw.py`).

| context | Kindi read | llama.cpp read | Kindi write | llama.cpp write |
|---:|---:|---:|---:|---:|
| 512 | **2,121** | 1,137 | **116.4** | 113.5 |
| 8,192 | **2,129** | 1,137 | **120.3** | 111.8 |

`kindi verify` on this card: PASS (5 of 12 prompts token-exact, 7 end at a llama.cpp tie, 0 fail).

### GeForce RTX 2060, 12 GB (rented machine)

| context | Kindi read | llama.cpp read | Kindi write | llama.cpp write |
|---:|---:|---:|---:|---:|
| 512 | 1,952 | 1,256 | 164.0 | 143.7 |
| 8,192 | 1,992 | 1,251 | 155.1 | 139.2 |
| 32,768 | 1,734 | 1,156 | 134.6 | 128.4 |

The same card with 6 GB held back (6,494 MB). Kindi: `KINDI_LT_RESERVE_MB=6494`, 60 of 128 experts per layer on
the card, 6 decode slots per layer, the rest in host memory. llama.cpp: a separate process holds 6,494 MB of the
card (cudaMalloc), and `-ngl 99 -fa 1 -ncmoe 28` is the lowest value that fits (the experts of 28 layers on the CPU).

| context | Kindi read | llama.cpp read | Kindi write | llama.cpp write |
|---:|---:|---:|---:|---:|
| 512 | 1,415 | 520 | 52.7 | 51.5 |
| 8,192 | 1,423 | 526 | 54.0 | 51.0 |
| 32,768 | 1,415 | 515 | 57.1 | 47.0 |

Write: the text test above (another machine of the same host, PCIe x8). With the random prompt of `kindi bench`,
Kindi writes 147.3 / 129.0 / 110.3 and llama.cpp (`llama-bench`) 54.4 / 53.7 / 51.8.

### GeForce GTX 1080, 8 GB (PCIe 2.0 x16)

Kindi keeps 95 of 128 experts per layer on the card, 6 decode slots per layer for the last used host experts,
and the rest in host memory. llama.cpp also has to put
experts in host memory to fit 8 GB: `-ngl 99 -fa 1 -ncmoe 9` (the experts of 9 layers on the CPU), the lowest value
that fits (6 to 8 run out of memory). The host CPU (Xeon W3503, 2 cores) has no AVX, so llama.cpp was built for
SSE4.2 only.

| context | Kindi read | llama.cpp read | Kindi write | llama.cpp write |
|---:|---:|---:|---:|---:|
| 512 | 745 | 646 | 75.0 | 4.7 |
| 8,192 | 687 | 615 | 89.5 | 4.7 |
| 32,768 | 536 | 495 | 73.9 | 4.0 |

Write: the text test above. With the random prompt of `kindi bench`, Kindi writes 111.1 / 106.1 / 86.5, and
`llama-bench` gives llama.cpp 3.3 / 4.0 / 4.3 (the mean of 6 runs; single runs from 2.5 to 5.1).

### Tesla P100-SXM2, 16 GB (rented machine)

| context | Kindi read | llama.cpp read | Kindi write | llama.cpp write |
|---:|---:|---:|---:|---:|
| 512 | 654 | 516 | 155.4 | 122.6 |
| 8,192 | 702 | 506 | 144.8 | 118.4 |
| 32,768 | 605 | 449 | 118.4 | 108.0 |

llama.cpp on the P100: the same commit, built for sm_60 with CUDA 12.2.

### Quadro K6000, 12 GB (stock clocks)

| context | Kindi read | llama.cpp read | Kindi write | llama.cpp write |
|---:|---:|---:|---:|---:|
| 512 | 206.3 | 164.6 | 61.9 | 35.1 |
| 8,192 | 191.7 | 140.5 | 48.3 | 28.7 |
| 32,768 | 124.7 | 118.1 | 32.8 | 23.3 |

llama.cpp on the K6000: the same commit, built for sm_35 with CUDA 11.4. The K6000 rows were measured before the
last changes of the 0.4.5 branch. These changes do not change the K6000 path: Kepler cards keep the separate decode
launches, and the change to lt_attn_pre8 gives the same results bit for bit.

## Kindi 0.5.0: NVIDIA CMP 170HX (GA100, sm_80, 40 GB)

| Item | Value |
|---|---|
| Cards | 1 or 2 NVIDIA CMP 170HX, 40 GB HBM2e, device id 0x2082, VBIOS 92.00.6D.00.0A, 250 W, PCIe gen4 x16 (rented) |
| Kindi | v0.5.0-dev f6ca4df, `kfair` (one pass; read to the depth, then the mean of 3 x 32 written tokens) |
| llama.cpp | c96baf4, `llama-bench -ngl 99 -fa 1` (write `-n 64 -d D -r 5`, read `-p N -r 3`) |
| K/V cache | 27B: q8_0 for both engines; 33B: f16 for both engines |
| Verify | `kindi verify` PASS on every row (27B two cards: 3 exact, 1 llama.cpp tie; 33B: 8 exact, 3 ties) |

Write, tokens/s:

| Setup | 512 | 8K | 32K | 64K |
|---|---|---|---|---|
| 27B, Kindi, one card | 113.4 | 110.1 | 92.9 | 77.2 |
| 27B, llama.cpp, one card | 106.1 | 96.3 | 74.1 | 56.1 |
| 27B, Kindi, two cards | 121.2 | 118.8 | 107.8 | 93.4 |
| 27B, llama.cpp, two cards | 107.3 | 97.8 | 76.1 | 58.2 |
| 33B, Kindi, one card | 59.7 | 55.5 | 42.0 | 32.6 |
| 33B, llama.cpp, one card | 48.5 | 45.9 | 40.0 | 34.3 |
| 33B, Kindi, two cards | 64.0 | 59.1 | 45.0 | 35.3 |
| 33B, llama.cpp, two cards | 53.1 | 50.2 | 43.2 | 37.0 |

Read, tokens/s:

| Setup | 512 | 8K | 32K | 64K |
|---|---|---|---|---|
| 27B, Kindi, one card | 1,785 | 1,607 | 1,249 | 956 |
| 27B, llama.cpp, one card | 1,398 | 1,381 | 1,219 | 1,062 |
| 27B, Kindi, two cards | 1,350 | 1,264 | 1,030 | 820 |
| 27B, llama.cpp, two cards | 1,405 | 1,373 | 1,208 | 1,039 |
| 33B, Kindi, one card | 1,170 | 1,251 | 1,013 | 799 |
| 33B, llama.cpp, one card | 1,169 | 1,141 | 957 | 788 |
| 33B, Kindi, two cards | 1,519 | 2,177 | 1,869 | 1,505 |
| 33B, llama.cpp, two cards | 1,139 | 2,046 | 1,793 | 1,494 |

Kindi writes faster on every 27B row, and on the 33B up to 32K tokens. On one card, Kindi reads the 27B faster up to
32K tokens. We did not test the CMP 170HX board with 8 GB.

## Kindi 0.5.0: AMD Radeon 840M (Ryzen AI 5 340, RDNA 3.5, gfx1152)

| Item | Value |
|---|---|
| Graphics | Radeon 840M, 4 compute units, up to 2,900 MHz; shared DDR5-5600-class memory, 128-bit; measured read 84 GB/s (rented) |
| Software | ROCm with HIP 7.16, CuPy 14.2.0 (cupy-rocm-7-0); native gfx1152 code, no `HSA_OVERRIDE_GFX_VERSION` |
| Kindi | rdna35-340 cb19bc1 (merged into v0.5.0-dev), `kfair` (one pass; read to the depth, then the mean of 3 x 32 written tokens) |
| llama.cpp | c96baf4, HIP for gfx1152, `llama-bench -ngl 99 -fa 1` (write `-n 64 -d D -r 5`, read `-p N -r 3`) |
| K/V cache | 27B: q8_0 for both engines; 2B, 4B, 9B: Kindi's single-card engine, llama.cpp f16 |
| Verify | `kindi verify` ALL PASS (token-exact) on all four models, no ties |

Tokens/s at 512 and 8K tokens of context. Bold: Kindi is faster.

| Model | Kindi write | llama.cpp write | Kindi read | llama.cpp read |
|---|---|---|---|---|
| 2B, 512 | **104.7** | 51.8 | 265 | 408 |
| 2B, 8K | **91.5** | 49.0 | 243 | 354 |
| 4B, 512 | **49.1** | 29.2 | 105 | 155 |
| 4B, 8K | **41.7** | 26.6 | 97 | 135 |
| 9B, 512 | **25.5** | 23.6 | 56 | 88 |
| 9B, 8K | **23.5** | 21.9 | 53 | 81 |
| 27B, 512 | **7.90** | 5.71 | **30.2** | 22.2 |
| 27B, 8K | **7.34** | 4.42 | **27.7** | 20.8 |

Kindi writes faster on every row. Write uses about 52-60% of the measured bandwidth with Kindi (llama.cpp: 26-56%).

## Kindi 0.5.0: AMD Radeon 8060S (Strix Halo, Ryzen AI MAX+ 395, RDNA 3.5, gfx1151)

| Item | Value |
|---|---|
| Graphics | Radeon 8060S, 40 compute units, up to 2,900 MHz; shared 256-bit LPDDR5X; measured read 240 GB/s (rented) |
| Software | ROCm with HIP 7.16, CuPy 14.2.0 (cupy-rocm-7-0); native gfx1151 code, no `HSA_OVERRIDE_GFX_VERSION` |
| Kindi | v0.5.0-dev 38fcef2, `kfair` (one pass; read to the depth, then the mean of 3 x 32 written tokens) |
| llama.cpp | c96baf4, HIP for gfx1151, `llama-bench -ngl 99 -fa 1` (write `-n 64 -d D -r 5`, read `-p N -r 3`) |
| K/V cache | 27B: q8_0 for both engines; 2B, 4B, 9B: Kindi's defaults, llama.cpp f16 |
| Engines | the plan engine for all four models |
| Verify | `kindi verify` ALL PASS on all four models |

Tokens/s at 512 and 8K tokens of context. Bold: Kindi is faster.

| Model | Kindi write | llama.cpp write | Kindi read | llama.cpp read |
|---|---|---|---|---|
| 2B, 512 | **260.0** | 236.1 | **3,985** | 3,153 |
| 2B, 8K | **234.5** | 210.9 | **3,488** | 3,045 |
| 4B, 512 | **143.2** | 128.6 | **1,952** | 1,642 |
| 4B, 8K | **122.0** | 110.7 | **1,674** | 1,541 |
| 9B, 512 | **88.8** | 86.0 | **1,174** | 746 |
| 9B, 8K | **80.4** | 77.6 | **1,076** | 725 |
| 27B, 512 | **30.59** | 28.31 | **375** | 301 |
| 27B, 8K | **28.42** | 24.34 | **333** | 292 |

Kindi writes and reads faster on every row. The AH-Nanbeige4.3-33B backend of Kindi needs CUDA, so this table has no
33B row.

## Kindi 0.5.0: NVIDIA GeForce RTX 3090 (sm_86, 24 GB)

| Item | Value |
|---|---|
| Card | one GeForce RTX 3090, 24 GB, 350 W, driver 615.71 (rented) |
| Kindi | v0.5.0-dev f6ca4df, `kfair` (one pass; read to the depth, then the mean of 3 x 32 written tokens) |
| Kindi settings | 2B, 4B, 9B, 27B: `KINDI_TC=1 KINDI_FD=1 KINDI_ATT_TC=1 KINDI_TC_TMAX=4` (plus `KINDI_TP=1` for the 2B, 4B and 9B), context 16,384; 33B: context 16,640 |
| llama.cpp | c96baf4, `llama-bench -ngl 99 -fa 1` (write `-n 64 -d D -r 5`, read `-p N -r 3`) |
| K/V cache | 27B: q8_0 for both engines; 33B: f16 for both engines; 2B, 4B, 9B: f16 |
| Verify | `kindi verify` passes on every model (4B: 3 exact, 1 llama.cpp tie; 33B: 8 exact, 3 ties) |

Tokens/s at 512 and 8K tokens of context. Bold: Kindi is faster.

| Model | Kindi write | llama.cpp write | Kindi read | llama.cpp read |
|---|---|---|---|---|
| 2B, 512 | 583.4 | 605.4 | 9,166 | 11,972 |
| 2B, 8K | 564.2 | 572.7 | 8,458 | 12,964 |
| 4B, 512 | 333.4 | 355.1 | 5,689 | 6,440 |
| 4B, 8K | 313.5 | 321.3 | 4,863 | 6,426 |
| 9B, 512 | **220.6** | 216.1 | 4,011 | 4,292 |
| 9B, 8K | **211.1** | 203.5 | 3,586 | 4,262 |
| 27B, 512 | 85.9 | 89.1 | **1,486** | 1,439 |
| 27B, 8K | 78.1 | 79.9 | 1,288 | 1,364 |
| AH-Nanbeige4.3-33B, 512 | **52.5** | 42.7 | 1,192 | 1,255 |
| AH-Nanbeige4.3-33B, 8K | **46.7** | 39.8 | 1,111 | 1,137 |

On the RTX 3090, Kindi writes the 33B 17% to 23% faster and the 9B 2% to 4% faster, and it reads the 27B faster at
512 tokens.

## Kindi 0.4.0: AH-Nanbeige4.3-33B IQ3_XXS and the 27B, initial agent test

In the initial test (the 65 tasks of kbench and 50 LiveCodeBench problems, one run each), the 33B solved 98 and the 27B solved 96.

## Kindi 0.4.0: Tesla P100 (sm_60)

Two rented machines gave the numbers. Do not mix them. Box A: 2x Tesla P100-SXM2-16GB (NVLink, 300 W), driver 535.230.02,
Kindi pascal-p100 b8c6e3f and p100-fp16-prompt 56d896d. Box B: 2x Tesla P100 16 GB (PCIe, 200 W). llama.cpp: Prism fork
c96baf4, CUDA 12.8, `-ngl 99 -fa 1`, 3 runs. The 27B uses q8_0 K/V in both engines. Details and raw rows:
`docs/dev/bench-0.4.0/p100.md`.

27B, box A. Read: a prompt of N tokens from position 0. Write: generation at depth N.

| Setup | Read 512 | Read 8K | Write 512 | Write 8K |
|---|---|---|---|---|
| Kindi, one card, fp16 prompt path on | 167.6 | 159.6 | 36.0 | 34.6 |
| Kindi, 2 cards, fp16 on | 146.8 | 143.7 | 39.7 | 39.0 |
| llama.cpp, one card | 146.4 | 142.7 | 18.7 | 17.7 |

27B, box B, write:

| Setup | 512 | 8K | 30K |
|---|---|---|---|
| Kindi, 2 cards | 40.4 | 39.5 | 34.9 |
| llama.cpp, 2 cards | 14.9 | 15.9 | 13.6 |
| Kindi, one card | 29.6 | 26.8 | |
| llama.cpp, one card | 14.4 | 13.3 | |

AH-Nanbeige4.3-33B IQ3_XXS, 2 cards (box B), write:

| Setup | 512 | 8K | 32K | 64K |
|---|---|---|---|---|
| Kindi | 21.0 | 19.6 | 15.7 | 12.3 |
| llama.cpp | 9.6 | 8.9 | | |

27B on two cards, full context (box B, PCIe, stock 200 W, tensor parallel): the full 262,144 tokens of context fit, with 8.6 GB free on each card.
Kindi reads a prompt of 262,016 tokens at 67.6 tokens/s (3,877 s) and writes 15.8 tokens/s at a depth of 262,016.

For the 2B on one card (box B), Kindi writes 166 / 162 / 149 tokens/s at 512 / 8K / 24K or deeper, and llama.cpp 129 /
122 / 120. Speculation is off on the P100.

## Kindi 0.4.0: AH-Nanbeige4.3-33B IQ3_XXS, one Quadro K6000

One K6000 12 GB (a VM, driver 470.256.02), f16 K/V, context 8,448. Kindi keeps 13 layers in host memory and computes 55%
of their FFN rows on the CPU (8 threads). llama.cpp ad1b88c with `-ngl 42` (the other 10 layers on the CPU). Write is
the speed of 32 tokens after the prompt.

| Prompt | Kindi write | llama.cpp write |
|---|---|---|
| 480 tokens | 3.85 | 2.19 |
| 2,016 tokens | 3.55 | 1.96 |
| 8,160 tokens | 3.56 | 1.52 |

At a context of 1,024 (6 host layers), Kindi writes 4.4 to 5.0 tokens/s (two runs), and llama.cpp (4 layers on the CPU) writes 2.5 to 2.7.
`kindi verify` passes: 9 of 11 prompts token-exact, 2 end at a llama.cpp tie, 0 fail.

## Kindi 0.4.0: AH-Nanbeige4.3-33B IQ3_XXS, two RTX 3080 cards

| Item | Value |
|---|---|
| Cards | 2x GeForce RTX 3080 10 GB, 190 W limit, PCIe gen1/2 x1 |
| Model | AH-Nanbeige4.3-33B-IQ3_XXS.gguf (13.1 GB, sha256 9d4f18e8...a6c239), 52 layers |
| Layer split | Kindi 26 + 26; llama.cpp `-sm layer` |
| K/V cache | f16 for both engines |
| llama.cpp | build 11382 (11fe02151), `llama-bench -ngl 99 -fa 1 -sm layer` |
| Verify | `kindi verify` PASS: 8 of 11 prompts token-exact, 3 end at a llama.cpp tie, 0 fail |

Write speed (Kindi: 64 tokens at each depth; llama.cpp: tg128 at each depth, 3 runs):

| Context depth | Kindi | llama.cpp |
|---|---|---|
| 0 | 45.5 | 32.3 |
| 2,048 | 44.3 | 35.3 |
| 8,192 | 40.7 | 33.3 |
| 16,384 | 37.0 | 31.0 |

Read speed (a prompt of N tokens from position 0):

| Prompt | Kindi | llama.cpp |
|---|---|---|
| 512 | 1,047 | 447 to 539 (it changed from run to run) |
| 2,048 | 1,267 | 1,120 |
| 8,192 | 1,259 | 1,238 |
| 16,384 | 1,209 | 1,190 |

Prompt lookup (context 8,192, `kindi serve`, 400 tokens out): a file copied with changes 91.6 tokens/s against 45.1
without it (2.03 times); a tool call 1.17 times; prose the same.

The 33B also runs on a single RTX 3090 (24 GB). The "Kindi 0.5.0: NVIDIA GeForce RTX 3090" section has its numbers.

Through `kindi serve`: 45.4 tokens/s greedy, 44.2 sampled (temperature 0.7, top-k 20, top-p 0.8). Build: branch
nanbeige-iq3 (engine fde34d6, server 1ca5066).

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
Kindi 0.3.0-dev b340cbe, gpu-unlock loaded, 2026-10-24. On the GTX 1080, Kindi reads longer prompts faster.

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

Seconds for each request (prompt and reply), `examples/prompt_cache_bench.py`. Kindi and llama.cpp (b11804), each with
its cache on. The replies with Kindi's cache on were the same as with the cache off in all 12 requests of each model.

| Job, request | Nemotron, Kindi | Nemotron, llama.cpp | 27B, Kindi |
|---|---:|---:|---:|
| Agent, turn 2 | **0.67** | 1.29 | 2.52 |
| Agent, turn 3 | **0.86** | 1.30 | 2.93 |
| Regenerate | **0.43** | 4.93 | 1.85 |
| Document, question 2 | **0.70** | 3.16 | 14.53 |
| Document, question 3 | **0.42** | 3.10 | 11.46 |
| Two agents, A turn 2 | **0.58** | 1.40 | 2.61 |
| Two agents, B turn 2 | **0.53** | 1.44 | 2.79 |

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
Kindi's Nemotron support is experimental, and we are still optimizing it.

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

## GeForce GTX 1650 SUPER, 4 GB · Ternary-Bonsai-2

2026-11-03. Kindi 0.3.0, local Docker image built from repository commit `3762126`, CUDA 10.2 / CuPy 9.6, NVIDIA
driver 565.57.01. The card is Turing (sm_75); CUDA reported 3,716 MiB total memory. The PTQ1_0 GGUFs were used
(2B sha256 `36061ad4…`, 4B `56711263…`, 9B `3561df7a…`). Each model passed `kindi verify` (four prompts,
token-exact). The exact head was used; no fast-head clusters were built. The host desktop remained active, so the
context that fits depends on memory available to Kindi.

`kindi bench` warms each operation once. Its prompt-read figure is based on 20 one-token feeds near the start of the
context; each write figure is the average of 10 decode steps at the indicated position. These are built-in benchmark
measurements, not the multi-token prompt-read method or cross-engine comparison used in the tables above. Read and
write figures are tokens per second:

| Model | Requested / fitted context | Read | Write @100 | @2,000 | @8,000 | @16,000 | @24,000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2B | 24,576 / 24,576 | 143.1 | 119.7 | 117.9 | 115.1 | 115.0 | 105.9 |
| 4B | 24,576 / 24,576 | 76.5 | 67.6 | 65.4 | 59.7 | 53.4 | 47.3 |
| 9B | 24,576 / 19,200 | 61.6 | 54.2 | 52.6 | 49.7 | 45.7 | — |

The 9B run did not reach position 24,000 because the engine fitted its requested context to 19,200. A separate
context-fit run with the default `KINDI_MEM_RESERVE_MB=96` requested 131,072 tokens; the engine used 131,072 for the
2B (all that was requested), 58,112 for the 4B and 19,200 for the 9B. Contexts above 131,072 were not tested for the
2B. The fitted values depend on free GPU memory and are not guarantees for other systems. At the time of this test,
`kindi models` reported the 9B as too large on the card's 3,716 MiB, although it loaded and passed verification with
the reduced context.

To reproduce a context-fit run, convert each PTQ1_0 GGUF into its own pack, then run `bench` with the requested
context. For example, with the 4B pack at `/data/pack4b` in the mounted data directory:

```sh
docker run --rm --gpus all -u "$(id -u):$(id -g)" \
  -v /path/to/kindi-data:/data -e KINDI_PACK=/data/pack4b \
  -e KINDI_STATE_DIR=/data/state -e KINDI_CTX=131072 \
  kindi:0.3.0 bench
```

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
