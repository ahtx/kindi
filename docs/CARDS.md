# How to run Kindi on each card

The installer ([INSTALL.md](INSTALL.md)) uses these settings. It selects them from the cards that it finds.

Blocking sync (`KINDI_BLOCKING_SYNC`, on by default) lets the host thread sleep while it waits for the card. It
costs 2-3% of the decode speed on the GTX 1080. With one server on its own card, set `KINDI_BLOCKING_SYNC=0`
(the installer does this). The RTX 3080 profile keeps it on: there, many servers share few CPU cores.

This page gives the settings for each tested card. It also gives the speeds that we measured with these settings.
All the cards use the same Docker image (CUDA 10.2, CuPy 9.6). The `kindi` folder of this repository is the code.

To make a pack, use `kindi convert` (see the README). Put the pack in `$HOME/.kindi/<pack>`. In the container, it
is at `/data/<pack>`. The examples serve on `127.0.0.1:8093`.

```
docker run -d --name kindi --restart unless-stopped --gpus all -u $(id -u):$(id -g) \
  -p 127.0.0.1:8093:8093 -v $HOME/.kindi:/data -e HOME=/tmp \
  -e KINDI_PACK=/data/pack -e KINDI_STATE_DIR=/data/state ... kindi:0.1.0
```

Replace `...` with the variables in the section for your card. To use one card of two, set `--gpus device=N`.

## Quadro K420, 1 GB (Kepler, sm_30)

| item | value |
|---|---|
| model | Ternary-Bonsai-2-2B |
| variables | `KINDI_CTX=24576` (the default) |
| generation | 22.9 tokens/s (fast head, 2,000 tokens of context); 20.8 tokens/s (exact head, 512 tokens) |
| prompt reading | 31.6 tokens/s at 3K tokens, 28.1 at 12K (one token at a time: 25.3 and 23.4; dev build of 10-21) |

- The prompt path uses chunks of 128 tokens on this card. Then the context stays at 24,576 tokens.
- To read prompts one token at a time, set `KINDI_CHUNK=0`.

## Quadro K420, 2 GB (Kepler, sm_30)

| item | value |
|---|---|
| model | Ternary-Bonsai-2-4B (the installer's choice), or the 2B |
| variables | `KINDI_CTX=23808` for the 4B (the context that fits), `KINDI_CTX=65536` for the 2B |
| generation | 2B: 21.2 tokens/s (exact head), 24.8 (fast head); 4B: 9.8 tokens/s |

## Quadro K1200, 4 GB (Maxwell, sm_50)

| item | value |
|---|---|
| models | 2B (`/data/pack`), 4B (`/data/pack4b`) or 9B (`/data/pack9b`) |
| variables | `KINDI_CTX=24576` |
| generation (exact / fast head) | 2B 71.6 / 88.3 tokens/s, 4B 35.0 / 42.2, 9B 20.7 / 24.9 |
| prompt reading at 3K tokens | 2B 186.6 tokens/s, 4B 74.5, 9B 40.0 |

- The card in an x16 slot is faster than the card in an x4 slot.

## Two Quadro K1200 cards (tensor parallel)

| item | value |
|---|---|
| model | Ternary-Bonsai-2-27B (`/data/pack27b`) |
| variables | `KINDI_PACK=/data/pack27b`, `KINDI_CTX=24576`, `KINDI_HEAD=exact`, `KINDI_CHECKPOINT_BUDGET_MB=4096`, `KINDI_REPEAT_STOP=1` |
| docker | `--gpus all` (the engine uses both cards), `--memory 12g` |
| generation | 12.6 tokens/s |
| prompt reading | 20.9 tokens/s at 512 tokens, 18.4 at 22K (dev code of 10-21, the chunk-path copy) |

- The K/V cache is q8_0. The output matches llama.cpp with `-ctk q8_0 -ctv q8_0`.
- Set the host to text mode. A desktop session turns on the CUDA watchdog, and the watchdog stops the exchange
  kernels.

## Quadro K6000, 12 GB (Kepler, sm_35)

| item | value |
|---|---|
| model | Ternary-Bonsai-2-27B |
| variables | `KINDI_PACK=/data/pack27b`, `KINDI_CARDS=1`, `KINDI_CTX=65536`, `KINDI_HEAD=exact`, `KINDI_REPEAT_STOP=1`, `KINDI_MAX_TOKENS=65536` |
| generation | 20.4 tokens/s at 512 tokens, 16.4 at 8K, 13.1 at 24K |
| prompt reading | 36.2 tokens/s at 512 tokens, 30.2 at 8K, 26.2 at 24K |

- The card needs the NVIDIA 470 driver. The 470 driver does not report the compute capability, so the installer
  reads it from the card name.
- With all of the 12 GB, the engine fits a context of 156,672 tokens. The installer sets 65,536.
- Nemotron also runs on this card (experimental): 20.1 tokens/s of generation at 512 tokens. The installer does not
  offer it on Kepler cards.

## GeForce GTX 1080, 8 GB (Pascal, sm_61)

| item | value |
|---|---|
| model | Ternary-Bonsai-2-27B |
| variables | `KINDI_PACK=/data/pack27b`, `KINDI_CARDS=1`, `KINDI_CTX=65536`, `KINDI_HEAD=exact`, `KINDI_REPEAT_STOP=1`, `KINDI_MAX_TOKENS=65536` |
| generation | 28.4 tokens/s at a short context, 16.0 at 60K |
| prompt reading | 249 tokens/s at 512 tokens, 237 at 3K, 213 at 12K, 188 at 22K |

- The first start writes the 5-trit weights into the pack (`g5n_*.bin`, 5.6 GB). This takes about 12 minutes.
  Later starts take about 30 seconds.
- The prompt products use int8 activations on this card. For fp32, set `KINDI_CHUNK_Q8=0` (110 tokens/s at 512
  tokens).
- On a host with more than one GPU, set `CUDA_VISIBLE_DEVICES` to the GTX 1080.
- On Windows (WSL2 or native), the WDDM driver keeps part of the card memory. Kindi fits the context to the free
  memory, so an 8 GB card gets a smaller context. A GTX 1070 with no monitor connected got 43,008 tokens for the 27B.
  `KINDI_MEM_RESERVE_MB` sets the memory that the fit keeps free. A smaller value gives more context, but the
  risk of an out-of-memory error is higher.

## GeForce RTX 3080, 10 GB (Ampere, sm_86)

| item | value |
|---|---|
| model | Ternary-Bonsai-2-27B |
| variables | `KINDI_PACK=/data/pack27b`, `KINDI_CARDS=1`, `KINDI_CTX=131072`, `KINDI_MEM_RESERVE_MB=128`, `KINDI_HEAD=exact`, `KINDI_CHECKPOINT_BUDGET_MB=1024`, `KINDI_PARK_MAX=1`, `KINDI_REPEAT_STOP=1` |
| docker | `--memory 14g` (the first load needs it) |
| context | 110,592 tokens (the fit to the free memory) |
| generation | 47-49 tokens/s at a short context, 29 at 64K |
| prompt reading | 491 tokens/s at 512 tokens, 381 at 22K |

These figures are from kindi-rel, 10-21, not the 0.2.0 build.

Tensor cores and speculative decoding (sm_80 and later). This is the profile of the RTX 3080 cluster servers, and
the installer uses it:

| variable | value | |
|---|---|---|
| `KINDI_TC` | `1` | int8 tensor-core products and CUDA graphs for the decode |
| `KINDI_FD` | `1` | split flash-decode attention: the K/V is read once per query, in chunks of 512 positions |
| `KINDI_ATT_TC` | `1` | the attention of the prompt chunks on the tensor cores |
| `KINDI_SPEC` | `draft2b` | speculative decoding with the 2B as the draft model (also `ngram` or `both`). Off by default |
| `KINDI_SPEC_K` | `3` | the draft tokens per verify pass (default 4) |
| `KINDI_SPEC_PACK` | the 2B pack | for `draft2b` (for example `/data/pack2b`) |
| `KINDI_SPEC_WINDOW` | `8192` | the context of the draft model |
| `KINDI_TC_TMAX` | `4` | the most tokens in one verify pass |
| `KINDI_CKPT_VRAM` | `1` | a checkpoint is a copy in card memory; its copy to the host waits until the server is idle |
| `KINDI_MEM_RESERVE_MB` | `160` | the memory that the CUDA graphs need after the context fit |
| `KINDI_DENSE_SPACING` | `2048` | the spacing of the checkpoints in the opening of a conversation |
| `KINDI_CHECKPOINT_BUDGET_MB` | `1280` | the host memory for checkpoints and parked conversations (`1024` on the agent servers) |
| `KINDI_PARK_MAX` | `1` | parked conversations (`0` on the agent servers: one conversation each) |
| `KINDI_PREFIX_MAX` | `2` | saved openings on disk |

With this profile: context 84,992 tokens; at 190 W on agent prompts 99.9 tokens/s greedy and 91.3 sampled; prompt
reading 1,093 tokens/s at 512 to 855 at 22K.

- With `KINDI_TC=1`, the server reads prompts in passes of up to `KINDI_TC_TMAX` tokens.
- Speculative greedy output is the same as the output without speculation.
- The cluster runs seven servers on one 4-core host with `--memory 6g`. One server on a computer of its own can use
  the default budgets.

## Tesla P100, 16 GB (Pascal, sm_60)

The Tesla P100 is a supported card in Kindi 0.4.0. A used card costs $78 to $160. It has 16 GB of HBM2 memory with a
bandwidth of 732 GB/s. Photo: GP100 chip of a Tesla P100, by Fritzchens Fritz, Wikimedia Commons (CC0).

| item | value |
|---|---|
| models | 2B, 4B, 9B, 27B; the 33B on two cards (`kindi use nanbeige-33b`) |
| variables | `KINDI_CARDS=1`, `KINDI_REPEAT_STOP=1`, `KINDI_MAX_TOKENS=65536` (the installer sets them) |
| docker | `--memory 14g` |
| generation, 27B, one card (SXM2, 300 W) | 36.0 tokens/s at 512 tokens, 34.6 at 8K (llama.cpp: 18.7 and 17.7) |
| generation, 27B, two cards | 39.7 at 512 tokens, 39.0 at 8K |
| prompt reading, 27B, one card | 167.6 tokens/s at 512 tokens, 159.6 at 8K (llama.cpp: 146.4 and 142.7) |
| generation, 33B, two cards (PCIe, 200 W) | 21.0 tokens/s at 512 tokens, 19.6 at 8K, 15.7 at 32K, 12.3 at 64K (llama.cpp: 9.6 at 512, 8.9 at 8K) |

- On the same P100, Kindi writes the 27B 1.9 to 2.7 times as fast as llama.cpp (one card, SXM2 box: 1.9; PCIe box: 2.0
  on one card and 2.5 to 2.7 on two cards).
- The fp16 prompt path is on by default for sm_60 (`KINDI_P100_FP16_PROMPT=0` turns it off). With it, Kindi reads a
  prompt 14% (512 tokens) and 12% (8K) faster than llama.cpp on the same card. Without it, the read speed is below
  llama.cpp.
- Two P100 cards (PCIe, 200 W) hold the full 262,144-token context of the 27B, with 8.6 GB free on each card. Kindi reads a prompt of 262,016 tokens at 67.6 tokens/s and writes 15.8 tokens/s at that depth. Two cards also run the 33B.
- Speculative decoding is off on the P100. On this card, a verify pass over several tokens costs about as much as the
  same number of single steps, so a draft cannot save time.
- `kindi verify` passes on one P100 for the 2B, 4B and 27B (token-exact), and on two P100 cards for the 33B.
- The numbers come from two rented machines (SXM2 with NVLink, and PCIe). `docs/dev/bench-0.4.0/p100.md` has the
  conditions of each row.

## GeForce RTX 20, 30, 40 and 50 series

Kindi 0.4.0 supports all four series. The tensor-core path and speculative decoding use cards of compute capability
8.0 and later (RTX 30, 40 and 50). The RTX 20 series (Turing, sm_75) uses the default engine.

| card | what we measured |
|---|---|
| RTX 3080 (190 W) | 27B: 62.3 tokens/s at 512 tokens. With the 2B draft model: 91.1 (1.46 times) at 512 tokens, 103.2 (1.68 times) at 8K with k=2, and 111.5 (1.82 times) at 8K with k=3. The same output as without the draft model. Against llama.cpp: see [BENCHMARKS.md](BENCHMARKS.md) |
| RTX 3060 12 GB | 27B: 41.1 tokens/s at 512 tokens; with the 2B draft model 60.6 (1.48 times), 73.5 (1.82 times) at 8K, 66.7 (1.81 times) at 24K |
| other RTX 20, 40 and 50 cards | supported; `kindi verify` passes. We give no speed numbers |

## AH-Nanbeige4.3-33B IQ3_XXS

| item | value |
|---|---|
| model | AH-Nanbeige4.3-33B IQ3_XXS (our IQ3_XXS quantization of Nanbeige4.3-33B, made with bartowski's imatrix): AH-Nanbeige4.3-33B-IQ3_XXS.gguf (`kindi use nanbeige-33b`); model card: https://huggingface.co/ahtx/AH-Nanbeige4.3-33B-IQ3_XXS-GGUF |
| cards | one card with 16 GB or more (for example an RTX 3090), or several cards with about 17 GB in total (measured: 2x RTX 3080, 2x Tesla P100); a Quadro K6000 with layers in host memory |
| variables | `KINDI_CTX=16384`, `KINDI_REPEAT_STOP=1`, `CUDA_VISIBLE_DEVICES=<the two cards>` |
| docker | `--memory 10g` |
| generation | 2x RTX 3080 (190 W): 45.5 tokens/s at a short context, 44.3 at 2K, 40.7 at 8K, 37.0 at 16K |
| generation | 2x Tesla P100: 21.0 tokens/s at 512 tokens, 19.6 at 8K. Quadro K6000: 3.56 tokens/s at 8K |
| prompt reading | 1,047 tokens/s at 512 tokens, 1,209 at 16K (2x RTX 3080) |

It is a strong model for coding work, and it can do even better on a slightly faster GPU.

The backend puts the layers in order on the cards. Only the hidden state goes between the cards, so PCIe x1 is
sufficient. The GGUF file has no MTP head. Prompt lookup is on by default: when the model copies or edits text, it
writes about 2 times as fast (91.6 against 45.1 tokens/s on two RTX 3080 cards at 8K).

On one Quadro K6000 (12 GB), 13 layers stay in host memory (at a context of 8,448 tokens). The card copies them in
while it computes, and the CPU computes 55% of the FFN rows of each host layer. Kindi writes 3.9 tokens/s at 480
tokens and 3.56 at 8,160 tokens, and llama.cpp (the same card, 10 layers on the CPU) writes 2.19 and 1.52.

An agent tool such as opencode can use the 27B and the 33B together, because both run on Kindi's OpenAI-compatible
server. Use the 27B for fast steps and the 33B for harder coding steps.

## AMD Radeon 780M (RDNA3, gfx1103)

The Radeon 780M is the graphics unit of the Ryzen 7 7840HS and Ryzen 9 7940HS. It has no memory of its own and uses the
memory of the computer (32 GB or more for the 27B). Kindi runs it through ROCm, in the `kindi-rocm` image, with the kernel
option `amdgpu.cwsr_enable=0`. [AMD.md](AMD.md) gives the settings.

| item | value |
|---|---|
| model | Ternary-Bonsai-2-27B, Nemotron or the 2B |
| generation | 10.6 tokens/s at 512 tokens (27B) |
| prompt reading | 103.7 tokens/s at 512 tokens (27B) |

## CPU (x86-64 with AVX2)

| item | value |
|---|---|
| models | 2B, 4B or 9B |
| variables | `KINDI_DEVICE=cpu`, `OMP_NUM_THREADS=<cores of one socket>`, `OMP_PROC_BIND=close`, `OMP_PLACES=cores` |

- The CPU engine needs gcc with OpenMP. It builds `libkcpu.so` at the first start (`KINDI_CPU_CACHE`).
- The output is token-exact with llama.cpp.
