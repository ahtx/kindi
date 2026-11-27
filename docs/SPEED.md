# Kindi 0.8 speed tests

Read means new prompt tokens per second. Write means generated tokens per second. Tests ran one request at a time, with the same model file, prompt and context on each row. Every "after" number uses the 0.8 defaults. [Turbo](TURBO.md) lists the faster settings that give slightly different output; its numbers are not in this table. The "before" column is the same build with the new setting turned off, or the 0.7.2 release where marked.

| Model and hardware | Before | Prompt tokens | Read before and after | Write before and after |
|---|---|---:|---:|---:|
| Bonsai 2B PTQ1_0, GTX 1080 8 GB | fusion off | 2,048 | 1,081.09 to 1,080.78 | 56.09 to 71.91 |
| Bonsai 4B PTQ1_0, GTX 1080 8 GB | fusion off | 2,048 | 435.48 to 434.75 | 43.72 to 53.59 |
| Bonsai 9B PTQ1_0, GTX 1080 8 GB | fusion off | 2,048 | 236.57 to 236.18 | 34.98 to 41.70 |
| Bonsai 2B PTQ1_0, RTX 3080 10 GB | fusion off | 2,048 | 3,357.40 to 3,375.32 | 220.77 to 241.29 |
| Bonsai 2B PTQ1_0, Quadro K1200 4 GB | fusion off | 2,048 | 191.13 to 191.00 | 79.52 to 82.55 |
| Bonsai 2.5 27B PTQ1_0, RTX 3090 24 GB | no MTP head | 4,096 | Not measured | 85.1 to 151.9 |
| Qwen3.8-Flash-Next GSQ-RCO IQ3_S, RTX 3060 12 GB, HTTP | 0.8 without the prompt profile | 14,064 | 361 to 870 | 30.7 to 29.5 |
| Same | Same | 4,601 | 345 to 454 | 36.4 to 34.5 |
| DeepSeek V4.1 Flash 1017 UD-Q4_K_XL, RTX 3060 12 GB + M4000 8 GB, T=0.7 | full top-p sort | 13,779 | 91.7 to 92.3 | 12.84 to 16.49 |
| Nemotron 3.5 Lightning Q2_0, GTX 1080 8 GB | 512-token chunks | 2,048 | 647.21 to 758.19 | 86.77 to 85.99 |
| Qwen3.8-Flash-Next GSQ-RCO IQ3_S, dual Xeon E5-2673 v4, CPU only, 40 threads, HTTP | ik_llama.cpp, same file | 5,870 | 78-81 to 82-86 | Greedy 9.5-10.1 to 14.5-15.1; T=0.7 9.8-10.2 to 14.4-14.8 |
| Inkling 2 Mini IQ2_M, GTX 1080 8 GB | 0.7.2 | 512 | 114.56 to 114.19 | 20.92 to 21.91 |

All Bonsai rows are greedy, with a 4,096-token context and 96 timed decode steps. Each arm ran three times per process, with two processes per arm in off/on/on/off order. The baseline is the same build with `KINDI_DENSE_FUSE=0`. The GTX 1080 host has a Xeon W3503. The Bonsai 2.5 test used an 8,192-token context, greedy decoding, three repeats per run and two runs per mode, with the MTP head imported by `python -m kindi.bonsai25_mtp`.

The Flash-Next rows used dual Xeon E5-2673 v4 CPUs with 755 GiB of RAM, one RTX 3060, the installer's engine settings with page-locked experts, 17,408 context, two MTP drafts, greedy decoding and 128 output tokens. Each rate is the mean of two server runs per arm, in control/new/new/control order. At T=0.7 the 4,601-token prompt read at 504 tokens/s and wrote 33.4 tokens/s, against 346 and 33.1.

The CPU-only Flash-Next row used 8,192 context, the same GGUF file and the same HTTP harness for both engines; Kindi drafts with the MTP head (its greedy tokens match plain decoding) and passes `kindi verify` with 173 of 174 tokens.

The DeepSeek row used dual Xeon E5-2673 v4 CPUs, 81,920 context, DSpark, T=0.7, top-p 0.95 and a fixed seed, with 128 output tokens. The seeded reply was the same in both arms. Greedy replies were unchanged.

The Lightning row used 4,096 context, 64 timed decode steps and three runs in each of two arms. The Inkling run used a 2,304-token context.

## Composite

For every row where Kindi and another engine were measured on the same hardware, with the same model file and the same context, take the ratio of Kindi's speed to the other engine's speed. Read and write rows count separately, and each row counts once. The composite is the geometric mean of all the ratios. Every measured row is included. Older rows keep the Kindi version that was measured, so later Kindi gains are not credited.

Across 262 rows against llama.cpp, ik_llama.cpp and Strata, on 29 hardware setups and 8 models, the composite is 1.55. Kindi is faster in 218 rows. Four more configurations finished only on Kindi, and are not in the ratio. The rows are in [BENCHMARKS.md](BENCHMARKS.md) and in the tables above.
