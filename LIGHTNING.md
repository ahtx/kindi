# Nemotron Lightning on Quadro K6000

This extension runs the pinned Nota Nemotron 3.5 Lightning 30B-A3B ternary checkpoint with native CUDA kernels on Kepler sm_35. Model computation stays on the selected GPU. Tokenization, request-local sampling and the bounded prefix cache run on the CPU.

The accepted graph has 23 Mamba-2, 23 MoE and 6 attention blocks. Its experts use ReLU squared, sigmoid routing and a shared expert. Attention uses NoPE. The misleading positional metadata in the community GGUF is not interpreted as an instruction to apply Gemma or Qwen RoPE.

## Runtime and model

The accepted K6000 runtime uses (see BUILD.md for the source recipe): CUDA 10.2, CuPy 9.6 and Python 3.6.9 on driver 470.256.02. The image extension installs no packages and contains no model weights. RTX 4080 is used for reference validation, not as a required production dependency.

| Artifact | Identity |
| --- | --- |
| Kindi base source | `6ac903e5cb00a6d55b6cb60f6ad7445cafe93051`, version 0.3.0 |
| GGUF repository | `boro64/Ternary-Nemotron-3.5-Lightning-30B-A3B-GGUF` |
| GGUF revision | `570610701b08e16ffcfaa899a0bdec94f0cda819` |
| File size | 9,472,656,320 bytes |
| SHA-256 | `38854de649212dc2761006b75afbc41412e7531f763c3ff965bd31f969bb14f3` |
| Nota source revision | `85889ce1c6c89bc6c999e28cf01c2dce0d0c4ef4` |

`kindi convert EXISTING.gguf --out PACK` writes a small manifest pointing to the existing GGUF. It does not unpack or duplicate the weights. `kindi verify` checks three complete vocabulary heads against the accepted RTX reference and checks bit-exact snapshot restoration. `kindi chat`, `bench`, `info` and `serve` dispatch to the native adapter when the manifest selects `kindi_lightning`.

## Production profile

| Setting | Value |
| --- | --- |
| Context allocation | 32,768 tokens, prompt plus requested output |
| Weight residency | 9,464,761,344 bytes, about 8.815 GiB |
| Recurrent and convolution state | 49,930,240 bytes, FP32 |
| KV | FP32 default: 384 MiB at 32K; optional FP16 192 MiB / Q8_0 102 MiB |
| Workspace reserve | 128 MiB split attention + 512 MiB general reserve; compressed auto reads also reserve 64 MiB for one expanded layer |
| Head | Exact full vocabulary |
| Prefill | tile16 dense/experts, chunk 128; actual last-chunk length, no padded state advancement |
| GPU concurrency | One serialized worker |
| Host prefix cache | 512 MiB, at most eight entries |

The sequential diagnostic path is available with `KINDI_LIGHTNING_PREFILL_CHUNK=0`. The V3 container selects `KINDI_LIGHTNING_DENSE_PREFILL=tile16`, `KINDI_LIGHTNING_EXPERT_PREFILL=tile16` and `KINDI_LIGHTNING_ATTENTION=split`. Direct library use retains the old defaults unless these variables are set. Storage is selected with `lightning_kv = fp32|fp16|q8_0` or the global CLI flag `--lightning-kv`. Contracts and experimental alternatives are described in `LIGHTNING_ADAPTERS.md`.

Mamba chunk execution keeps the recurrence in temporal order inside the CUDA kernel. It does not materialize a state for every token or use a different parallel SSD reduction. The tiled prefill implementation reuses unpacked weights for eight inputs after GPU expert dispatch while retaining the sequential per-token accumulation order. Split attention computes key tiles independently and merges them in their original order. All arithmetic and Mamba state remain FP32; compressed KV changes only storage.

## API and state

Endpoints are `/health`, `/v1/models`, `/v1/chat/completions` and `/v1/completions`. Requests support text, UTF-8 streaming, exact stop substrings, temperature/top-p/top-k, seed, usage, thinking and the checkpoint's function-call template. `tool_choice` accepts `auto` and `none`. Unsupported sampling/response options are rejected before streaming starts. Multimodal input and continuous batching are outside this adapter.

Every request creates its own RNG from the supplied seed. A prefix cache hit never inherits another request's sampling state. Snapshots contain recurrent state, convolution history, KV, the exact consumed token list and valid logits. They are matched by model SHA, adapter/tokenizer source, kernel source, numeric profile, CUDA runtime and compute capability. Snapshots remain in RAM and are discarded on restart.

Cancellation synchronizes at bounded chunk boundaries before checking the disconnect flag. This prevents an entire long prompt from being queued onto the GPU after a client has disconnected. A recoverable failed request resets its active state; an unrecoverable CUDA failure makes the engine reject new work and health return 503 until this backend process is restarted. Docker unhealthy status alone does not trigger a restart.

## Validation and scope

The evidence directory beside this source contains independent NumPy operator checks, original-source tensor checks, 1,307 tokenizer comparisons, eight chat-template comparisons, complete-model PyTorch comparisons and K6000 acceptance results. The strict sequential and chunked paths agree bit for bit for final Mamba state, convolution history, full KV and logits at 1,024, 4,096 and 8,192 tokens on both RTX 4080 and K6000.

The v2 evidence adds full 32K document retrieval for FP32/FP16/Q8 on both GPUs, with exact comparison of a 512-token sequential/batched tail from the same prefix. Separate full 32768-token sequential/batched FP32 comparison on K6000 was bit exact. Compressed reference alignment and teacher-forced calibration are documented in `LIGHTNING_ADAPTERS.md`; lossy storage is not claimed to preserve all FP32 outputs. An allocation or synthetic boundary check at a larger capacity is not a semantic long-context benchmark. The user deferred full 64K/128K runs. MTP, speculative decoding, multi-GPU operation and additional real MoE adapters remain separate work. The GGUF's advertised million-token context is not a K6000 result.

Legacy cuda-memcheck is excluded on this host: kernel oops in `nvidia_uvm` was observed while it ran, despite the tool reporting zero memory errors. The old boot also contained Xid 32/69. The user rebooted manually; the original cause is not conclusively established. Current-boot kernel logs are part of the acceptance evidence. Drivers, BIOS, clocks and protected Bonsai/Bolt containers are unchanged.

Performance must be read together with prompt length, warmed repetitions and actual clocks. K6000's automatic power cap can lower its clock under sustained prefill; the implementation does not change clocks, power limits or driver settings. Benchmark scripts save token counts, EOS policy, TTFT, prefill/decode rates, latency percentiles, memory and NVIDIA telemetry.

## Licensing and origins

The extension retains Kindi's `LICENSE` and `NOTICE`. The model's OpenMDW 1.1 license is included in `licenses/Nemotron-OpenMDW-1.1.txt`; weights are mounted separately.

Architecture and layout were cross-checked against the pinned Nota checkpoint, Apple's MLX-LM Nemotron/SSM sources and llama.cpp's Nemotron implementation. The runtime kernels here are native CUDA implementations and do not require MLX or llama.cpp:

- [Nota checkpoint](https://huggingface.co/nota-ai/Nemotron-3.5-Lightning-30B-A3B-Nota-Ternary-MLX-1014/tree/85889ce1c6c89bc6c999e28cf01c2dce0d0c4ef4)
- [GGUF checkpoint and model license](https://huggingface.co/boro64/Ternary-Nemotron-3.5-Lightning-30B-A3B-GGUF/tree/570610701b08e16ffcfaa899a0bdec94f0cda819)
- [NVIDIA CUDA 10.2 Kepler tuning guide](https://docs.nvidia.com/cuda/archive/10.2/kepler-tuning-guide/index.html)
- MLX-LM Nemotron reference `28e9ccd9cb52d1d4a6f674b7914bae2de49d9aad`; SSM reference `1e511c1f72a93f47e3c6d73254f1b5da2ae92d2d`.
- llama.cpp Nemotron reference `68d20b884c3878217de79eade0a249efde96ab38`.

The surrounding implementation workspace preserves exact source URLs, audit scripts, reference excerpts and hashes in `reference/` and `evidence/`.

## Publication V3 update — 2026-10-30

See README.md, BUILD.md, BENCHMARKS-LIGHTNING.md and LICENSING.md. References above to
evidence folders describe the internal development record; those private folders
are not part of this release. V3 tile16 acceptance supersedes the older V2 profile.
