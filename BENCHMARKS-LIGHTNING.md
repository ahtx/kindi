# Lightning V3: historical K6000 validation and measurements

Measured 2026-10-30 on a Quadro K6000 (12 GB), i7-5775C host, NVIDIA driver
470.256.02, CUDA 10.2 / CuPy 9.6 / Python 3.6.9, GPU clock 993 MHz. These are measurements
of accepted development V3, not a fresh run of the publication container.

| Input tokens | Prefill tok/s | Decode tok/s | Read seconds |
|---:|---:|---:|---:|
| 1,000 | 81.5201 | 27.6823 | 12.2669 |
| 4,000 | 77.4174 | 26.5961 | 51.6680 |
| 8,000 | 72.6253 | 25.3085 | 110.1545 |
| 16,000 | 64.1978 | 23.0744 | 249.2297 |

Medians of three independent reset/full-prefill runs per size after full warmup;
FP32 KV, context 32768, 256 decode steps, no prefix-cache reuse. Input counts include
the chat template. EOS is ignored only for the fixed-length benchmark. Synchronized
engine wall time excludes model loading, warmup, HTTP/network, snapshots, logit
copies and correctness checks; rare prefill progress messages are included.
All repeats matched state/logit hashes and token IDs. Peak observed temperature
was 68 C. Validation buffers inflate measured VRAM, so those figures are not presented
as production API memory consumption.

A separate controlled tile8/tile16/tile16/tile8 ABBA comparison with 32 outputs
measured prefill gains of 17.31% at 1024 input and 16.42% at 4096 input. Decode did not
materially improve. This is V2 versus V3 on the same model/hardware, not a comparison
with Bonsai or Nemotron Bolt. It does not support model-quality claims.

Historical acceptance: 57 operator cases on K6000 and RTX 4080; independent reference
checks; complete states/logits/IDs; independent full tile8/tile16 FP32 prefills at
32736+32 tokens; three-depth 32K retrieval with 3/3 facts for FP32/FP16/Q8; API 100-request
stress, cancellation, cache and restart. The compressed-profile retrieval compares
its last 512-token tail from a shared prefix; it is not two full independent prefills.
FP16/Q8 storage is lossy and is not claimed bit-identical to FP32.

Full-model acceptance stops at 32768 total tokens. 64K/128K, speculative/MTP and
arbitrary other MoE/Gemma checkpoints are not accepted features. cuBLAS profiles
that change reduction order remain experimental. Legacy cuda-memcheck was excluded
after a historical nvidia_uvm kernel oops; this is not a claim that the incident's
cause was conclusively established. Raw internal reports contain host-specific
metadata and are not automatically redistributed. Reproduction scripts and the
small numerical oracle fixtures are included in this source release.
