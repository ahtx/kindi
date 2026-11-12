# Lightning backend: clean-room record

This note records how we made the Kindi backend for Nemotron 3.5 Lightning 30B-A3B. It also gives the status of
each file of the contributed fork.

## Source of the work

- Contributor: slavalevski-tech. The contributor agreed to the Kindi CLA 1.1 on 2026-11-01 and applied it to
  ahtx/kindi PR #16.
- Fork: https://github.com/slavalevski-tech/kindi. We read two commits: 097c429 (the head of PR #16) and 5afe811
  (the fork's "V10" main). We did not run code from the fork.
- The fork's provenance file names llama.cpp files as cross-check references for its model graph, its scan and its
  packed formats. Kindi does not contain llama.cpp code. So we rewrote the backend. We did not copy code from the
  fork. During the review we read parts of the fork: its documents, its dispatch changes, the format comment of
  packed.py and the start of its tokenizer.

## How we wrote the Kindi backend

- The model graph comes from the public model configuration (NVIDIA-Nemotron-3.5-Lightning-30B-A3B config.json) and
  from the GGUF metadata of the model file.
- The Q2_0 block layout comes from the data. We compared one row of the GGUF file with the same row of the Nota AI
  MLX checkpoint. The result: weight j of a 64-weight block is in byte j / 4, bits 2 (j % 4). Its value is
  (code - 1) * d, and d is the fp16 scale at the start of the block.
- The kernels use the conventions of Kindi's own kernels. These are the int8 runs of `nbq.cu`, the tensor-core
  tiles of `nbq_mm.cu` and `tc.cu`, the attention of `nbq_fa.cu` and the Kepler arithmetic of `nbq_k35.cu`.
- The tokenizer pattern comes from the model's tokenizer.json. The chat template comes from the GGUF file.
- llama.cpp is a black box only. We used its outputs (the verify reference), its speed and its tokenizer output. We
  did not read its source.

## Status of the fork files

| Fork file or group | Status | Kindi file |
| --- | --- | --- |
| `kindi/lightning.py`, `lightning_batch.py`, `moe.py`, `packed.py`, `kv.py`, `prefix_cache.py` | Rewritten | `kindi/lightning.py` |
| `kindi/lightning.cu`, `lightning_batch.cu`, `lightning_kv.cu`, `lightning_prefill.cu`, `lightning_prefill_aos.cu`, `lightning_attention_load2.cu`, `lightning_attention_load4.cu` | Rewritten | `kindi/lightning.cu`, `lightning_tc.cu`, `lightning_k35.cu` |
| `kindi/lightning_tokenizer.py` | Rewritten | `kindi/lightning_tokenizer.py` |
| `kindi/lightning_server.py`, `lightning_cli.py` | Rewritten | `kindi/lightning_gen.py`, `lightning_chat.py`, `lightning_cli.py` |
| Dispatch changes in `kindi/__init__.py`, `__main__.py`, `config.py`, `convert.py`, `server.py` | Rewritten | the same files (a small dispatch on backend `kindi_lightning`) |
| `kindi/source_offer.py` | Dropped |  |
| `kindi/lightning_reference*.npz`, `lightning_kv_reference.json` | Dropped | `kindi/reference-nemotron-lightning-30b.json` (llama.cpp output) |
| `tests/*.py` (about 30 files) | Dropped: they test the fork's own functions |  |
| Changelogs, readmes, provenance and licensing documents (about 15 files), `BUILD.md`, `INSTALLATION.md` | Dropped |  |
| `Dockerfile.lightning`, `compose.yaml`, `.env.example`, `lightning.conf.example`, `release_entry.py`, `scripts/check_release.py` | Dropped |  |
| `tests/bench_*.py`, `profile_lightning.py`, `diagnose_prefill.py`, `calibrate_kv.py` | Dropped |  |
| `licenses/Nemotron-OpenMDW-1.1.txt` | Dropped | a link to the model licence in the documentation |

We kept no fork file. We credit the contributor in NOTICE, for the backend's scope (this model and file) and for the
validation work.

## Later work on the branch (11-03)

These parts are new. No fork file or llama.cpp source was used for them:

- `lt_attn_pre6` (prompt attention for cards without tensor cores) and the 16-byte K/V loads in `lt_attn_pre8`.
- The 4-launch decode MoE layer (`lt_router_topk`, `lt_quant2`) and the decode slots for host experts
  (`lt_cache_dec`).
- The sm_75 PTX target in `kindi/tc.py`.
- The server changes in `lightning_gen.py` and `nanbeige_gen.py` (role markers, control tokens, quoted calls, header
  slips). We made them from the model's own replies in the terminal-bench runs and in the live tool tests.

The llama.cpp numbers for cards that do not hold the model use its own settings: `-ncmoe` to put experts on the CPU,
and on the RTX 2060 a separate process that holds the same memory that Kindi holds back. These are black-box runs.
