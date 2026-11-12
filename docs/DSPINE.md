# DSpine: the block drafter of the 27B

Kindi 0.6.0 drafts the Ternary-Bonsai-2 27B with DSpine, a draft model for the Qwen3.8-27B family that Tencent
published under the Apache License 2.0 (tencent/Qwen3.8-27B-DSpine-b16 on Hugging Face). The weights are Tencent's.
The code that runs them (`kindi/dspine.py`, `kindi/dspine.cu`) is Kindi's own implementation.

## What it does

DSpine drafts a whole block of tokens in one pass. Its context is the 27B's own hidden states: the outputs of target
layers 1, 16, 31, 46 and 61 of every token that the 27B reads. The 27B's plans copy these five outputs into a feature
buffer. The drafter projects them to one vector per position, and each of its 5 layers keeps a K/V ring of the last
8192 positions.

A draft block is the current token, followed by 7 mask tokens. Each draft layer attends to the ring and causally
within the block. A code bridge between the layers passes a low-rank message from each block row to the next. The
block rows go through the 27B's own output norm and ternary head. A top-16 refinement chain then picks each draft
token, given the token before it. The 27B checks the block (the current token and 7 drafts) in one 8-token verify pass.

## Exactness

The 27B is unchanged. Its plans only copy the five layers' outputs (`k_feat_T`), and the 27B verifies every draft
token. So greedy output is the non-speculative greedy output, token for token. Sampled requests use speculative
sampling with a deterministic draft (`spec.spec_accept`), so the sampled tokens have the target's distribution.

Gate (RTX 3080, the installer's 27B profile, 8 prompts, 2,009 tokens): the greedy output with DSpine is the same
as without speculation. Test this with the `KINDI_FD=1` decode kernels of the profile. With `KINDI_FD=0`, the 2B drafter,
n-gram speculation and DSpine all differ from plain decoding at the same near-tie tokens. That difference is in the
verify and decode passes, not in a drafter.

## Speed

Same card (RTX 3080 at 190 W), same model, context 32768, the installer's profile (`KINDI_FD=1`, `KINDI_ATT_TC=1`):

| Drafter | Tokens per verify pass | Write tok/s |
|---|---|---|
| **DSpine (block 8, `KINDI_TC_TMAX=8`)** | **3.10** | **115.5** |

Code and math prompts accept more tokens per pass (up to 4.6) than prose (about 2.3 to 2.8).

The draft is one CUDA graph (about 160 kernels): int4 linears with int8 activations (dp4a), the bridge in fp16, and
the 27B's head for the candidates. It takes about 8 ms on a 3080, including the readback.

## Settings

| Setting | Meaning |
|---|---|
| `KINDI_DSPINE=DIR` | the folder with DSpine's `config.json` and `model.safetensors` (the installer downloads it to `~/kindi/dspine`) |
| `KINDI_SPEC=auto` | drafts with DSpine when it is attached (else the 2B drafter, else nothing); `KINDI_SPEC=dspine` asks for it |
| `KINDI_TC_TMAX=8` | the verify passes up to 8 tokens (the block). With 4, DSpine drafts 3 tokens per pass |
| `KINDI_DSPINE_K` | drafts per pass, at most 7 (default 7) |
| `KINDI_DSPINE_MB` | the card memory kept out of the context fit for the drafter (default 1400) |
| `install.sh --no-dspine` | leave DSpine out: the 27B drafts with the 2B, as in 0.5 |

## Memory

On the card: about 1.4 GB (the int4 linears, the fp16 bridge and the K/V rings). In host memory (mapped, pinned):
the bridge codebook (0.5 GB) and the head-by-write table for the tail (0.25 GB). Loading takes about 25 s: the
weights are quantized to int4 at load.

## Limits

- One NVIDIA card with the tensor-core plans (sm_61 and newer, not the P100's lane layout). Not on two cards or on ROCm.
- The 27B shape only (5120 wide, 64 layers). Other models keep their drafters.
- A restored prompt cache does not restore the drafter's ring. After a restore, the drafter sees only the
  positions read after it, until it has read more of the conversation. The output stays exact. Only acceptance is lower.
- Not measured yet on Pascal (GTX 1080). The draft kernels use dp4a, so they also run on sm_61.

## Tests

- `tools/dspine/test_kernels.py`: the kernels against NumPy references (one card, no model).
- `tools/dspine/bench_kernels.py`: kernel times at the draft's shapes.
- `tools/dspine/bench_spec.py`: plain, DSpine and the 2B drafter on the same prompts, with a check that the greedy
  tokens are the same, the tokens per pass and the write speed.
