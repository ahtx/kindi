# MTP: multi-token prediction

## What it is

An MTP head is a small extra layer that predicts the token after the next one. Kindi uses it to draft tokens.
The main model then checks all the drafted tokens in one pass. Kindi keeps the drafted tokens that equal the
main model's own tokens, and adds one token of its own.

The output is exactly the same as without MTP. MTP changes only the speed.

Kindi has one head: `kindi-mtp-2b-v1.npz` (61 MB), for Ternary-Bonsai-2-2B. It is our own work.

- Download: https://huggingface.co/ahtx/kindi-mtp-heads (the installer downloads it and checks its sha256).
- Base model: Ternary-Bonsai-2-2B by Prism ML (Apache-2.0).
- The head contains weights derived from Qwen3.5-2B by the Qwen team at Alibaba Cloud (Apache-2.0).
- The manifest is `kindi/mtp_heads.json`.

## The modes

| Mode | Setting | Drafts |
|---|---|---|
| A | `KINDI_SPEC=mtp`, `KINDI_MTP_HEAD=<file>` | The model's own head drafts k tokens (`KINDI_SPEC_K`). |
| B0 | `KINDI_SPEC=draft2b`, `KINDI_SPEC_PACK=<2B pack>` | The 2B model drafts k tokens for the 27B. |
| B2 | B0 and `KINDI_SPEC_DRAFT_HEAD=<file>` | The 2B does one step; then the 2B's head drafts the other k - 1 tokens. |
| auto | `KINDI_SPEC=auto` | Kindi selects the mode for each step (see below). |

## Which cards

MTP runs only on the tensor-core path: NVIDIA sm_80 and later, with `KINDI_TC=1`.

On older cards, Kindi does not use MTP. On these cards, a pass over T tokens costs almost T single steps, so a
draft cannot save time. These are the measurements for T = 2, 3 and 4 tokens per pass (pass time / step time):

| Card | T = 2 | T = 3 | T = 4 | Mode A, predicted |
|---|---|---|---|---|
| Quadro K1200 (sm_50), 2B | 1.80 | 2.61 | 3.42 | 0.80x (k=1) |
| GTX 1080 (sm_61), 27B | 1.82 | 2.81 | 3.65 | 0.95x (k=1) |

## Measured speed

These measurements used greedy decoding, 128 tokens (2B: 512 tokens, the median of 3 runs), and the same text on
each card. The cells give tok/s and (speed-up). The full tables are in `docs/dev/bench-0.4.0/mtp.md`.

| Card | Model and mode | 512 tokens of context | 8K | 24K |
|---|---|---|---|---|
| RTX 3080 (190 W) | 27B, no speculation | 62.3 | 61.4 | 58.2 |
| RTX 3080 (190 W) | 27B, 2B drafter, k=2 | 91.1 (1.46x) | 103.2 (1.68x) | 98.7 (1.70x) |
| RTX 3080 (190 W) | 27B, 2B drafter, k=3 | 85.1 (1.37x) | 111.5 (1.82x) | 102.5 (1.76x) |
| RTX 3060 | 27B, no speculation | 41.1 | 40.4 | 36.9 |
| RTX 3060 | 27B, auto | 60.6 (1.48x) | 73.5 (1.82x) | 66.7 (1.81x) |
| RTX 3080 (190 W) | 2B, no MTP | 370.8 | 368.6 | 349.0 |
| RTX 3080 (190 W) | 2B, head, k=2 | 409.0 (1.10x) | 395.5 (1.07x) | 336.1 (0.96x) |
| RTX 3060 | 2B, no MTP | 316.9 | 314.4 | 287.0 |
| RTX 3060 | 2B, head, k=2 | 350.7 (1.11x) | 340.3 (1.08x) | 273.3 (0.95x) |

The output of every mode was the same as without speculation. B2 was slower than the 2B drafter alone (B0) on
both cards.

## How auto mode selects

- A model with its own head (the 2B with `KINDI_MTP_HEAD`): the head, greedy requests only.
  - Up to 8K tokens of context (`KINDI_SPEC_MTP_K2_MAX`, default 8192): k=2.
  - Above that: no speculation.
- A model with the 2B drafter (the 27B): B0.
  - Below 1K tokens of context (`KINDI_SPEC_B0_K2_MAX`, default 1024): k=2.
  - At 1K and above: k=3.
  - Sampled requests use B0 with speculative sampling.
  - B2 is off (`KINDI_SPEC_B2_MAX`, default 0). Set it to a context length to use B2 below that length.
- Every other case: no speculation.

The speculation guard (`KINDI_SPEC_GUARD`, on by default) pauses speculation when it is slower than plain
decoding. It works from 32K tokens of context (`KINDI_SPEC_GUARD_FROM`, default 32768). Below 32K, its pauses cost
9-12% on the RTX 3060.

## How to turn it off

Set `KINDI_SPEC=0` to turn off all speculation. To keep the 2B drafter and stop only the head, set
`KINDI_SPEC=draft2b` and remove `KINDI_SPEC_DRAFT_HEAD`. On the 2B, remove `KINDI_MTP_HEAD`.

The installer turns on `KINDI_SPEC=auto` only on tensor-core profiles. For the 27B, auto needs only the 2B
drafter. For the 2B, the installer downloads the head file and checks its sha256. If that fails, it prints one
line, and the 2B runs without MTP.
