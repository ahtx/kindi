# Qwen3.8-Flash-Next

Qwen3.8-Flash-Next is a mixture-of-experts model of about 125B parameters. Each token uses about 6B of them.
Kindi runs ISTA-DASLab's GSQ-RCO IQ3_S file of it (two GGUF shards, 83.6 GB) with its own engine. The graphics card holds the dense layers,
the output head and the most used experts. The CPU runs the other experts from host RAM. The
card's and the CPU's expert kernels give the same bits, so where an expert runs changes the speed, not the output.

## Requirements

| Part | Requirement |
|---|---|
| Graphics card | NVIDIA, compute capability 5.0 or later; cards below 7.5 use a CUDA 12.x PyTorch build (the installer picks it). 8 GB free on the main card; 12 GB or more is better (more experts on the card). With several cards, Kindi splits the layers over them |
| RAM | Flash-Next keeps most experts in host RAM, and each token reads most of the 50 GB from it. About 96 GB or more is recommended. With less RAM Kindi reads the experts from the disk, which is much slower. On hosts with less than about 128 GB, set `KINDI_QXG_PLE_LOCK=0`. Memory thrash occurred on a 97 GB host with the default lock |
| CPU | x86-64 with AVX2. The CPU runs the experts that are not on the card: more cores give more speed |
| Disk | about 92 GB free: the model (83.6 GB) and the MTP head during its conversion |
| Software | Linux, the NVIDIA driver, Python 3.10 or newer with `venv`, gcc. The installer puts PyTorch and the server's packages in its own environment |

`kindi cards` lists your cards and `kindi plan --model flashnext` shows where the model would run.

## Install

```bash
bash install.sh --model flashnext            # or: bash install.sh --place "flashnext:0"
```

The installer:

1. checks the computer and picks the cards (`kindi plan`);
2. makes a Python environment with PyTorch in `~/.kindi/flashnext/venv`;
3. downloads the two shards from ISTA-DASLab's repository on Hugging Face and checks each shard's sha256;
4. fetches the MTP head (see below) and converts it;
5. runs `kindi verify`: 8 prompts, 174 generated tokens, read teacher-forced against a reference run of the same
   file; each token must be the reference's top token or a near-tie (top two within 0.35 nats);
6. installs and starts the service `kindi-flashnext.service`: Kindi's OpenAI server on `127.0.0.1:8102`.

`--no-mtp` (or `KINDI_MTP=0`) leaves out the MTP head. `tools/install_flashnext.sh --help` lists the other options.

## The MTP head

The model's multi-token prediction (MTP) head is not in the GGUF file. The installer fetches its 31 tensors (5.2 GB)
from Qwen's official release, `Qwen/Qwen3.8-Flash-Next` on Hugging Face. It reads them by byte range, not the whole repository,
and converts them on your computer with Kindi's importer (`kindi/qwen4exp_mtp_import.py`, then a compact form with
Q2_0 experts). Kindi does not host converted weights.

The head is under the **Qwen Community License 1.0** (the `LICENSE` file of Qwen's release). The converted file keeps
the notice in its `general.license` field. The install means you accept that license.

The installer adds the head by default, and the server drafts with it by default (`KINDI_FN_DRAFT=auto`). Greedy requests
get 2 drafted tokens for each step, and the tokens are the same as plain generation. Drafting runs when
the model is on one card of compute capability 5.0 or later. The card needs about 6.2 GB free for the dense layers, the head
and the buffers. On compute capability 8.0 and later (RTX 30 series and later) the drafted output is the same as plain
output. On older cards (Maxwell, Pascal, Volta, Turing) the server writes about 15-30% faster with the head. There the output can
differ from plain output where two tokens are almost equally likely. Set `KINDI_FN_DRAFT=0` there for exact output.
With several cards, less memory or sampled requests, the server runs plain and does not load the
head, so the head takes no card memory from the experts.
An install with `--no-mtp`, or `KINDI_FN_DRAFT=0`, runs plain; `KINDI_FN_HEAD_IDLE=1` loads the head without drafting.

## Using it

The server is the same OpenAI API as for the other Kindi models (`/v1/chat/completions`, streaming, `/v1/models`,
`/v1/tokenize`, `/health`). The model id is `qwen38-flash-next`.

- Thinking: on by default; `"chat_template_kwargs": {"enable_thinking": false}` turns it off. The thinking comes back
  as `reasoning_content`.
- Tool calls: the model's chat template (from the GGUF) with tool calls in XML; Kindi returns them as OpenAI
  `tool_calls`.
- Sampling: when the request does not set them, Qwen's published values: temperature 0.6, top_p 0.95, top_k 20 with
  thinking; 0.7, 0.8, 20 without.
- A new turn of the same chat reads only the new tokens.
- `/health` has `experts`. It shows the experts on the cards, the card hit rate and the CPU-expert time per token of the last
  request. It also shows whether the server has loaded the MTP head.

## Speed

One computer gave these measurements with the default install: an RTX 3060 12 GB, two Xeon E5-2673 v4 (40 cores), 755 GB of RAM,
the model in RAM. Writing: 8 new chats (not seen before by the expert placement), 200 tokens each, short context,
greedy, three runs. Reading: one prompt of 4,096 or 16,384 tokens.

| | tokens/s |
|---|---|
| Writing (the default) | 39.4 (39.84, 39.22, 39.35) |
| Reading a 4K-token prompt | 489 |
| Reading a 16K-token prompt | 465 |

`kindi verify` passes 174 of 174 tokens on this build (it checks the exact path).

The next test used a Quadro M4000 8 GB (Maxwell, sm_52) as the only card, with the same CPUs, RAM and model file. The default install
writes 13.1 tokens/s (14.09, 13.09, 12.88) and reads a 4K-token prompt at 54.7 tokens/s; `kindi verify` passes 174 of
174. llama.cpp on the same card and the same file writes 5.4 tokens/s and reads 4K at 34.2.

Three speed features are on by default. Each has a switch that gives the exact or the float path back:

- Fast mode leaves out the lowest-ranked CPU expert picks when it writes, and runs the prompt's attention in TF32. On
  kbench it solves the same tasks as exact mode (62 of 65 each), but its answers are not token-identical to the exact
  path. `KINDI_FAST=0` gives exact output; `kindi verify` always checks the exact path.
- Drafting with the MTP head (see above). Below compute capability 8.0 the output can differ from plain output where
  two tokens are almost equally likely; `KINDI_FN_DRAFT=0` gives exact output.
- The dense layers read 8-bit integer inputs (dp4a). On kbench it solves 62 of 65 tasks (the float path: 61 of 65).
  `KINDI_QXG_IDENSE=0` gives the float path back.

Measured on 11-22 with the 0.7.0 development build.
## Licenses

- The GGUF file: ISTA-DASLab's repository `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` (its card gives Apache-2.0;
  the base model is Qwen's).
- The MTP head: Qwen Community License 1.0 (above).
- Kindi: AGPL-3.0 (LICENSE).
