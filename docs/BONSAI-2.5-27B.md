# Ternary-Bonsai-2.5 27B

Prism ML's Bonsai 2.5 27B is a PTQ1_0 ternary GGUF with the same geometry as Bonsai 2 27B (64
layers, 5,120 wide, 17,408 FFN, 24 heads / 4 KV heads). The release ships two files:

| file | what it is |
|---|---|
| `Ternary-Bonsai-2.5-27B-PTQ1_0.gguf` | the language model |
| `Ternary-Bonsai-2.5-27B-PTQ1_0-mtp.gguf` | the same language model plus its MTP draft head as a 65th block (`blk.64`, Q4_0, with `nextn.{eh_proj,enorm,hnorm,shared_head_norm}`) |

## What this branch changes

- **`kindi convert` accepts the `-mtp` file.** The 65th block is not a model layer: the converter
  builds the 64-layer pack, skips that block's tensors, excludes them from the tensor-type check
  (they are Q4_0 inside a PTQ1_0 file) and records `"mtp_head": {"block": 64, "tensors": [...]}` in
  `kindi_pack.json`. Both files therefore produce the same pack. The head is found by its own
  `nextn.eh_proj.weight` marker at a known model's layer count rather than by trusting the
  metadata's block count, so a file whose count already excludes the head (64 while `blk.64`
  exists) converts the same way as Prism's 65-block one, and a real layer past the head is refused
  instead of being dropped.
- **Catalog entries** `bonsai-2.5-27b` and `bonsai-2.5-27b-mtp` (file, repo, sha256, size, card
  memory), so `kindi use` downloads and checks the right file.
- **`reasoning_effort = low` renders the `xhigh` instruction.** The 2.5 card says `low` is not
  supported and that the model then behaves close to `xhigh`; the template's own `low` text is not
  what the model was trained on, so sending it would be a setting the model does not honour. `low`
  is still accepted (clients send it), `xhigh` stays the default and `medium` the shorter-answer
  setting.

## Verified

Measured on one host while developing this support (x86-64, 12 threads, no CUDA device; the CPU
engine), with the files named above:

- Both files convert. The plain file gives the same pack as Bonsai 2 27B's geometry; the `-mtp` file
  gives a pack whose manifest records the head block, and its source sha256 matches the catalog.
- Both packs load on the CPU engine and generate on-topic replies for thinking and instruct
  requests, and their token ids are identical to each other at the same seed and settings - the head
  is recorded, not used.
- The 27B template renders byte-identically to the published template (26 of 26 probe messages), and
  `reasoning_effort = low` produces the same ids as `xhigh` on a shared seed, which is what the card
  predicts.

These checks were run in a fork with a larger test suite; this branch carries the converter,
catalog and template changes. The two decisions that make the intake work are pure functions, so
`tests/test_bonsai25_intake.py` covers them without a pack, a GPU or a compiler: `mtp_head_block`
over the plain file, Prism's 65-block count, a count that excludes the head, a layer past the head
and a marker at a block index that is not a known model's size; and `low`/`xhigh`/`medium`
rendering with an unknown effort refused. It needs numpy and pytest only:

    python -m pytest tests/test_bonsai25_intake.py      # 8 passed

## Not verified, and what is missing here

- **No GPU run.** No CUDA device was available; the Arc A770 toolchain is probe-only, so every
  number above is CPU-only. Nothing is claimed for a card.
- **No exactness reference.** There is no llama.cpp reference transcript for this release, so
  `kindi verify` reports "no llama.cpp reference" and exits without a PASS. Converter correctness is
  argued from structure (tensor count, dimensions, packcheck), not from token equality with another
  engine.
- **The MTP head is not used.** It is recorded in the manifest and nothing drafts with it: there is
  no speculative decoding for the 27B here, so the `-mtp` file is (correctly) no faster than the
  plain one in this tree.
- **No per-model sampling defaults.** This tree has no model profile system, and its sampler has
  `min_p` and `repeat_penalty` but **no `presence_penalty`**. A client that wants the card's sets
  must send them itself; the thinking set's `presence_penalty = 0.0` needs nothing, but the instruct
  set's `presence_penalty = 1.5` cannot be expressed in this tree yet.

## The card's sampling sets

| mode | temperature | top_p | top_k | min_p | presence_penalty | repetition_penalty |
|---|---:|---:|---:|---:|---:|---:|
| thinking | 1.0 | 0.95 | 20 | 0.0 | 0.0 | 1.0 |
| instruct (non-thinking) | 0.7 | 0.80 | 20 | 0.0 | 1.5 | 1.0 |

Reasoning effort applies to thinking requests only: `xhigh` (default, most accurate), `medium`
(shorter answers, a balance of speed and accuracy), `low` (unsupported; behaves close to `xhigh`).

## Benchmarking it fairly: the settings a comparison must differentiate

A benchmark that reports one number for "Bonsai 2.5 27B" hides the choices that change its output.
At minimum, a fair comparison separates:

1. **Thinking vs instruct.** They are different sampling sets, not two labels for one run. Report
   them as separate rows, with `enable_thinking` recorded.
2. **Reasoning effort (`xhigh`, `medium`, `low`)** on thinking requests. `low` must reproduce
   `xhigh` token for token on the same seed; a difference means the template mapping regressed.
3. **Every sampler value, including the "off" ones** (`min_p = 0`, `repeat_penalty = 1`,
   `presence_penalty` per mode). Record them from the values actually used, not from the intent.
4. **The sampler order.** Kindi applies top-k, then top-p, then min-p, then temperature (llama.cpp's
   order); Hugging Face and vLLM apply the temperature first. At temperature < 1 those pick from
   different distributions, so a comparison must name the order as well as the values.
5. **The file.** Plain and `-mtp` packs must produce identical ids at identical settings; if they
   differ, the head leaked into the model.
6. **Seed and prompt set.** One seed per prompt across efforts and modes; publish the ids or a text
   prefix, so a reader can see the runs are the same model rather than only reading a rate.
7. **Length budget and context.** Fixed `max_tokens` and context; thinking replies spend tokens on
   reasoning, so a short budget silently truncates one mode and not the other.
8. **Host state.** A rate is only comparable on an idle machine with the same build, thread count
   and context. A busy host still yields valid ids and text but meaningless tokens/s; such a run
   belongs in a settings check, not in a speed table.
9. **Drafting and caches.** Whether any drafter was active and whether the prompt cache was warm
   changes the rate without changing the model.
10. **Quality claims need more than a few prompts.** Two prompts and a short budget show that the
    settings run; they say nothing about which effort answers better.
