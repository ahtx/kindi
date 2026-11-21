# Official v0.6.1 greedy parity goldens (2B/9B, 100/2K/8K)

This page pins **official stock Kindi v0.6.1** (`7fececbff13adc88bf21b8f07dea4e1c91996a9d`)
greedy continuations for the 2B and 9B at token positions 100, 2,000 and
8,000. The fork must reproduce the same token ids from the same fixed prompt
prefixes.

Host: i7-8086K + NVIDIA GeForce RTX 4060 Ti 8,188 MiB, driver 617.42, native
Windows, Python 3.13.6, CuPy 14.2.0, CUDA runtime 12.9 (12090). Packs:
`~/.kindi/packs/bonsai-2b` and `~/.kindi/packs/bonsai-9b`; their
`kindi_pack.json` hashes are recorded in the goldens.

The check is one command:

```sh
python tools/stock_parity.py --check
```

`--check` verifies that the local `v0.6.1` tag points at the pinned commit,
extracts a fresh stock tree from it, runs both trees, exits non-zero on the
first differing token, and refreshes the tracked machine gate
[`parity-report.json`](parity-report.json). The captures are regenerated only
by an explicit `--capture` from the official tag.

## Fixed prompts and settings

The prompt document is [`prompts.json`](prompts.json). It contains the first
100, 2,000 and 8,000 token ids of one fixed source document (the source
template, repeat count and SHA-256 are recorded in the file). The ids are
stored, not re-tokenized, so a tokenizer change cannot move the goalposts.

Both sides run:

- `ctx=8192`, the plain dense engine, exact head, no speculation, tensor cores,
  tandem, prompt cache or K/V format change;
- `temperature=0`, 32 greedy tokens per cell, no early stop;
- the same 2B and 9B packs as the earlier stock pages.

Stock v0.6.1 needs its native-Windows `libcuda.so.1` -> `nvcuda.dll` loader
shim; the worker applies that one name mapping before importing stock Kindi. It
does not patch an engine, kernel, plan or tokenizer.

## Goldens

[`goldens.json`](goldens.json) stores the stock outputs, the stock tag and
commit, the prompt hash, each pack's `kindi_pack.json` SHA-256, and the decoded
text for human inspection. The six cells are:

| model | position | prompt tokens | generated tokens |
|---|---:|---:|---:|
| 2B | 100 | 100 | 32 |
| 2B | 2,000 | 2,000 | 32 |
| 2B | 8,000 | 8,000 | 32 |
| 9B | 100 | 100 | 32 |
| 9B | 2,000 | 2,000 | 32 |
| 9B | 8,000 | 8,000 | 32 |

The prompt lengths are prefixes of the same fixed document, so the rows are
lengths of one document rather than three independently sampled conversations.

## Reproduction

```sh
# Check against the tracked goldens and refresh parity-report.json (default).
python tools/stock_parity.py --check

# Write a report somewhere else (for example, a narrowed debug run).
python tools/stock_parity.py --check --json /tmp/stock-parity.json

# Re-capture from the official tag. This replaces tracked evidence and is not
# part of ordinary CI.
python tools/stock_parity.py --capture --force
```

`--models 2b` or `--models 9b` and `--positions 100,2000` narrow a run for
debugging; the default is the full six-cell matrix. A narrowed or otherwise
non-default `--check` does not overwrite the tracked report unless `--json`
names a path. The harness refuses a pack whose `kindi_pack.json` hash differs
from the one recorded in the goldens.

## Result

The mandatory run reports **12/12 rows exact**: the six stock rows reproduce
the goldens and the six fork rows match them token for token. The machine gate
is `parity-report.json`; every canonical `--check` refreshes it, and a failed
canonical run writes `ok: false` with a failure record (the diagnostic is on
stderr), so a previous pass cannot remain current. This is a token parity
result, not a speed result.

## Limits

- The harness needs the two packs and a working CUDA GPU. The CPU-only suite
  checks the prompt/golden structure and comparison logic, not model output.
- Stock is the source tag plus the Windows loader shim, not a published
  release tarball or container.
- The matrix is one card, one pack per model, 32 greedy tokens, and one fixed
  document at three prefix lengths. It does not cover sampling, chat templates,
  tool calls, structured output, cache/session behavior, or other cards.
- The decoded text is recorded for inspection; parity is decided on token ids.
