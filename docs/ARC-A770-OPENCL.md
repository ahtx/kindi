# Arc A770 OpenCL probes (against 0.7.2)

This branch adds an experimental **OpenCL** path for Intel Arc GPUs on top of the 0.7.2 release.
It is a set of probes, not a backend: nothing here is wired into `serve`, `chat` or the CUDA
kernels, and no support claim is made. It exists so the ternary pack format can be exercised on an
Intel card from the released tree.

## What runs

| command | what it does |
|---|---|
| `kindi intel-cl` | compiles and runs vector-add/GEMM on the device (tests the loader and the toolchain) |
| `kindi intel-ternary` | the packed ternary GEMV (`pack_gemv4`, `pack_gemv_persist`) on synthetic shapes |
| `kindi intel-norm` | the decode's norm/RMS + signed FWHT fold and the embedding row |
| `kindi intel-pack` | every ternary tensor shape of a real pack, timed, validated against the host model |
| `kindi intel-step` | one full decode step, teacher-forced against the CPU engine |
| `kindi intel-run` | greedy decode of a prompt on the device, no CPU engine at all |

Everything reads the pack `kindi convert` already writes (`bcodes.bin`, `dpm.bin`, `f32.bin`,
`signs_*.f32`, `bcodes_index.json`, `kindi_pack.json`), so no pack-format change is involved.

## Measured on one A770 (16 GB, driver 32.0.101.9175, OpenCL C 1.2, no `cl_khr_fp64`)

| run | result |
|---|---|
| `intel-run`, Bonsai 2 2B, `"The capital of France is"` | 4 tokens `[11751, 13, 198, 760]` = `" Paris.\nThe"`, **147.14 ms/step**, 516 enqueues/step |
| `intel-run`, Bonsai 2.5 27B, same prompt | 3 tokens `[11751, 13, 248046]` = `" Paris.<|im_end|>"`, **831.29 ms/step**, 1366 enqueues/step |
| `pytest tests/test_intel_*.py` | 41 passed, no GPU needed |

Both token sequences are the ones the CPU engine produces from the same packs, so the device path
is correct where it runs. It is **not fast**: the step pays one blocking read per GEMV group and
leaves the delta rule, conv, attention, softmax and the FFN elementwise on the host, which is why a
27B token costs ~0.8 s here.

## Notes

- The driver refuses a single allocation over 4095 MiB, so a 27B pack's 6.7 GB of codes are grouped
  into device buffers on tensor boundaries and each GEMV's offset is chunk-relative.
- `pack_gemv4`/`pack_gemv_persist` accumulate in float32. An earlier integer-only validation did not
  catch that the folded input was truncated to int32 before the dot; the tests use fractional
  inputs on purpose.
- `phase_bytes` now packs per row (`(nb, N) -> (nb, N/4)`), which is byte-identical for the
  128-column phases the converter writes but also handles wider test matrices.
- The head tensor is taken from the pack manifest: the 2B ties it to `token_embd.weight`, the 2.5
  27B ships a separate `output.weight`.
- `intel-step` needs the CPU engine as its oracle, which needs `gcc` with OpenMP on the host. The
  device-only commands (`intel-run`, `intel-pack`, `intel-cl`) do not.
