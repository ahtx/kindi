# Kindi 0.3.0 + Lightning backend V3 for Quadro K6000

A modified fork of [Kindi](https://github.com/ahtx/kindi), originally written by
**Amir Husain**. Maintained here by **slavalevski-tech**. Modifications dated
**2026-10-30** are described in [MODIFICATIONS.md](MODIFICATIONS.md).
This is an independent fork, not an official upstream release.

The backend runs **boro64/Ternary-Nemotron-3.5-Lightning-30B-A3B-GGUF** on the
Quadro K6000 with native CUDA/CuPy kernels. NVIDIA created the base model,
Nota AI produced the ternary checkpoint, and boro64 converted it to GGUF.
This repository distributes the backend source only. **No model weights are
included or automatically downloaded.** Lightning is distinct from Nemotron Bolt E4B.

## License and source

Program code: **AGPL-3.0-only**, without warranty. You may copy, modify and
redistribute it under [LICENSE](LICENSE). Preserve [NOTICE](NOTICE), upstream
attribution and change notices. [Licensing guidance](LICENSING.md) explains
source distribution and the separate model/dependency licenses. The upstream
commercial offer is preserved in [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md);
it does not relicense the additional contributions in this fork.

The Lightning API serves its complete shipped source at `/source/archive`,
with a prominent offer at `/` and `/source`, a license at `/license`, a link
in API documentation, and source/license links in HTTP response headers.
No API key is needed for the source download. Keep these routes reachable
through any reverse proxy. Rebuild and restart after changing source.

## Accepted V3 profile

- Linux, Quadro K6000 12 GB (Kepler sm_35), NVIDIA driver 470.256.02 in the accepted run.
- CUDA 10.2, CuPy 9.6.0, Python 3.6.9; Docker with NVIDIA Container Toolkit.
- `optimized` kernels, `tile16` dense/expert prefill, `split` attention, chunk 128.
- Exact full vocabulary head; FP32 state and default KV. FP16/Q8_0 are optional KV storage profiles.
- **32,768 tokens total: input plus output.** Full-model 64K/128K is not validated.
- One inference worker; prefix cache bounded to 512 MiB and eight entries.

See [BUILD.md](BUILD.md) for the source build and launch procedure using your
existing GGUF, [MODEL.md](MODEL.md) for its pinned identity, and
[BENCHMARKS-LIGHTNING.md](BENCHMARKS-LIGHTNING.md) for measured results and limits.
An RTX 4080 was used for independent validation; it is not a production requirement.
The runtime does not use llama.cpp, MLX or PyTorch for model computation.

## Validation status

Historical V3 acceptance covered operator/reference checks, full state/logit/token
comparisons, 32K retrieval, API, 100 requests, cancellation, cache and restart.
The publication copy changes packaging, attribution, release identity, explicit
V3 launch defaults and source-offer HTTP routes; accepted numerical kernels are
retained. Local publication checks are described in [VALIDATION.md](VALIDATION.md).
A clean Linux Docker build, K6000 numerical verification and basic API/restart
installation checks passed on 2026-10-30. This was not a rerun of the historical
long-context/stress benchmarks; see VALIDATION.md for scope.

Upstream documentation and its original benchmark claims remain under `docs/`
and [UPSTREAM-README.md](UPSTREAM-README.md). Those describe upstream
models/profiles, not Lightning V3. The upstream installer is retained for source
completeness; use BUILD.md for this backend rather than `install.sh`.
