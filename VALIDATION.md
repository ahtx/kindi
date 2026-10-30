# Publication installation validation — 2026-10-30

A separate test Compose project was built from public ubuntu:18.04 with
`docker compose build --no-cache lightning`. No private base image was used.
The existing GGUF was mounted read-only; no model was downloaded or copied.

## Results

- Fresh Docker source build: passed, including the 135-file source manifest check.
- New image: Python 3.6.9, CuPy 9.6.0, NumPy1.19.5, FastAPI0.79.1,
  Starlette0.19.1, Uvicorn0.16.0, Pydantic1.9.2.
- CPU tests inside the new image:7 source-offer tests, protocol/cache checks and
  288 independent seeded sampling comparisons passed.
- `convert`: created the pack manifest for the existing GGUF, passed.
- `verify` on K6000, V3 tile16/split/chunk 128/FP32/ctx 32768: passed.
  Maximum absolute errors on the three reference logits arrays were
  6.67572021484375e-06, 7.659196853637695e-06 and 3.814697265625e-06;
  snapshot restoration was bit-exact.
- API: health, inference authentication, ordinary completion (`42`), SSE completion,
  public source/license routes, and restart followed by healthy startup passed.
- The source ZIP returned by the running API contained exactly 135 reviewed files;
  every uncompressed file matched the tested publication tree byte for byte.
- No new NVRM Xid/Oops/BUG kernel log entries were found during the GPU test.
- The original working service was temporarily stopped with user authorization,
  then restored using its original container, image and configuration. It became
  healthy and independently answered 42. Other containers retained their identities,
  configuration and state. The test containers were removed afterward.

Tested local image identity (not a published registry image):
`sha256:8494986d0099f6f4f72eacc4690d5eaf680fa07a7640d467a84dd9dd005cb15b`.
Its source archive SHA256 was
`2b52319a879bffb38a109392817cfeefcf6d9adf26d622252ef4fe08f9d23f9f`.
After the test, documentation was updated and Git attributes/line endings were
prepared for publication. Runtime logic and the tested Docker build steps are unchanged.

## Issue found and fixed

The first attempt exhausted its temporary-disk reserve during CUDA extraction and
was cancelled before it could fill the host disk. Dockerfile.lightning now downloads
and extracts CUDA under a BuildKit tmpfs. The second clean build completed. BUILD.md
also puts private settings before Compose commands and states build-resource needs.

## Scope

This was an installation/numerical smoke/API/restart check, not a repetition of
the historical 32K retrieval, 100-request stress or performance benchmarks. The engine
was configured at32768 total input+output capacity, but no new 32K prompt test was
performed. Those historical results remain separately summarized in
BENCHMARKS-LIGHTNING.md. No new claims about 64K/128K or other models are made.

LICENSE and COMMERCIAL-LICENSE remain unchanged in text; original copyright/SPDX
lines are retained. CUDA kernels match accepted development V3 after line-ending normalization. The three NPZ
fixtures contain numerical outputs, not model weights. The explicit manifest and
selected secret/host-pattern checks pass; pattern scanning is not a universal
proof of absence of secrets. Internal infrastructure logs/configuration are not
part of this public source payload. No GitHub repository or image was published.
