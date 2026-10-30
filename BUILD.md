# Build and run Lightning V3 on a K6000

This procedure was exercised from a public Ubuntu base with Docker layer reuse
disabled on 2026-10-30. Numerical verification and basic API/restart checks passed
on a Quadro K6000. See VALIDATION.md for the exact scope.

## Requirements

- Linux x86_64, Quadro K6000 12 GB, a working compatible NVIDIA driver and NVIDIA Container Toolkit.
- Docker with BuildKit and Docker Compose v2 or later; permission to run Docker (prefix commands with sudo if required).
- Existing pinned GGUF from MODEL.md. No weights are included or downloaded by this procedure.
- Allow at least 20 GB free disk for image layers/build cache. The first build downloads a 2.65 GB CUDA installer plus packages.
- The tested host had 32 GB system RAM. CUDA download/extraction uses several GB of temporary RAM via BuildKit tmpfs; leave ample memory free.

Check `nvidia-smi`. The tested runtime was CUDA 10.2 / CuPy 9.6.0 / Python 3.6.9 on
driver 470.256.02. Do not replace a working host driver or CUDA installation for this
recipe: the legacy runtime is installed inside Docker. The single K6000 must have
enough free VRAM for the model; unload other models using that GPU before inference.

Run the following from the extracted/cloned repository directory. This is the
complete fork; a separate installation of original Kindi is not required.

## 1. Prepare private settings BEFORE invoking Compose

```sh
mkdir -p .local/pack .local/state
cp lightning.conf.example .local/kindi.conf
cp .env.example .env
```

Edit `.env`: set `MODEL_DIRECTORY` to your existing absolute model directory on
the Docker host. Set `GPU_DEVICE` to the K6000 device index. The default host API
port is 18095; change `LIGHTNING_PORT` if occupied. Keep `.env` and `.local/` private.
If inference will be exposed beyond localhost, set your own random `api_key` in
`.local/kindi.conf` and configure authenticated transport. Compose binds to
`127.0.0.1` by default and ships no credentials.

## 2. Build from public sources

```sh
docker compose build lightning
```

The validation used `docker compose build --no-cache lightning`. The first stage
follows upstream's `docker/Dockerfile`, starting with public `ubuntu:18.04`, and
installs CUDA 10.2 and pinned primary Python dependencies. CUDA extraction is
RAM-backed to avoid large temporary disk use. The second stage includes this
fork's complete source and checks SOURCE-FILES.txt. No private base image is needed.

The source recipe is tested, but image bytes are not reproducible: the OS tag,
APT packages and some transitive dependencies are not digest locked. Save the
built image identity and `python3 -m pip freeze` with any new validation run.

## 3. Register and verify the existing GGUF

```sh
docker compose run --rm lightning convert /models/Ternary-Nemotron-3.5-Lightning-30B-A3B-GGUF.gguf --out /workspace/pack
docker compose run --rm lightning verify
```

Use your file's actual name if different, after checking its identity against
MODEL.md. `convert` creates only a small pack manifest referring to the existing
read-only `/models` file. It does not create another copy of the weights. Keep the
same model mount for conversion and serving. `verify` checks the model hash,
three full vocabulary outputs and bit-exact snapshot restoration; it uses the GPU.

## 4. Start and check the API

```sh
docker compose up -d lightning
curl --fail http://127.0.0.1:18095/health
curl --fail -o corresponding-source.zip http://127.0.0.1:18095/source/archive
```

Adjust the port if changed in `.env`. Health must report `ok:true`, context 32768,
`head:exact` and `kv:fp32`. Model loading and initial kernel compilation take time;
watch `docker compose logs --tail 30 lightning`. Connect a client to
`http://127.0.0.1:18095/v1`, model `nemotron-lightning-30b-a3b-ternary`.
Send your configured Bearer API key for inference. The source/license routes are
public regardless of the inference key. The downloaded ZIP excludes weights and
private configuration; it is built from the reviewed manifest.

The explicit V3 profile is optimized kernels, tile16 dense/expert prefill, split
attention, chunk 128, FP32 state/KV and 32768 total input+output tokens. Optional
`fp16`/`q8_0` KV storage can be selected in the private config and verified separately.
Do not use the fast head for Lightning. This backend does not include a chat UI.

## Checks and shutdown

```sh
python3 scripts/check_release.py
python3 -m unittest discover -s tests -p test_source_offer.py -v
# In an existing environment with NumPy and Jinja2:
PYTHONPATH=. python3 tests/test_protocol.py
PYTHONPATH=. python3 tests/test_sampling.py
# Stop only this Compose project; model files and bind-mounted state remain:
docker compose down
```

For a separate test installation use a unique Compose project (`docker compose -p
YOUR_TEST_NAME ...` consistently), separate `.local/` directories and a different
host port. Docker health does not automatically restart a failed backend. After
diagnosing an unrecoverable CUDA error, restart this backend with
`docker compose restart lightning`. Inspect GPU acceptance scripts before use;
legacy cuda-memcheck is excluded for this K6000 profile.
