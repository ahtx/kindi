# Ready-made container images

Kindi has ready-made images on GitHub's container registry. You do not need to clone the source or build an image.
You start a container, and it gets the model, builds the pack once and starts the server.

| Image | For |
|---|---|
| `ghcr.io/ahtx/kindi:cuda` (also `latest`) | NVIDIA cards from Kepler on (CUDA 10.2 in the image), and the CPU engine |
| `ghcr.io/ahtx/kindi:rocm6` | AMD RDNA3 graphics (ROCm 6.3 in the image; tested: Radeon 780M) |
| `ghcr.io/ahtx/kindi:rocm7` | AMD RDNA 3.5 graphics (ROCm 7 in the image; tested: Radeon 840M, Radeon 8060S) |

`cuda`, `rocm6`, `rocm7` and `latest` move to each new release. To keep one release, use the version tag, for
example `ghcr.io/ahtx/kindi:0.6.0-cuda`. Each release tag points to an image that passed `kindi verify` (the
token-for-token check against llama.cpp) on a real card before it was published. Kindi 0.6.0: `0.6.0-cuda` on NVIDIA GeForce RTX 3080; `0.6.0-rocm6` on AMD Radeon Graphics (with HSA_OVERRIDE_GFX_VERSION=11.0.2); `0.6.0-rocm7` on AMD Radeon 780M Graphics (RDNA3, run with HSA_OVERRIDE_GFX_VERSION=11.0.2).

The images are the same as the ones that the [one-command installer](INSTALL.md) builds. The installer also selects
the models and the settings for your cards, so it is still the easiest way on one computer. The images are for
Compose, Podman, automated installs and servers. [DOCKER.md](DOCKER.md) tells how to build the images yourself.

## Host requirements

- **NVIDIA:** the NVIDIA driver (the 470 branch for Kepler cards; a current driver for later cards) and the NVIDIA
  Container Toolkit. `docker run --rm --gpus all ubuntu nvidia-smi` must show the cards.
- **AMD:** the `amdgpu` driver of Linux (`/dev/kfd` and `/dev/dri/renderD*` must exist). ROCm is in the image. On the
  Radeon 780M, also see [AMD.md](AMD.md).
- **Disk:** the model, its pack and its cache need about three times the size of the model file.

## Quick start with Compose

The compose files need Docker Compose v2 (the `docker compose` command) or podman-compose. The old Python
`docker-compose` 1.x is no longer maintained, and it fails with current Docker versions; use the `docker run` commands
below instead.

NVIDIA:

```sh
curl -fsSLO https://raw.githubusercontent.com/ahtx/kindi/main/compose.yaml
KINDI_MODEL=bonsai-4b docker compose up -d
docker compose logs -f
```

AMD (the numbers of your `video` and `render` groups are necessary, because the GPU device files belong to them):

```sh
curl -fsSLO https://raw.githubusercontent.com/ahtx/kindi/main/compose.amd.yaml
export VIDEO_GID=$(getent group video | cut -d: -f3) RENDER_GID=$(getent group render | cut -d: -f3)
KINDI_MODEL=bonsai-4b KINDI_TAG=rocm6 docker compose -f compose.amd.yaml up -d
```

On RDNA 3.5 graphics (Radeon 840M, 8060S), use `KINDI_TAG=rocm7`. On the Radeon 780M, add
`HSA_OVERRIDE_GFX_VERSION=11.0.2`.

The first start downloads the model from Hugging Face and checks its sha256. Then it builds the pack and starts the
server on `127.0.0.1:8093`. This takes from a few minutes (2B) to about 20 minutes (27B). The next starts take
seconds. When `/health` gives `"ok":true`, the server is ready:

```sh
curl -s http://127.0.0.1:8093/health
```

The settings go in the environment or in a `.env` file beside the compose file:

| Setting | Default | |
|---|---|---|
| `KINDI_MODEL` | `bonsai-4b` | a model id, or a GGUF file in the container (see below) |
| `KINDI_TAG` | `cuda` (`rocm6` in `compose.amd.yaml`) | the image tag, for example `0.6.0-cuda` |
| `KINDI_PORT` | `8093` | the port on the host |
| `KINDI_BIND` | `127.0.0.1` | `0.0.0.0` serves the network: then set `KINDI_API_KEY` |
| `KINDI_API_KEY` | empty | clients must then send `Authorization: Bearer <key>` |
| `KINDI_CTX` | `24576` | the context in tokens (Kindi reduces it to what the card holds) |
| `KINDI_DATA` | the named volume | a host folder for `/data` (an absolute path) |
| `KINDI_VOLUME` | `kindi-data` | the name of the named volume (the same volume as in the `docker run` examples) |
| `KINDI_UID` | `1000:1000` | the user in the container (see "Users and permissions") |
| `KINDI_VERIFY_ON_START` | `0` | `1`: run `kindi verify` once for each pack and kind of card before the server starts |

## The model

`KINDI_MODEL` takes one of two forms:

- **A model id.** `docker compose run --rm kindi models` lists them and tells which fit your cards: `bonsai-2b`,
  `bonsai-4b`, `bonsai-9b`, `bonsai-27b`, `nemotron-e4b` and `nanbeige-33b`. The container downloads the GGUF into
  `/data/models` (a stopped download continues at the next start) and checks its sha256.
- **A GGUF file in the container**, for example `KINDI_MODEL=/data/models/my-model.gguf`. Put the file into the
  volume, or mount its folder: `-v /srv/models:/data/models:ro`. A file with the name of a known model is checked
  against that model's sha256 once. A file name only (`my-model.gguf`) means a file in `/data/models`.

The container builds the pack once, into `/data/packs/<name>`, and uses it at each start. When the GGUF changes (its
sha256, or for an unknown file its size and time), the next start builds the pack again. When two containers start
at the same time on the same volume, the second one waits for the first one, and the pack is built once. A stopped
build leaves no part of a pack.

Without `KINDI_MODEL`, the image serves the pack at `KINDI_PACK` (default `/data/pack`), the same as in earlier
releases. `KINDI_DOWNLOAD=0` stops all downloads (for computers without internet access): then the GGUF must be in
`/data/models`.

## Without Compose

```sh
docker run -d --name kindi --restart unless-stopped --gpus all \
    -p 127.0.0.1:8093:8093 -v kindi-data:/data -e KINDI_MODEL=bonsai-9b ghcr.io/ahtx/kindi:cuda
```

```sh
docker run -d --name kindi --restart unless-stopped --device /dev/kfd --device /dev/dri \
    --group-add $(getent group video | cut -d: -f3) --group-add $(getent group render | cut -d: -f3) \
    --security-opt seccomp=unconfined -p 127.0.0.1:8093:8093 -v kindi-data:/data \
    -e KINDI_MODEL=bonsai-4b ghcr.io/ahtx/kindi:rocm6
```

The other commands work in the same image: `docker run --rm --gpus all -v kindi-data:/data -e KINDI_MODEL=bonsai-9b
ghcr.io/ahtx/kindi:cuda verify` (or `bench`, `info`, `models`).

## Users and permissions

The images run as the user `1000:1000`, not as root. A named volume (`kindi-data`) works without changes. For a host
folder:

- **Rootful Docker:** set `KINDI_UID` to the owner of the folder, for example `KINDI_UID=$(id -u):$(id -g)`
  (`-u $(id -u):$(id -g)` without Compose).
- **Rootless Docker and rootless Podman:** set `KINDI_UID=0:0` (`-u 0:0`). The root user of the container is your
  user on the host. The user 1000 of the container would be a different user on the host, which cannot write your
  folder.

## Podman

`podman-compose up -d` uses the same compose files. GPU access in rootless Podman is best effort in this release.
Full Podman support (the installer, and rootless GPU access on all cards) is planned for a later release.

- **NVIDIA:** Podman gives the cards through CDI. Make the CDI file once with `sudo nvidia-ctk cdi generate --output
  /etc/cdi/nvidia.yaml`. podman-compose 1.1 and later convert the GPU part of `compose.yaml` to a CDI device. With an
  earlier version, replace the `deploy:` part with `devices: ["nvidia.com/gpu=all"]`. Without Compose:
  `podman run --device nvidia.com/gpu=all -u 0:0 ...`.
- **AMD:** in rootless Podman, replace the `group_add:` part of `compose.amd.yaml` with `group_add: [keep-groups]`, and
  set `KINDI_UID=0:0`.

## Upgrade

```sh
docker compose pull && docker compose up -d
```

The models, the packs and the cache stay in the volume. A new release can change the pack format. Then the next
start builds the pack again.
