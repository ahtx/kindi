# Running Kindi with Docker

To install Kindi with one command (the image, the models and the servers), see [INSTALL.md](INSTALL.md). This page
tells how to do the same steps by hand.

Kindi's images are built in two tiers, so that a code change rebuilds only the code and not the dependencies:

| Tier | Image | Dockerfile | Contents | For |
|---|---|---|---|---|
| base | `kindi-base:0.3.0` | `docker/base/Dockerfile` | Ubuntu 18.04, CUDA 10.2, Python 3.6, CuPy 9.6, FastAPI | NVIDIA cards from Kepler on, and the CPU engine |
| base | `kindi-base-rocm:0.3.0` | `docker/base/Dockerfile.rocm` | Ubuntu 24.04, ROCm 6.3.3, Python 3.12, CuPy 14.2.0 built for ROCm, FastAPI | AMD RDNA3 graphics (tested: Radeon 780M) |
| code | `kindi:0.3.0` | `docker/Dockerfile` | the Kindi code, on top of `kindi-base:0.3.0` | NVIDIA cards from Kepler on, and the CPU engine |
| code | `kindi-rocm:0.3.0` | `docker/Dockerfile.rocm` | the Kindi code, on top of `kindi-base-rocm:0.3.0` | AMD RDNA3 graphics (tested: Radeon 780M) |

The base image holds everything that changes slowly - the OS, the CUDA / ROCm toolchain, Python, CuPy and the Python
packages. The code image is just `COPY kindi` on top of it, so a code change rebuilds in seconds. The code images keep
the `kindi:0.3.0` and `kindi-rocm:0.3.0` tags, so `knd`, the installer and the run examples below keep working.

Everything Kindi reads or writes lives in one volume mounted at `/data`:

| path in the container | |
|---|---|
| `/data/pack` | the converted model (`kindi convert`) and `head_clusters.npz` (`kindi fast-head`) |
| `/data/state` | kernel cache, saved prompt openings, head mode |
| `/data/kindi.conf` | optional config file (pass `-e KINDI_CONFIG=/data/kindi.conf`) |

## Host prerequisites

NVIDIA:

1. The NVIDIA driver. Kepler cards (K420) need the **470** branch; on a distribution that would upgrade it, hold the
   package. Later cards: a current driver.
2. Docker and the **NVIDIA Container Toolkit** (`docker run --rm --gpus all ubuntu nvidia-smi` must show the cards).

AMD:

1. The `amdgpu` driver of Linux: `/dev/kfd` and `/dev/dri/renderD*` must exist. ROCm itself is in the image.
2. Docker.
3. On the Radeon 780M, the kernel option `amdgpu.cwsr_enable=0` ([AMD.md](AMD.md)).

## Build

Two scripts build the two tiers. The version in the image tag comes from `KINDI_VERSION` (default `0.3.0`, matching
`install.sh`):

```sh
docker/build-base.sh cuda    # kindi-base:0.3.0 - NVIDIA and CPU, about 15 minutes (downloads the CUDA 10.2 toolkit)
docker/build-base.sh rocm    # kindi-base-rocm:0.3.0 - AMD; builds CuPy for ROCm, 30-45 minutes
docker/build-code.sh cuda    # kindi:0.3.0 - the Kindi code on top of the base image, seconds
docker/build-code.sh rocm    # kindi-rocm:0.3.0 - the Kindi code on top of the ROCm base image, seconds
```

Without an argument, each script builds both variants (`all`). Build the base image only when the dependencies
change; after a code change, `docker/build-code.sh` alone is enough. The same result can be produced by hand:

```sh
docker build -t kindi-base:0.3.0 -f docker/base/Dockerfile .
docker build -t kindi:0.3.0 --build-arg BASE_IMAGE=kindi-base:0.3.0 -f docker/Dockerfile .
```

A `.dockerignore` keeps the build context small: it excludes the converted models (`pack`), the downloaded GGUFs
(`models`), the caches (`state`), `.git` and Python bytecode. Without it, every build would upload the pack (tens of
GB) to the Docker daemon.

After `docker/build-code.sh cuda`, `./knd info` runs the freshly built code, because the code image keeps the
`kindi:0.3.0` tag that `knd` uses.

## Commands

```sh
D="docker run --rm --gpus all -u $(id -u):$(id -g) -v $HOME/.kindi:/data kindi:0.3.0"
$D info
$D convert /data/Ternary-Bonsai-2-2B-PTQ1_0.gguf
$D verify
$D fast-head
$D bench
```

On AMD graphics, replace `--gpus all` with the AMD devices, and add the group IDs of `video` and `render`:

```sh
A="docker run --rm --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined \
   --group-add $(getent group video | cut -d: -f3) --group-add $(getent group render | cut -d: -f3) \
   -e HSA_OVERRIDE_GFX_VERSION=11.0.2 -e HSA_ENABLE_SDMA=0 \
   -u $(id -u):$(id -g) -v $HOME/.kindi:/data kindi-rocm:0.3.0"
$A verify
```

`HSA_OVERRIDE_GFX_VERSION=11.0.2` is for the Radeon 780M (gfx1103) only. Other RDNA3 graphics do not need it.

## The server

```sh
docker run -d --name kindi --restart unless-stopped --gpus all -u $(id -u):$(id -g) \
    -p 127.0.0.1:8093:8093 -v $HOME/.kindi:/data kindi:0.3.0
docker logs -f kindi                                  # request log
docker exec kindi python3 -m kindi head fast          # switch the output head
```

The option `-p 127.0.0.1:8093:8093` keeps the server on this machine. To serve the network, use `-p 8093:8093` and
set an API key: `-e KINDI_API_KEY=<long random string>`.

To run one server on each card, start one container for each card, with its own name, port, card and cache folder:

```sh
docker run -d --name kindi-2 --restart unless-stopped --gpus all -u $(id -u):$(id -g) \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_VISIBLE_DEVICES=1 -e KINDI_STATE_DIR=/data/state-gpu1 \
    -p 127.0.0.1:8094:8093 -v $HOME/.kindi:/data kindi:0.3.0
```

Options go in as environment variables (`-e KINDI_CTX=16384`) or in `/data/kindi.conf`.
