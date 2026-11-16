# Running Kindi with Docker

To install Kindi with one command (the image, the models and the servers), see [INSTALL.md](INSTALL.md). To use the
ready-made images from GitHub's container registry (no clone and no build), see [CONTAINERS.md](CONTAINERS.md). This
page tells how to build the images and do the steps by hand.

Kindi has three images:

| Image | Dockerfile | Contents | For |
|---|---|---|---|
| `kindi:0.6.0` | `docker/Dockerfile` | Ubuntu 18.04, CUDA 10.2, Python 3.6, CuPy 9.6, FastAPI, Kindi | NVIDIA cards from Kepler on, and the CPU engine |
| `kindi-rocm:0.6.0` | `docker/Dockerfile.rocm` | Ubuntu 24.04, ROCm 6.3.3, Python 3.12, CuPy 14.2.0 built for ROCm, FastAPI, Kindi | AMD RDNA3 graphics (tested: Radeon 780M) |
| `kindi-rocm7:0.6.0` | `docker/Dockerfile.rocm7` | Ubuntu 24.04, ROCm 7 (HIP 7.16), Python 3.12, CuPy 14.2.0 (cupy-rocm-7-0), FastAPI, Kindi | AMD RDNA 3.5 graphics (tested: Radeon 840M, Radeon 8060S) |

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
   Docker finds the toolkit only when its daemon starts. After you install the toolkit, restart the daemon
   (`sudo systemctl restart docker`, or `systemctl --user restart docker` for rootless Docker).

**Rootless Docker.** Replace `-u $(id -u):$(id -g)` with `-u 0:0` in the commands below. Rootless Docker runs the
container's root as your user. Any other container user, the image's own user 1000 included, is a different user on
the host: it is not the owner of `/data`, and `kindi convert` stops with
`PermissionError: [Errno 13] Permission denied: '/data/pack-...'`. The installer finds rootless Docker and uses
`-u 0:0`.

The images run as the user 1000:1000 when no `-u` is given. Their entry point (`python3 -m kindi.container`) takes
the same commands as `python3 -m kindi`, and with `KINDI_MODEL` set it also gets the model and builds its pack on the
first start ([CONTAINERS.md](CONTAINERS.md)).

AMD:

1. The `amdgpu` driver of Linux: `/dev/kfd` and `/dev/dri/renderD*` must exist. ROCm itself is in the image.
2. Docker.
3. On the Radeon 780M, the kernel option `amdgpu.cwsr_enable=0` ([AMD.md](AMD.md)).

## Build

```sh
docker build -t kindi:0.6.0 -f docker/Dockerfile .              # NVIDIA and CPU, about 15 minutes
docker build -t kindi-rocm:0.6.0 -f docker/Dockerfile.rocm .    # AMD; builds CuPy for ROCm, 30-45 minutes
```

## Commands

```sh
D="docker run --rm --gpus all -u $(id -u):$(id -g) -v $HOME/.kindi:/data kindi:0.6.0"
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
   -u $(id -u):$(id -g) -v $HOME/.kindi:/data kindi-rocm:0.6.0"
$A verify
```

`HSA_OVERRIDE_GFX_VERSION=11.0.2` is for the Radeon 780M (gfx1103) only. Other RDNA3 graphics do not need it.

## The server

```sh
docker run -d --name kindi --restart unless-stopped --gpus all -u $(id -u):$(id -g) \
    -p 127.0.0.1:8093:8093 -v $HOME/.kindi:/data kindi:0.6.0
docker logs -f kindi                                  # request log
docker exec kindi python3 -m kindi head fast          # switch the output head
```

The option `-p 127.0.0.1:8093:8093` keeps the server on this machine. To serve the network, use `-p 8093:8093` and
set an API key: `-e KINDI_API_KEY=<long random string>`.

To run one server on each card, start one container for each card, with its own name, port, card and cache folder:

```sh
docker run -d --name kindi-2 --restart unless-stopped --gpus all -u $(id -u):$(id -g) \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_VISIBLE_DEVICES=1 -e KINDI_STATE_DIR=/data/state-gpu1 \
    -p 127.0.0.1:8094:8093 -v $HOME/.kindi:/data kindi:0.6.0
```

Options go in as environment variables (`-e KINDI_CTX=16384`) or in `/data/kindi.conf`.
