# Running Kindi with Docker

To install Kindi with one command (the image, the models and the servers), see [INSTALL.md](INSTALL.md). To use the
ready-made images from GitHub's container registry (no clone and no build), see [CONTAINERS.md](CONTAINERS.md). This
page tells how to build the images and do the steps by hand.

Kindi has three images:

| Image | Dockerfile | Contents | For |
|---|---|---|---|
| `kindi:0.7.0` | `docker/Dockerfile` | Ubuntu 18.04, CUDA 10.2, Python 3.6, CuPy 9.6, FastAPI, Kindi | NVIDIA cards from Kepler on, and the CPU engine |
| `kindi-rocm:0.7.0` | `docker/Dockerfile.rocm` | Ubuntu 24.04, ROCm 6.3.3, Python 3.12, CuPy 14.2.0 built for ROCm, FastAPI, Kindi | AMD RDNA3 graphics (tested: Radeon 780M) |
| `kindi-rocm7:0.7.0` | `docker/Dockerfile.rocm7` | Ubuntu 24.04, ROCm 7 (HIP 7.16), Python 3.12, CuPy 14.2.0 (cupy-rocm-7-0), FastAPI, Kindi | AMD RDNA 3.5 graphics (tested: Radeon 840M, Radeon 8060S) |

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
docker build -t kindi:0.7.0 -f docker/Dockerfile .              # NVIDIA and CPU, about 15 minutes
docker build -t kindi-rocm:0.7.0 -f docker/Dockerfile.rocm .    # AMD; builds CuPy for ROCm, 30-45 minutes
```

## Commands

```sh
D="docker run --rm --gpus all -u $(id -u):$(id -g) -v $HOME/.kindi:/data kindi:0.7.0"
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
   -u $(id -u):$(id -g) -v $HOME/.kindi:/data kindi-rocm:0.7.0"
$A verify
```

`HSA_OVERRIDE_GFX_VERSION=11.0.2` is for the Radeon 780M (gfx1103) only. Other RDNA3 graphics do not need it.

## The server

```sh
docker run -d --name kindi --restart unless-stopped --gpus all -u $(id -u):$(id -g) \
    -p 127.0.0.1:8093:8093 -v $HOME/.kindi:/data kindi:0.7.0
docker logs -f kindi                                  # request log
docker exec kindi python3 -m kindi head fast          # switch the output head
```

The option `-p 127.0.0.1:8093:8093` keeps the server on this machine. To serve the network, see
[Serve your local network](#serve-your-local-network).

To run one server on each card, start one container for each card, with its own name, port, card and cache folder:

```sh
docker run -d --name kindi-2 --restart unless-stopped --gpus all -u $(id -u):$(id -g) \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_VISIBLE_DEVICES=1 -e KINDI_STATE_DIR=/data/state-gpu1 \
    -p 127.0.0.1:8094:8093 -v $HOME/.kindi:/data kindi:0.7.0
```

Options go in as environment variables (`-e KINDI_CTX=16384`) or in `/data/kindi.conf`.

## Serve your local network

The server listens on `127.0.0.1` unless you publish its port to the network. Keep `--gpus all` (or the AMD devices)
when you change the port options. A container started without `--gpus` sees no GPU: Kindi then runs on the CPU, much
more slowly, and its log says `WARNING: no GPU is visible`. `curl -s localhost:8093/health` shows `"device": "gpu"`
when the card is in use.

Publish the port on all interfaces, with an API key:

```sh
docker run -d --name kindi --restart unless-stopped --gpus all -u $(id -u):$(id -g) \
    -p 8093:8093 -e KINDI_API_KEY=<long random string> -v $HOME/.kindi:/data kindi:0.7.0
```

Or use the host's network (no port mapping; set the port with `KINDI_PORT`):

```sh
docker run -d --name kindi --restart unless-stopped --gpus all -u $(id -u):$(id -g) \
    --network host -e KINDI_PORT=8093 -e KINDI_API_KEY=<long random string> -v $HOME/.kindi:/data kindi:0.7.0
```

Other computers then use `http://<this computer's address>:8093/v1` and send the key as
`Authorization: Bearer <key>`. The installer does all of this with `bash install.sh --lan` ([INSTALL.md](INSTALL.md)):
it keeps the GPU options, makes the key and prints the addresses.

- Run one server on a card that does not drive your display, if you can. On a desktop, the card that drives the
  screen also renders the desktop; a model that fills its memory can make the desktop slow or glitchy.
- Allow the port in the firewall for your local network only. Do not forward it from your router.
- We tested both recipes on a Linux host with an RTX 3080 (2B model, a request from another computer of the network).

## pi and other agents beside the container

pi (the coding agent that Kindi Bridge uses) talks to Kindi through a provider of its own in `~/.pi/agent/models.json`.
The installer prints this entry for each server:

```json
{"providers": {
  "kindi": {"baseUrl": "http://127.0.0.1:8093/v1", "api": "openai-completions", "apiKey": "none",
    "compat": {"supportsDeveloperRole": false, "supportsReasoningEffort": false,
               "thinkingFormat": "qwen-chat-template", "thinkingTokenBudgetField": "thinking_budget_tokens"},
    "models": [{"id": "bonsai2-27b", "name": "bonsai2-27b", "reasoning": true, "input": ["text"],
                "contextWindow": 24576, "maxTokens": 4096}]}
}}
```

Then start pi with `pi --provider kindi --model bonsai2-27b`, or pick the model in pi.

- Do not pick pi's built-in **OpenAI** provider: it sends the requests to OpenAI's own servers (with the key `none`,
  they answer 401), not to Kindi. Kindi serves the chat completions API (`"api": "openai-completions"`), not
  OpenAI's newer Responses API.
- The model id can be any name: Kindi answers with the model it runs. `curl -s 127.0.0.1:8093/v1/models` shows its
  own name.
- With the default `-p 127.0.0.1:8093:8093`, the server answers on the IPv4 address only. `127.0.0.1` always works;
  `localhost` worked too in our test (pi 1.3.0, Node.js 22, which tries IPv4 after IPv6).
- pi in a container of its own: `127.0.0.1` is that container, not the host. Run it with `--network host` (tested), or
  give it the host's network address and publish Kindi's port to the network as above.
- With an API key, put the key in `apiKey`.

We tested these settings with pi 1.3.0 against Kindi 0.6.0 on the host, beside the container.

## Podman

The images run under Podman too. The NVIDIA cards go in through CDI: make the CDI file once with
`sudo nvidia-ctk cdi generate --output /etc/cdi/nvidia.yaml`, then use `--device nvidia.com/gpu=all` in place of
`--gpus all`. In rootless Podman, use `-u 0:0` (the container's root is your user). For example:

```sh
podman run -d --name kindi --device nvidia.com/gpu=all -u 0:0 \
    -p 127.0.0.1:8093:8093 -v $HOME/.kindi:/data:Z kindi:0.7.0
```

`podman build -t kindi:0.7.0 -f docker/Dockerfile .` builds the image. Compose files and AMD: see
[CONTAINERS.md](CONTAINERS.md#podman). GPU access in rootless Podman is best effort; the installer uses Docker. We
have not tested every Podman version.

## A NAS or a home server

A NAS that runs Docker containers can run Kindi if it can give a container an NVIDIA card (the NVIDIA driver and
container toolkit on the NAS) or the AMD devices. What Kindi needs:

- the GPU in the container (`docker run --rm --gpus all ubuntu nvidia-smi` shows the card), or `KINDI_DEVICE=cpu` for
  the CPU engine (2B, 4B, 9B);
- a folder for `/data` on a fast disk: the model file and its pack take from about 2 GB (2B) to about 24 GB (27B),
  and the first start writes the pack;
- enough RAM, and a container memory limit that leaves room for long prompts (a user's server was restarted by its
  container limit at 60K-token prompts);
- the port published to your network, with an API key (above).

Container managers on NAS systems usually offer the same settings as fields: the image (`kindi:0.7.0`, built on
the NAS or another computer), the GPU, the volume for `/data`, the port and the environment variables. We have not
tested a NAS ourselves; reports are welcome on GitHub.
